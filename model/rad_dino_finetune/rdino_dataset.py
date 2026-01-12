import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from albumentations import Lambda as A_Lambda # Not strictly needed for this change
from monai.data import Dataset # Assuming this is torch.utils.data.Dataset or similar
import pandas as pd # Import pandas for type hint

class RADDINO_LateralDatasetPNG(Dataset):
    def __init__(self, df: pd.DataFrame, args, training=True):
        # *** CHANGE: Check for required columns ***
        required_cols = ['png_path', 'Binary_Label', 'Orientation']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

        self.df = df.reset_index(drop=True)
        self.args = args
        self.training = training
        self.min_side = self.args.size

        # Build Albumentations transforms.
        if self.training:
            # Define transforms lists based on args
            if self.args.edge_augmentation:
                base_transforms = [
                    A.RandomResizedCrop(height=self.min_side, width=self.min_side, scale=(0.8, 0.9), ratio=(1.0, 1.0), p=1),
                    A.Rotate(limit=self.args.rotate_angle, p=self.args.rotate_percentage),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0.0, p=0.8, border_mode=cv2.BORDER_CONSTANT),
                ]
            else:
                 base_transforms = [
                    A.Resize(self.min_side, self.min_side, p=1),
                    A.Rotate(limit=self.args.rotate_angle, p=self.args.rotate_percentage),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0.0, p=0.8, border_mode=cv2.BORDER_CONSTANT),
                ]

            # Add other augmentations conditionally
            other_augments = []
            if not args.original:
                 other_augments.extend([
                    # *** CHANGE: Removed A.HorizontalFlip - flip is now deterministic based on Orientation ***
                    # A.HorizontalFlip(p=args.horizontalflip_percentage),
                    A.Sharpen(alpha=(0.1,0.4),lightness=(0.5,1.0),p=args.sharp_percentage),
                 ])
            if not args.flip:
                other_augments.extend([
                    A.HorizontalFlip(p=args.horizontalflip_percentage),
                ])

            other_augments.extend([
                A.RandomBrightnessContrast(brightness_limit=self.args.rbc_brightness, contrast_limit=self.args.rbc_contrast, p=self.args.rbc_percentage),
                A.RandomGamma(gamma_limit=(self.args.gamma_min, self.args.gamma_max), p=self.args.gamma_percentage) if self.args.gamma_truefalse else None,
                A.GaussNoise(var_limit=(self.args.gaussian_min, self.args.gaussian_max), p=self.args.gaussian_percentage) if self.args.gaussian_truefalse else None
            ])

            additional_transforms = []
            if self.args.elastic_truefalse:
                additional_transforms.append(A.ElasticTransform(alpha=self.args.elastic_alpha, sigma=self.args.elastic_sigma, alpha_affine=self.args.elastic_alpha_affine, p=self.args.elastic_percentage))
            additional_transforms.append(A.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, p=0.5))

            # Combine all transforms for training
            transforms_list = base_transforms + other_augments
            if additional_transforms:
                transforms_list.append(A.OneOf(additional_transforms, p=0.5))

            # Add final steps
            transforms_list.extend([
                A.Lambda(image=self.normalize),
                ToTensorV2(),
            ])

        else: # Validation/Testing transforms
            transforms_list = [
                A.Resize(self.min_side, self.min_side, p=1),
                A.Lambda(image=self.normalize),
                ToTensorV2(),
            ]

        # Remove any None values from the transform list.
        self.transforms = A.Compose([t for t in transforms_list if t is not None])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1) Load label, PNG path, and Orientation.
        row = self.df.iloc[idx]
        label = row['Binary_Label']
        png_path = row['png_path']
        # *** CHANGE: Get orientation ***
        orientation = row['Orientation']

        # 2) Load PNG image using cv2.
        # Read in grayscale, adjust if your images are color (cv2.IMREAD_COLOR)
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or failed to load: {png_path}")

        # *** CHANGE: Deterministic flip based on Orientation BEFORE other transforms ***
        if self.args.flip:
            if orientation == 'L':
                # print(f"Flipping L image: {os.path.basename(png_path)}") # Optional: for debugging
                image = cv2.flip(image, 1) # 1 for horizontal flip

        # 3) Apply Albumentations transforms to the (potentially flipped) image.
        # Ensure image is in the correct format (e.g., HWC for most albumentations)
        # If grayscale, might need to stack channels if subsequent transforms expect 3 channels
        # Example: if image.ndim == 2: image = np.stack([image]*3, axis=-1)

        data_transformed = self.transforms(image=image)
        final_img_tensor = data_transformed['image']

        # Ensure the output tensor has the correct channel dimension if needed
        # If grayscale and transforms didn't add channel dim:
        # if final_img_tensor.ndim == 2:
        #     final_img_tensor = final_img_tensor.unsqueeze(0) # Add channel dim -> [1, H, W]

        sample = {
            'image': final_img_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'png_name': os.path.basename(png_path)
            # Optional: You might want to include the original orientation for analysis
            # 'original_orientation': orientation
        }
        return sample

    # ---------------------------
    # Utility Methods
    # ---------------------------
    def normalize(self, image, option=False, **kwargs):
        """Normalize image to [0,1] and optionally scale to [-1,1]."""
        # Convert the image to float32 if it's not already
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Avoid division by zero if image is flat (e.g., all black)
        img_max = image.max()
        img_min = image.min()
        if img_max > img_min: # Check if image is not flat
            image -= img_min
            image /= (img_max - img_min) # Normalize to [0, 1]
        elif img_max == 0: # Handle all black image
             pass # Already 0
        else: # Handle flat non-black image (all pixels same value > 0)
             image[:] = 1.0 # Set all to 1

        if option:
            image = (image - 0.5) / 0.5 # Scale to [-1, 1]

        return image