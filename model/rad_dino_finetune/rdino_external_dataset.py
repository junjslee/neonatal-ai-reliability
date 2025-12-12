import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from albumentations import Lambda as A_Lambda # Custom Lambda not strictly needed here
from monai.data import Dataset # Assuming this is torch.utils.data.Dataset or similar
import pandas as pd # Import pandas for type hint


class RADDINO_ExternalLateralDataset(Dataset):
    """
    Custom Dataset for classification using PNG images from an EXTERNAL source.
    Expects a DataFrame with columns: 'png_path', 'Binary_Label', and 'Orientation'.
    Deterministically flips images with 'L' orientation to 'R' before other transforms.
    """
    def __init__(self, df: pd.DataFrame, args, training=False):
        # *** CHANGE: Check for required columns ***
        required_cols = ['png_path', 'Binary_Label', 'Orientation']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame for External Dataset must contain columns: {required_cols}")

        # Reset index for potentially safer iloc access, although less critical if not using custom samplers
        self.df = df.reset_index(drop=True)
        self.args = args
        self.min_side = self.args.size

        # Define transforms for external testing (typically just resize, normalize, totensor)
        # No data augmentation should be applied here.
        transforms_list = [
            A.Resize(self.min_side, self.min_side, p=1),  # Resize image
            A.Lambda(image=self.normalize), # Use standard A.Lambda
            ToTensorV2()
        ]

        # Remove any None values (though none expected here)
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
        orientation = row['Orientation']

        # 2) Load PNG image using cv2.
        # Read in grayscale, adjust if your images are color (cv2.IMREAD_COLOR)
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or failed to load: {png_path}")

         # *** CHANGE: Deterministic flip based on Orientation BEFORE other transforms ***
        if self.args.flipped_orientation_during_training:
            if orientation == 'L':
                # print(f"Flipping L image: {os.path.basename(png_path)}") # Optional: for debugging
                image = cv2.flip(image, 1) # 1 for horizontal flip
                
        # 3) Apply Albumentations transforms (Resize, Normalize, ToTensor).
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
            # Optional: Include original orientation if needed for analysis downstream
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
