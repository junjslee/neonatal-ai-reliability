import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from albumentations import Lambda as A_Lambda # Not needed
from monai.data import Dataset # Assuming torch.utils.data.Dataset or similar
import pandas as pd

class ExternalLateralDataset(Dataset):
    """
    Custom Dataset for classification using PNG images from an EXTERNAL source.
    Mirrors the preprocessing of RADDINO_LateralDatasetPNG (validation/test mode).
    Expects a DataFrame with columns: 'png_path', 'Binary_Label', and 'Orientation'.
    Deterministically flips images with 'L' orientation to 'R' before other transforms.
    Applies only necessary preprocessing (Resize, Normalize, ToTensor).
    Outputs 1-channel tensors by default.
    """
    def __init__(self, df: pd.DataFrame, args, training=False): # training flag is ignored, always False logic
        # --- Input Validation ---
        required_cols = ['png_path', 'Binary_Label', 'Orientation']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame for External Dataset must contain columns: {required_cols}")

        self.df = df.reset_index(drop=True)
        self.args = args
        # self.training = False # Explicitly set or ignore
        self.min_side = self.args.size # Assuming args.size is defined

        # --- Define transforms for external testing (validation/test mode logic) ---
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

        # 1) Load metadata
        row = self.df.iloc[idx]
        label = row['Binary_Label']
        png_path = row['png_path']
        orientation = row['Orientation']

        # 2) Load Image (Grayscale)
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or failed to load: {png_path}")

        # 3) Deterministic Flip based on Orientation (BEFORE transforms)
        if orientation == 'L':
            image = cv2.flip(image, 1) # 1 for horizontal flip

        # 4) Apply Albumentations transforms (Resize, Normalize, ToTensor).
        # Assumes model takes 1-channel input. If CNN needs 3 channels, stack here.
        data_transformed = self.transforms(image=image)
        final_img_tensor = data_transformed['image'] # Output is CxHxW (C=1 here)

        # --- Optional: Convert to 3 Channels if needed for a specific CNN ---
        # if final_img_tensor.shape[0] == 1:
        #     final_img_tensor = final_img_tensor.repeat(3, 1, 1) # Repeat channel dim
        # --------------------------------------------------------------------

        sample = {
            'image': final_img_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'png_name': os.path.basename(png_path)
        }
        return sample

    # --- Utility Method (Keep the same) ---
    def normalize(self, image, option=False, **kwargs):
        """Normalize image to [0,1] and optionally scale to [-1,1]."""
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        img_max = image.max()
        img_min = image.min()
        if img_max > img_min:
            image -= img_min
            image /= (img_max - img_min)
        elif img_max == 0: pass
        else: image[:] = 1.0
        if option: image = (image - 0.5) / 0.5
        return image