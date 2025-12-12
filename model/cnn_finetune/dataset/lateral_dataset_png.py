import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.data import Dataset
import pandas as pd

class LateralDatasetPNG(Dataset):
    """
    Implements the exact same logic as RADDINO_LateralDatasetPNG, 
    with conditional training augmentations and deterministic L/R flips.
    Expects a DataFrame with ['png_path','Binary_Label','Orientation'].
    """
    def __init__(self, df: pd.DataFrame, args, training=True):
        # --- Input validation ---
        required_cols = ['png_path', 'Binary_Label', 'Orientation']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

        self.df = df.reset_index(drop=True)
        self.args = args
        self.training = training
        self.min_side = self.args.size

        # --- Build transforms ---
        if self.training:
            # Base spatial transforms
            if self.args.edge_augmentation:
                base_transforms = [
                    A.RandomResizedCrop(
                        height=self.min_side, width=self.min_side,
                        scale=(0.8, 0.9), ratio=(1.0, 1.0), p=1
                    ),
                    A.Rotate(limit=self.args.rotate_angle, p=self.args.rotate_percentage),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.0, rotate_limit=0.0,
                        p=0.8, border_mode=cv2.BORDER_CONSTANT
                    ),
                ]
            else:
                base_transforms = [
                    A.Resize(self.min_side, self.min_side, p=1),
                    A.Rotate(limit=self.args.rotate_angle, p=self.args.rotate_percentage),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.0, rotate_limit=0.0,
                        p=0.8, border_mode=cv2.BORDER_CONSTANT
                    ),
                ]

            # Intensity-based and sharpen
            other_augments = []
            if not self.args.original:
                other_augments.append(
                    # A.HorizontalFlip(p=args.horizontalflip_percentage),
                    A.Sharpen(
                        alpha=(0.1, 0.4), lightness=(0.5, 1.0),
                        p=self.args.sharp_percentage
                    )
                )
            # brightness/contrast
            other_augments.append(
                A.RandomBrightnessContrast(
                    brightness_limit=self.args.rbc_brightness,
                    contrast_limit=self.args.rbc_contrast,
                    p=self.args.rbc_percentage
                )
            )
            # optional gamma
            if self.args.gamma_truefalse:
                other_augments.append(
                    A.RandomGamma(
                        gamma_limit=(self.args.gamma_min, self.args.gamma_max),
                        p=self.args.gamma_percentage
                    )
                )
            # optional Gaussian noise
            if self.args.gaussian_truefalse:
                other_augments.append(
                    A.GaussNoise(
                        var_limit=(self.args.gaussian_min, self.args.gaussian_max),
                        p=self.args.gaussian_percentage
                    )
                )

            # Elastic + coarse dropout
            additional_transforms = []
            if self.args.elastic_truefalse:
                additional_transforms.append(
                    A.ElasticTransform(
                        alpha=self.args.elastic_alpha,
                        sigma=self.args.elastic_sigma,
                        alpha_affine=self.args.elastic_alpha_affine,
                        p=self.args.elastic_percentage
                    )
                )
            # always add coarse dropout
            additional_transforms.append(
                A.CoarseDropout(
                    max_holes=4,
                    max_height=8,
                    max_width=8,
                    fill_value=0,
                    p=0.5
                )
            )

            # assemble full list
            transforms_list = base_transforms + other_augments
            transforms_list.append(
                A.OneOf(additional_transforms, p=0.5)
            )
            transforms_list += [
                A.Lambda(image=self.normalize),
                ToTensorV2(),
            ]
        else:
            # validation / test
            transforms_list = [
                A.Resize(self.min_side, self.min_side, p=1),
                A.Lambda(image=self.normalize),
                ToTensorV2(),
            ]

        # drop any Nones and compose
        self.transforms = A.Compose([t for t in transforms_list if t is not None])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        label = row['Binary_Label']
        png_path = row['png_path']
        orientation = row['Orientation']

        # load grayscale
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {png_path}")

        # *** CHANGE: Deterministic flip based on Orientation BEFORE other transforms ***
        if orientation == 'L':
            # print(f"Flipping L image: {os.path.basename(png_path)}") # Optional: for debugging
            image = cv2.flip(image, 1) # 1 for horizontal flip

        # apply transforms
        data = self.transforms(image=image)
        img_tensor = data['image']

        return {
            'image': img_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'png_name': os.path.basename(png_path)
        }

    def normalize(self, image, option=False, **kwargs):
        """Normalize to [0,1] and optionally to [-1,1]."""
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        elif img_max == 0:
            pass
        else:
            image[:] = 1.0

        if option:
            image = (image - 0.5) / 0.5

        return image
