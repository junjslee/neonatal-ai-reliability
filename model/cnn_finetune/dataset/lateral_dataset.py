import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Lambda as A_Lambda

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from monai.data import Dataset
import skimage.io
import skimage.util

# For DICOM
from pydicom.pixel_data_handlers import gdcm_handler
pydicom.config.image_handlers = [gdcm_handler]

# Custom Lambda subclass for Albumentations.
class MyLambda(A_Lambda):
    def __call__(self, force_apply=False, **data):
        return super().__call__(**data)

class LateralDataset(Dataset):
    """
    Custom Dataset for classification using DICOM images.
    All segmentation/mask-related processing has been removed.
    Expects a DataFrame with columns: 'img_dcm' and 'Binary_Label'.
    """
    def __init__(self, df, args, training=True):
        self.df = df
        self.args = args
        self.training = training
        self.apply_voi = False
        self.hu_threshold = None
        self.clipLimit = self.args.clahe_cliplimit
        self.min_side = self.args.size
        
        # Build Albumentations transforms.
        self.transforms = None
        
        if self.training:
            transforms_list = [
                A.Resize(self.min_side, self.min_side, p=1),
                A.Rotate(limit=self.args.rotate_angle, p=self.args.rotate_percentage),
                # A.HorizontalFlip(p=0.42),
                A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.0,rotate_limit=0.0,p=0.8,border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(brightness_limit=self.args.rbc_brightness,contrast_limit=self.args.rbc_contrast,p=self.args.rbc_percentage),
                A.RandomGamma(gamma_limit=(self.args.gamma_min, self.args.gamma_max),p=self.args.gamma_percentage) if self.args.gamma_truefalse else None,
                A.GaussNoise(var_limit=(self.args.gaussian_min, self.args.gaussian_max),p=self.args.gaussian_percentage) if self.args.gaussian_truefalse else None
            ]
            
            additional_transforms = []
            if args.elastic_truefalse:
                additional_transforms.append(A.ElasticTransform(alpha=self.args.elastic_alpha, sigma=self.args.elastic_sigma, alpha_affine=self.args.elastic_alpha_affine, p=self.args.elastic_percentage))
            additional_transforms.append(A.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, p=0.5))
            
            if additional_transforms:
                transforms_list.append(A.OneOf(additional_transforms, p=0.5))
                
            transforms_list.extend([
                MyLambda(image=self.normalize),
                ToTensorV2(), # no comma?
            ])
        else:
            transforms_list = [
                A.Resize(self.min_side, self.min_side, p=1), # Resize image
                MyLambda(image=self.normalize),
                ToTensorV2(),
            ]
        
        self.transforms = A.Compose(transforms_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # ============ 1) Load label and image path ============
        label = self.df.iloc[idx]['Binary_Label']
        img_dcm_path = self.df.iloc[idx]['img_dcm']
        
        # ============ 2) Load DICOM ===============
        dicom_obj = pydicom.dcmread(img_dcm_path)
        # Force BitsStored = 16 if needed.
        dicom_obj.BitsStored = 16
        dcm_img = dicom_obj.pixel_array
        dcm_img = self.get_pixels_hu(dcm_img, dicom_obj)
        
        # ============ 3) Pre-processing & custom steps (crop/rotate) =======
        # Crop based on the largest contour in the image.
        x, y, w, h = self.process_row_crop(dcm_img)
        dcm_cropped = self.process_row_crop_coords(dcm_img, x, y, w, h)
        
        # Compute rotation matrix and rotate the cropped image.
        M_new, angle_new, hh, ww, threshold = self.process_row_angle(dcm_cropped)
        dcm_rotated = self.process_row_angle_ok(dcm_cropped, M_new, angle_new, hh, ww)
        
        # Background-based pre-processing
        # dcm_rotated_background = self.
        # dcm_rotated_background_ok = self.process_row_angle_ok_background(dcm_rotated, )

        # <--###-->
        # For classification, we omit the additional background-based ROI extraction.
        # Instead, we directly normalize and resize the rotated image.
        dcm_rotated = self.normalize(dcm_rotated, option=True)
        final_dcm = self.resize_and_padding_with_aspect_clahe(dcm_rotated)
                                
        # ============ 4) Albumentations (final) ==============
        data_transformed = self.transforms(image=final_dcm)
        dcm_final = data_transformed['image']

        # ============ 5) Prepare output dict =============
        sample = {
            'image': dcm_final, 
            'label': torch.tensor(label, dtype=torch.float32),
            'dcm_name': os.path.basename(img_dcm_path)
        }
        return sample
    
    # -------------------------------------------------------
    #         Dataset Utility Methods
    # -------------------------------------------------------
    def get_pixels_hu(self, pixel_array, dicom_obj):
        """Convert raw DICOM pixel array to Hounsfield Units (HU) if possible."""
        try:
            pixel_array = apply_modality_lut(pixel_array, dicom_obj)
        except:
            pixel_array = pixel_array.astype(np.int16)
            intercept = dicom_obj.RescaleIntercept
            slope = dicom_obj.RescaleSlope
            if slope != 1:
                pixel_array = slope * pixel_array.astype(np.float64)
                pixel_array = pixel_array.astype(np.int16)
            pixel_array += np.int16(intercept)
        if self.apply_voi:
            pixel_array = apply_voi_lut(pixel_array, dicom_obj)
        if self.hu_threshold is not None:
            pixel_array[pixel_array < self.hu_threshold] = self.hu_threshold
        return np.array(pixel_array, dtype=np.int16)

    def normalize(self, image, option=False, **kwargs):
        """Normalize image to [0,1] and optionally scale to [-1,1]."""
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(np.unique(image)) != 1:
            image -= image.min()
            image /= image.max()
        if option:
            image = (image - 0.5) / 0.5
        return image #.astype('float32')
    
    def resize_and_padding_with_aspect_clahe(self, image):
        """Clip the image, apply CLAHE, and pad to a square shape."""
        args = self.args
        image = np.clip(
            image,
            a_min=np.percentile(image, args.clip_min),
            a_max=np.percentile(image, args.clip_max)
        )
        image -= image.min()
        image /= image.max()
        image = skimage.img_as_ubyte(image)
        image = A.PadIfNeeded(
            min_height=max(image.shape),
            min_width=max(image.shape),
            always_apply=True,
            border_mode=0
        )(image=image)['image']
        if self.clipLimit is not None:
            clahe_obj = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(args.clahe_limit, args.clahe_limit))
            image = clahe_obj.apply(image)
        image = skimage.util.img_as_float32(image) * 255.0
        return image

    def process_row_crop(self, dcm_img):
        """Crop the image based on the largest contour."""
        dcm_img_8u = cv2.normalize(dcm_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U).astype(np.uint8)
        _, binary = cv2.threshold(dcm_img_8u, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, w, h

    def process_row_crop_coords(self, img, x, y, w, h):
        """Return the cropped region of the image."""
        return img[y:y+h, x:x+w]

    def process_row_angle(self, cropped_img):
        """Compute rotation parameters based on the cropped image."""
        threshold = np.mean(cropped_img)
        if threshold < 80:
            threshold += 40
        elif 80 < threshold < 100:
            threshold += 40
        elif 100 < threshold < 110:
            threshold += 20
        elif 110 < threshold < 130:
            threshold += 10
        elif 140 < threshold < 150:
            threshold -= 10
        elif 150 < threshold < 160:
            threshold -= 20
        elif threshold > 160:
            threshold -= 30

        _, binary_img = cv2.threshold(cropped_img, threshold, 255, cv2.THRESH_BINARY_INV)
        binary_img = binary_img.astype(np.uint8)
        kernel = np.ones((7,7), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_new = max(contours, key=cv2.contourArea)
        rect_new = cv2.minAreaRect(contour_new)
        angle_new = rect_new[-1]
        if 45 < angle_new < 95:
            angle_new -= 90
        elif 5 < angle_new < 45:
            angle_new -= angle_new/2
        elif -45 < angle_new < -5:
            angle_new += angle_new/2
        (h_new, w_new) = cropped_img.shape[:2] # h_new, w_new = cropped_img.shape[:2]
        center_new = (w_new // 2, h_new // 2)
        M_new = cv2.getRotationMatrix2D(center_new, angle_new, 1.0)
        return M_new, angle_new, h_new, w_new, threshold

    def process_row_angle_ok(self, img, M, angle, h, w):
        """Rotate the image using the computed rotation matrix."""
        rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated_img