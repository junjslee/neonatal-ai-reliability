### RUN ###
# python workspace/jun/nec_lat/external_data_preprocessing.py --dicom_folder workspace/changhyun/nec_ch/PnuemoperiT_external/external_lateral_right_dcm --day 20250401 --temp_input png1 --output_cpu 24

import numpy as np
import cv2
import os
from glob import glob
import argparse
import shutil
from PIL import Image
from tqdm import tqdm
import albumentations as albu
import pydicom
import pydicom.pixel_data_handlers
import skimage.io
import skimage.util
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from monai.transforms import *
import multiprocessing

import numpy as np
import cv2
import os
from glob import glob
import argparse
import shutil
from PIL import Image
from tqdm import tqdm
import albumentations as albu
import pydicom
import pydicom.pixel_data_handlers
import skimage.io
import skimage.util
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from monai.transforms import *
import multiprocessing

def process_row_crop(dicom_e, padding_ratio=0.05):
    """Crop the image based on the largest contour with padding"""
    dcm_img_8u = cv2.normalize(dicom_e, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(dcm_img_8u, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = w + 2 * pad_w
    h = h + 2 * pad_h

    return x, y, w, h


def process_row_crop_coords(img, x, y, w, h):
    return img[y:y+h, x:x+w]   

def process_row_angle(cropped_img):
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

    h_new, w_new = cropped_img.shape[:2]
    center_new = (w_new // 2, h_new // 2)
    M_new = cv2.getRotationMatrix2D(center_new, angle_new, 1.0)
    
    return M_new, angle_new, h_new, w_new, threshold

def process_row_angle_ok(cropped_img, M_new, angle_new, h_new, w_new):
    rotated_img_new = cv2.warpAffine(cropped_img, M_new, (w_new, h_new), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_img_new

def normalize(image, option=False, **kwargs):
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if len(np.unique(image)) > 1:
        image -= image.min()
        image /= image.max()
    if option:
        image = (image - 0.5) / 0.5
    return image

def process_row_angle_ok_background(rotated_img, threshold):
    _, bin_rot = cv2.threshold(rotated_img, threshold, 255, cv2.THRESH_BINARY_INV)
    bin_rot = bin_rot.astype(np.uint8)
    kernel = np.ones((7,7),np.uint8)
    bin_rot = cv2.morphologyEx(bin_rot, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bin_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bin_rot)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    y, x = np.where(mask == 255)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    return xmin, xmax, ymin, ymax

def resize_and_padding_with_aspect_clahe(image, clipLimit=2.0, clip_min=0.5, clip_max=98.5):
    # Clip
    image = np.clip(image, a_min=np.percentile(image, clip_min), a_max=np.percentile(image, clip_max))
    image = image - np.min(image)
    if np.max(image) > 0:
        image = image / np.max(image)
    image = (image * 255).astype(np.uint8)

    # Clahe
    if clipLimit is not None:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
        image = clahe.apply(image)

    # Padding
    image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    
    # image = skimage.util.img_as_float32(image) * 255.0
    return image

def get_pixels_hu(img_dcm, img_dcm0, apply_voi=False, hu_threshold=None):
    try:
        img_dcm = apply_modality_lut(img_dcm, img_dcm0)
    except:
        img_dcm = img_dcm.astype(np.int16)
        intercept = img_dcm0.RescaleIntercept
        slope = img_dcm0.RescaleSlope
        if slope != 1:
            img_dcm = slope * img_dcm.astype(np.float64)
            img_dcm = img_dcm.astype(np.int16)
        img_dcm += np.int16(intercept)
        
    if apply_voi:
        img_dcm = apply_voi_lut(img_dcm, img_dcm0)
        
    if hu_threshold is not None:
        img_dcm[img_dcm < hu_threshold] = hu_threshold
        
    return np.array(img_dcm, dtype=np.int16)

def process_row_angle_ok_background_ok(img, xmin, xmax, ymin, ymax):
    return img[ymin:ymax+1, xmin:xmax+1] 
    
def resize_and_padding_with_aspect_clahe_temp_to_png(image, clipLimit=2.0, clip_min=0.5, clip_max=99.5):
    image = np.clip(image, a_min=np.percentile(image, clip_min), a_max=np.percentile(image, clip_max))
    image -= image.min()
    if image.max() != 0:
        image /= image.max()
    image = skimage.img_as_ubyte(image)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if clipLimit is not None:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        image = clahe.apply(image)
    image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    return image

def normalize_image(image, clip_min=0.0, clip_max=100.0):
    image = image.astype(np.float64)
    image -= image.min()
    max_val = image.max()
    if max_val != 0:
        image /= max_val
    image = skimage.util.img_as_float32(image)
    return (image * 255).astype(np.uint8)

def dicom_to_temp_png(dicom_path, output_folder, args):
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    # Read DICOM
    img_dcm__0 = pydicom.dcmread(dicom_path)
    img_dcm__0.BitsStored = 16
    img_dcm__ = img_dcm__0.pixel_array
    if img_dcm__0.PhotometricInterpretation == 'MONOCHROME1':
        img_dcm__ = img_dcm__.max() - img_dcm__
    img_dcm__ = get_pixels_hu(img_dcm__, img_dcm__0)
    
    filename_base = os.path.basename(dicom_path).replace('.dcm', '')
    filename = filename_base + ".png"
    
    special_skip_list = {"NEC_0914_H06_C06_GA01_BW01_000005_004"}
    if filename_base in special_skip_list:
        print(f"Special case for {filename_base}: applying CLAHE/padding without cropping/rotation")
        # Apply CLAHE and padding without cropping/rotation
        img_special = resize_and_padding_with_aspect_clahe_temp_to_png(
            img_dcm__, 
            clipLimit=args.clalimit, 
            clip_min=args.clip_min, 
            clip_max=args.clip_max
        )
        # Save with the special tag so we can identify it later
        temp_output_path = os.path.join(output_folder, filename_base + "_special_temp.png")
        image_pil = Image.fromarray(img_special)
        image_pil.save(temp_output_path)
        return temp_output_path

    # Full processing for regular files:
    x, y, w, h = process_row_crop(img_dcm__)
    img_dcm__a = process_row_crop_coords(img_dcm__, x, y, w, h)
    M_new, angle_new, h_new, w_new, threshold = process_row_angle(img_dcm__a)
    print("Computed rotation angle:", angle_new)
    img_dcm__b = process_row_angle_ok(img_dcm__a, M_new, angle_new, h_new, w_new)
    xmin, xmax, ymin, ymax = process_row_angle_ok_background(img_dcm__b, threshold)
    img_dcm__c = process_row_angle_ok_background_ok(img_dcm__b, xmin, xmax, ymin, ymax)
    img_dcm__d = normalize_image(img_dcm__c)
    temp_output_path = os.path.join(output_folder, filename_base + "_temp.png")
    image_pil = Image.fromarray(img_dcm__d)
    image_pil.save(temp_output_path)
    return temp_output_path

def process_and_save_png(png_path, output_folder, clip_min=0.0, clip_max=100.0, clalimit=None):
    img_dcm__c = skimage.io.imread(png_path)    
    img_dcm__d = resize_and_padding_with_aspect_clahe_temp_to_png(
        image=img_dcm__c, 
        clipLimit=clalimit, 
        clip_min=clip_min, 
        clip_max=clip_max
    )
    final_output_path = os.path.join(output_folder, os.path.basename(png_path).replace('_temp', ''))
    image_pil = Image.fromarray(img_dcm__d.astype(np.uint8))
    image_pil.save(final_output_path)
    return final_output_path 

# New top-level functions for multiprocessing to avoid lambda pickling issues
def process_dicom_file(args_tuple):
    dicom_path, temp_output_folder = args_tuple
    return dicom_to_temp_png(dicom_path, temp_output_folder)

def process_png_file(args_tuple):
    png_path, final_output_folder, clip_min, clip_max, clalimit = args_tuple
    return process_and_save_png(png_path, final_output_folder, clip_min, clip_max, clalimit)

def main():
    parser = argparse.ArgumentParser(description="DICOM to PNG Preprocessing Script")
    parser.add_argument('--dicom_folder', type=str, default='/home/brody9512/workspace/changhyun/nec_ch/nec_external_example/',
                        help='Folder containing .dcm files')
    parser.add_argument('--day', type=str, default='240306',
                        help='Day tag for output folder naming')
    parser.add_argument('--temp_input', type=str, default='png1',
                        help='Temporary stage folder name suffix')
    parser.add_argument('--clip_min', type=float, default=0.0,
                        help='Lower percentile for clipping')
    parser.add_argument('--clip_max', type=float, default=100.0,
                        help='Upper percentile for clipping')
    parser.add_argument('--clalimit', type=float, default=2.0,
                        help='CLAHE limit')
    parser.add_argument('--output_cpu', type=int, default=24,
                        help='Number of CPUs for multiprocessing')
    args = parser.parse_args()
    
    dicom_folder = args.dicom_folder
    day = args.day
    
    # Define temporary and final output folders
    temp_output_folder = f"/workspace/yeonsu/0.Projects/Pneumoperitoneum/temp_jun_preprocessiing/external_png_temp_on_{day}/"
    if os.path.exists(temp_output_folder):
        shutil.rmtree(temp_output_folder)
    os.makedirs(temp_output_folder, exist_ok=True)
    
    final_output_folder = f"/workspace/yeonsu/0.Projects/Pneumoperitoneum/temp_jun_preprocessiing/external_png_preprocessed_on_{day}/"
    if os.path.exists(final_output_folder):
        shutil.rmtree(final_output_folder)
    os.makedirs(final_output_folder, exist_ok=True)
    
    print(f"Step1: Converting .dcm -> _temp.png in: {temp_output_folder}")
    dicom_paths = glob(os.path.join(dicom_folder, '*.dcm'))
    print(f"Found {len(dicom_paths)} DICOM files.")
    
    pool = multiprocessing.Pool(processes=args.output_cpu)
    # Use starmap to pass (dicom_path, temp_output_folder, args) as a tuple
    dicom_args = [(path, temp_output_folder, args) for path in dicom_paths]
    pool.starmap(dicom_to_temp_png, dicom_args)
    pool.close()
    pool.join()
    
    print(f"Step2: Converting _temp.png -> final PNG in: {final_output_folder}")
    # Process regular temp PNG files (files with '_temp.png' but not those with '_special_temp.png')
    regular_temp_files = glob(os.path.join(temp_output_folder, '*_temp.png'))
    regular_temp_files = [p for p in regular_temp_files if not p.endswith('_special_temp.png')]
    print(f"Found {len(regular_temp_files)} regular temp PNG files.")
    
    pool2 = multiprocessing.Pool(processes=args.output_cpu)
    # Since process_png_file takes a single tuple argument, use pool2.map.
    png_args = [(png_path, final_output_folder, args.clip_min, args.clip_max, args.clalimit) for png_path in regular_temp_files]
    pool2.map(process_png_file, png_args)
    pool2.close()
    pool2.join()
    
    # Now, copy over the special temp files directly into the final folder.
    special_temp_files = glob(os.path.join(temp_output_folder, '*_special_temp.png'))
    print(f"Found {len(special_temp_files)} special temp PNG files.")
    for special_file in special_temp_files:
        basename = os.path.basename(special_file)
        # Optionally remove '_special_temp' from the name if you want uniformity:
        final_name = basename.replace('_special_temp', '')
        final_path = os.path.join(final_output_folder, final_name)
        shutil.copy(special_file, final_path)
        print(f"Copied special file {basename} to final folder as {final_name}")
    
    print("\nAll done. Final PNGs are in:", final_output_folder)

if __name__ == '__main__':
    main()

#########################################
# Updated function to determine cropping area using border flood fill 
#########################################
# def process_row_angle_ok_background(rotated_img, threshold, padding=10):
#     """
#     This function uses flood fill to determine the background, then calculates a bounding box
#     for the foreground (the baby’s body) from the rotated image.
#     It works along both the X and Y axes.
    
#     Additionally, if the image is horizontal (i.e. baby lying on its side), it adds extra horizontal padding.
#     """
#     # Ensure image is uint8 for floodFill
#     if rotated_img.dtype != np.uint8:
#         img_u8 = cv2.normalize(rotated_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     else:
#         img_u8 = rotated_img.copy()
    
#     # Use the provided threshold and create a binary mask
#     # (We use THRESH_BINARY_INV so that low-intensity (air) regions become white.)
#     _, otsu_mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)

#     # Invert for background detection
#     background_mask = cv2.bitwise_not(otsu_mask)

    
#     # Create a mask for floodFill; dimensions must be 2 pixels larger than image.
#     h, w = background_mask.shape
#     flood_mask = np.zeros((h+2, w+2), np.uint8)
#     filled = otsu_mask.copy()

#     flood_fill_mask = np.zeros((h+2, w+2), np.uint8)
#     flood_filled = background_mask.copy()
#     cv2.floodFill(flood_filled, flood_fill_mask, (0,0), 255)
#     # Invert back
#     flood_filled = cv2.bitwise_not(flood_filled)
#     # Combine to find foreground mask
#     foreground_mask = cv2.bitwise_and(otsu_mask, flood_filled)
    
#     # # Flood fill from all four corners. We assume that the plate/background touches the image borders.
#     # cv2.floodFill(filled, flood_mask, (0, 0), 255)
#     # cv2.floodFill(filled, flood_mask, (w-1, 0), 255)
#     # cv2.floodFill(filled, flood_mask, (0, h-1), 255)
#     # cv2.floodFill(filled, flood_mask, (w-1, h-1), 255)
#     # # Invert flood fill result: now foreground (body) becomes white.
#     # foreground = cv2.bitwise_not(filled)
#     # # Optionally, smooth the mask with a small morphology closing.
#     # kernel = np.ones((5, 5), np.uint8)
#     # foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    
#     # Find all nonzero pixels (foreground)
#     ys, xs = np.where(foreground_mask > 0)
#     if len(xs) == 0 or len(ys) == 0:
#         return 0, w-1, 0, h-1
#     xmin, xmax = int(xs.min()), int(xs.max())
#     ymin, ymax = int(ys.min()), int(ys.max())
    
#     # Apply padding to avoid cutting off important edges:
#     xmin = max(xmin - padding, 0)
#     xmax = min(xmax + padding, w-1)
#     ymin = max(ymin - padding, 0)
#     ymax = min(ymax + padding, h-1)
    
#     # Additional logic if the image is horizontal (baby is sideways)
#     aspect = (xmax - xmin) / (ymax - ymin) if (ymax - ymin) > 0 else 1.0
#     if aspect > 1.5:  # Adjust threshold based on your empirical data
#         extra_pad = int((xmax - xmin) * 0.15)
#         xmin = max(xmin - extra_pad, 0)
#         xmax = min(xmax + extra_pad, w-1)
    
#     return xmin, xmax, ymin, ymax

# #########################################
# # Other functions remain mostly the same:
# #########################################
# def process_row_crop(dicom_e, padding_ratio=0.05):
#     """Crop the image based on the largest contour with padding"""
#     dcm_img_8u = cv2.normalize(dicom_e, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     _, binary = cv2.threshold(dcm_img_8u, 50, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contour = max(contours, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(contour)

#     pad_w = int(w * padding_ratio)
#     pad_h = int(h * padding_ratio)

#     x = max(0, x - pad_w)
#     y = max(0, y - pad_h)
#     w = w + 2 * pad_w
#     h = h + 2 * pad_h

#     return x, y, w, h

# def process_row_crop_coords(img, x, y, w, h):
#     return img[y:y+h, x:x+w]   

# def process_row_angle(cropped_img):
#     threshold = np.mean(cropped_img)
#     if threshold < 80:
#         threshold += 40
#     elif 80 < threshold < 100:
#         threshold += 40
#     elif 100 < threshold < 110:
#         threshold += 20
#     elif 110 < threshold < 130:
#         threshold += 10
#     elif 140 < threshold < 150:
#         threshold -= 10
#     elif 150 < threshold < 160:
#         threshold -= 20
#     elif threshold > 160:
#         threshold -= 30

#     _, binary_img = cv2.threshold(cropped_img, threshold, 255, cv2.THRESH_BINARY_INV)
#     binary_img = binary_img.astype(np.uint8)
#     kernel = np.ones((7,7), np.uint8)
#     binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
#     contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contour_new = max(contours, key=cv2.contourArea)

#     rect_new = cv2.minAreaRect(contour_new)
#     angle_new = rect_new[-1]
#     if 45 < angle_new < 95:
#         angle_new -= 90
#     elif 5 < angle_new < 45:
#         angle_new -= angle_new/2
#     elif -45 < angle_new < -5:
#         angle_new += angle_new/2

#     h_new, w_new = cropped_img.shape[:2]
#     center_new = (w_new // 2, h_new // 2)
#     M_new = cv2.getRotationMatrix2D(center_new, angle_new, 1.0)
    
#     return M_new, angle_new, h_new, w_new, threshold

# def process_row_angle_ok(cropped_img, M_new, angle_new, h_new, w_new):
#     rotated_img_new = cv2.warpAffine(cropped_img, M_new, (w_new, h_new), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated_img_new

# def normalize(image, option=False, **kwargs):
#     if image.dtype != np.float32:
#         image = image.astype(np.float32)
#     if len(np.unique(image)) > 1:
#         image -= image.min()
#         image /= image.max()
#     if option:
#         image = (image - 0.5) / 0.5
#     return image

# def resize_and_padding_with_aspect_clahe(image, clipLimit=2.0, clip_min=0.5, clip_max=98.5):
#     image = np.clip(image, a_min=np.percentile(image, clip_min), a_max=np.percentile(image, clip_max))
#     image = image - np.min(image)
#     if np.max(image) > 0:
#         image = image / np.max(image)
#     image = (image * 255).astype(np.uint8)

#     if clipLimit is not None:
#         clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
#         image = clahe.apply(image)
    
#     image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    
#     return image

# def get_pixels_hu(img_dcm, img_dcm0, apply_voi=False, hu_threshold=None):
#     try:
#         img_dcm = apply_modality_lut(img_dcm, img_dcm0)
#     except:
#         img_dcm = img_dcm.astype(np.int16)
#         intercept = img_dcm0.RescaleIntercept
#         slope = img_dcm0.RescaleSlope
#         if slope != 1:
#             img_dcm = slope * img_dcm.astype(np.float64)
#             img_dcm = img_dcm.astype(np.int16)
#         img_dcm += np.int16(intercept)
#     if apply_voi:
#         img_dcm = apply_voi_lut(img_dcm, img_dcm0)
        
#     if hu_threshold is not None:
#         img_dcm[img_dcm < hu_threshold] = hu_threshold
        
#     return np.array(img_dcm, dtype=np.int16)

# def process_row_angle_ok_background_ok(img, xmin, xmax, ymin, ymax, pad=10):
#     """
#     Crops an image using the specified bounding box (xmin, xmax, ymin, ymax),
#     and adds extra padding (default 10 pixels) on all sides, not exceeding the image boundaries.
#     """
#     h, w = img.shape

#     # Compute new bounding box with extra padding.
#     xmin_new = max(0, xmin - pad)
#     xmax_new = min(w - 1, xmax + pad)
#     ymin_new = max(0, ymin - pad)
#     ymax_new = min(h - 1, ymax + pad)

#     return img[ymin_new:ymax_new+1, xmin_new:xmax_new+1]

# def resize_and_padding_with_aspect_clahe_temp_to_png(image, clipLimit=2.0, clip_min=0.5, clip_max=99.5):
#     image = np.clip(image, a_min=np.percentile(image, clip_min), a_max=np.percentile(image, clip_max))
#     image -= image.min()
#     if image.max() != 0:
#         image /= image.max()
#     image = skimage.img_as_ubyte(image)
#     if len(image.shape) > 2:
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     if clipLimit is not None:
#         clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
#         image = clahe.apply(image)
#     image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
#     return image

# def normalize_image(image, clip_min=0.0, clip_max=100.0):
#     image = image.astype(np.float64)
#     image -= image.min()
#     max_val = image.max()
#     if max_val != 0:
#         image /= max_val
#     image = skimage.util.img_as_float32(image)
#     return (image * 255).astype(np.uint8)

# #########################################
# # Updated dicom_to_temp_png using the new cropping method:
# #########################################
# def dicom_to_temp_png(dicom_path, output_folder, args):
#     import pydicom
#     from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
#     # Read DICOM
#     img_dcm__0 = pydicom.dcmread(dicom_path)
#     img_dcm__0.BitsStored = 16
#     img_dcm__ = img_dcm__0.pixel_array
#     if img_dcm__0.PhotometricInterpretation == 'MONOCHROME1':
#         img_dcm__ = img_dcm__.max() - img_dcm__
#     img_dcm__ = get_pixels_hu(img_dcm__, img_dcm__0)
    
#     filename_base = os.path.basename(dicom_path).replace('.dcm', '')
#     filename = filename_base + ".png"
    
#     special_skip_list = {
#         "NEC_0914_H06_C06_GA01_BW01_000005_004", 
#         "NEC_0914_H06_C06_GA01_BW01_000023_007", 
#         "NEC_0914_H06_C06_GA02_BW03_000018_012", 
#         "NEC_0914_H06_C06_GA01_BW01_000005_015", 
#         "NEC_0914_H06_C06_GA02_BW03_000018_002", 
#         "NEC_0914_H06_C06_GA02_BW03_000018_002",
#         "NEC_0914_H06_C06_GA03_BW03_000020_006",
#         "NEC_1016_H04_C06_GA01_BW01_000045_015__0000",
#         "NEC_1016_H04_C06_GA01_BW01_000066_003__0000",
#         "NEC_1016_H04_C06_GA01_BW01_000066_009__0000",
#         "NEC_1016_H04_C06_GA01_BW01_000066_011__0000",
#         "NEC_1016_H04_C06_GA01_BW01_000066_013__0000",
#         "NEC_1016_H04_C06_GA02_BW01_000010_009__0000",
#         "NEC_1016_H04_C06_GA02_BW02_000012_005__0000",
#         "NEC_1016_H04_C06_GA02_BW02_000035_003__0000",
#         "NEC_1016_H04_C06_GA02_BW02_000035_004__0000",
#         "NEC_1026_H04_C06_GA01_BW01_000066_009__0000",
#         "NEC_1026_H04_C06_GA01_BW01_000066_013__0000",
#         "NEC_1026_H04_C06_GA02_BW01_000010_009__0000",
#         "NEC_1026_H04_C06_GA02_BW02_000035_003__0000",
#         "NEC_1026_H04_C06_GA02_BW02_000035_004__0000",
#         "NEC_1103_H04_C06_GA02_BW03_000011_003__0000",
#     }
#     if filename_base in special_skip_list:
#         print(f"Special case for {filename_base}: applying CLAHE/padding without cropping/rotation")
#         # Apply CLAHE and padding without cropping/rotation
#         img_special = resize_and_padding_with_aspect_clahe_temp_to_png(
#             img_dcm__, 
#             clipLimit=args.clalimit, 
#             clip_min=args.clip_min, 
#             clip_max=args.clip_max
#         )
#         # Save with the special tag so we can identify it later
#         temp_output_path = os.path.join(output_folder, filename_base + "_special_temp.png")
#         image_pil = Image.fromarray(img_special)
#         image_pil.save(temp_output_path)
#         return temp_output_path

#     # Full processing for regular files:
#     x, y, w, h = process_row_crop(img_dcm__)
#     img_dcm__a = process_row_crop_coords(img_dcm__, x, y, w, h)
#     M_new, angle_new, h_new, w_new, threshold = process_row_angle(img_dcm__a)
#     print("Computed rotation angle:", angle_new)
#     img_dcm__b = process_row_angle_ok(img_dcm__a, M_new, angle_new, h_new, w_new)
#     xmin, xmax, ymin, ymax = process_row_angle_ok_background(img_dcm__b, threshold)
#     img_dcm__c = process_row_angle_ok_background_ok(img_dcm__b, xmin, xmax, ymin, ymax, pad=25)
#     img_dcm__d = normalize_image(img_dcm__c)
#     temp_output_path = os.path.join(output_folder, filename_base + "_temp.png")
#     image_pil = Image.fromarray(img_dcm__d)
#     image_pil.save(temp_output_path)
#     return temp_output_path

# #########################################
# # Helper functions for multiprocessing
# #########################################
# def process_and_save_png(png_path, output_folder, clip_min=0.0, clip_max=100.0, clalimit=None):
#     img_dcm__c = skimage.io.imread(png_path)    
#     img_dcm__d = resize_and_padding_with_aspect_clahe_temp_to_png(
#         image=img_dcm__c, 
#         clipLimit=clalimit, 
#         clip_min=clip_min, 
#         clip_max=clip_max
#     )
#     final_output_path = os.path.join(output_folder, os.path.basename(png_path).replace('_temp', ''))
#     image_pil = Image.fromarray(img_dcm__d.astype(np.uint8))
#     image_pil.save(final_output_path)
#     return final_output_path 

# def process_dicom_file(args_tuple):
#     dicom_path, temp_output_folder, args = args_tuple
#     return dicom_to_temp_png(dicom_path, temp_output_folder, args)

# def process_png_file(args_tuple):
#     png_path, final_output_folder, clip_min, clip_max, clalimit = args_tuple
#     return process_and_save_png(png_path, final_output_folder, clip_min, clip_max, clalimit)

# #########################################
# # Main function integrating the new cropping method
# #########################################
# def main():
#     parser = argparse.ArgumentParser(description="DICOM to PNG Preprocessing Script")
#     parser.add_argument('--dicom_folder', type=str, default='/home/brody9512/workspace/changhyun/nec_ch/nec_external_example/',
#                         help='Folder containing .dcm files')
#     parser.add_argument('--day', type=str, default='240306',
#                         help='Day tag for output folder naming')
#     parser.add_argument('--temp_input', type=str, default='png1',
#                         help='Temporary stage folder name suffix')
#     parser.add_argument('--clip_min', type=float, default=0.5,
#                         help='Lower percentile for clipping (in percentile format)')
#     parser.add_argument('--clip_max', type=float, default=98.5,
#                         help='Upper percentile for clipping (in percentile format)')
#     parser.add_argument('--clalimit', type=float, default=2.0,
#                         help='CLAHE limit')
#     parser.add_argument('--output_cpu', type=int, default=24,
#                         help='Number of CPUs for multiprocessing')
#     args = parser.parse_args()
    
#     dicom_folder = args.dicom_folder
#     day = args.day
    
#     # Define temporary and final output folders
#     temp_output_folder = f"/workspace/yeonsu/0.Projects/Pneumoperitoneum/External EDA/external_png_temp_on_{day}/"
#     if os.path.exists(temp_output_folder):
#         shutil.rmtree(temp_output_folder)
#     os.makedirs(temp_output_folder, exist_ok=True)
    
#     final_output_folder = f"/workspace/yeonsu/0.Projects/Pneumoperitoneum/External EDA/external_png_preprocessed_on_{day}/"
#     if os.path.exists(final_output_folder):
#         shutil.rmtree(final_output_folder)
#     os.makedirs(final_output_folder, exist_ok=True)
    
#     print(f"Step1: Converting .dcm -> _temp.png in: {temp_output_folder}")
#     dicom_paths = glob(os.path.join(dicom_folder, '*.dcm'))
#     print(f"Found {len(dicom_paths)} DICOM files.")
    
#     pool = multiprocessing.Pool(processes=args.output_cpu)
#     dicom_args = [(path, temp_output_folder, args) for path in dicom_paths]
#     pool.starmap(dicom_to_temp_png, dicom_args)
#     pool.close()
#     pool.join()
    
#     print(f"Step2: Converting _temp.png -> final PNG in: {final_output_folder}")
#     regular_temp_files = glob(os.path.join(temp_output_folder, '*_temp.png'))
#     regular_temp_files = [p for p in regular_temp_files if not p.endswith('_special_temp.png')]
#     print(f"Found {len(regular_temp_files)} regular temp PNG files.")
    
#     pool2 = multiprocessing.Pool(processes=args.output_cpu)
#     png_args = [(png_path, final_output_folder, args.clip_min, args.clip_max, args.clalimit) for png_path in regular_temp_files]
#     pool2.map(process_png_file, png_args)
#     pool2.close()
#     pool2.join()
    
#     # Now, copy over any special temp files if necessary.
#     special_temp_files = glob(os.path.join(temp_output_folder, '*_special_temp.png'))
#     print(f"Found {len(special_temp_files)} special temp PNG files.")
#     for special_file in special_temp_files:
#         basename = os.path.basename(special_file)
#         final_name = basename.replace('_special_temp', '')
#         final_path = os.path.join(final_output_folder, final_name)
#         shutil.copy(special_file, final_path)
#         print(f"Copied special file {basename} to final folder as {final_name}")
    
#     print("\nAll done. Final PNGs are in:", final_output_folder)

# if __name__ == '__main__':
#     main()

