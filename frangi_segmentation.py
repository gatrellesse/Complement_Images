import numpy as np
import cv2 as cv
from skimage.filters import frangi
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity

def frangi_segmentation(img, img_mask):
    # Enhance contrast using CLAHE
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)

    image_norm = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

    smoothed = gaussian_filter(image_norm, sigma=1)

    vessels = frangi(smoothed, sigmas=(1, 3), scale_step=0.5)

    vessels_rescaled = rescale_intensity(vessels, in_range='image', out_range=(0, 1))

    vessels_norm = (vessels_rescaled * 255).astype(np.uint8)
    _, binary = cv.threshold(vessels_norm, 20, 255, cv.THRESH_BINARY)

    binary_cleaned = remove_small_objects(binary.astype(bool), min_size=30)

    img_out = (img_mask & binary_cleaned)
    return img_out