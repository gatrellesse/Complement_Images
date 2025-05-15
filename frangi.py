import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt

from skimage.filters import frangi
from skimage import exposure, segmentation
from skimage.morphology import remove_small_objects, disk
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import local_maxima
from skimage.filters import threshold_otsu, sobel, gaussian
from skimage.segmentation import watershed
from skimage.morphology import skeletonize


def my_segmentation_frangi(img, img_mask):
    # Step 1: Normalize and enhance contrast
    img = img.astype(np.float32) / 255.0
    img = exposure.equalize_adapthist(img, clip_limit=0.03)

    # Step 2: Apply Frangi filter for vessel enhancement
    img = gaussian(img, sigma=0.8)
    vesselness = frangi(img, sigmas=np.arange(0.5, 6, 0.25), scale_range=None, scale_step=None, black_ridges=True,alpha=0.1, beta=0.5, gamma=50)
    vesselness *= img_mask  # Apply the mask

    # Step 3: Threshold the vesselness response
    thresh = threshold_otsu(vesselness)
    print('Threshold:', thresh)
    binary = vesselness > 9e-7

    # Step 4: Post-processing
    binary = remove_small_objects(binary, min_size=32)
    binary = ndi.binary_fill_holes(binary)

    # Step 5: Watershed to refine edges
    distance = ndi.distance_transform_edt(binary)
    local_max = local_maxima(distance)
    markers = label(local_max)
    labels = watershed(-distance, markers, mask=binary)

    img_out = (labels > 0)
    return img_out


def evaluate(img_out, img_GT):
    GT_skel = skeletonize(img_GT)
    img_out_skel = skeletonize(img_out)
    TP = np.sum(img_out_skel & img_GT)
    FP = np.sum(img_out_skel & ~img_GT)
    FN = np.sum(GT_skel & ~img_out)
    ACCU = TP / (TP + FP) if (TP + FP) > 0 else 0
    RECALL = TP / (TP + FN) if (TP + FN) > 0 else 0
    return ACCU, RECALL, img_out_skel, GT_skel


# Load image and GT
img = np.asarray(Image.open('./images_IOSTAR/star32_ODC.jpg')).astype(np.uint8)
nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
img_mask = np.ones(img.shape, dtype=bool)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 >= (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation_frangi(img, img_mask)
img_out[invalid_pixels] = 0

img_GT = np.asarray(Image.open('./images_IOSTAR/GT_32.png')).astype(np.uint8)

ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU, ', Recall =', RECALL)

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(232)
plt.imshow(img_out, cmap='gray')
plt.title('Frangi Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel, cmap='gray')
plt.title('Skeleton (Segmented)')
plt.subplot(235)
plt.imshow(img_GT, cmap='gray')
plt.title('Ground Truth')
plt.subplot(236)
plt.imshow(GT_skel, cmap='gray')
plt.title('Skeleton (GT)')
plt.tight_layout()
plt.show()
