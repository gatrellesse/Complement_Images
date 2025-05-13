import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk, remove_small_objects, local_maxima
from skimage import io, segmentation, color
from skimage.measure import label
from skimage.filters import gaussian
from skimage.morphology import disk
from scipy import ndimage as ndi
from PIL import Image
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt


def my_segmentation(img, img_mask, seuil):
    # White hat transform
    footprint = np.ones((3, 3))
    #white_hat = white_tophat(img, footprint)

    # 1. Gradient
    img = gaussian(img, sigma = 0.5)
    gradient = filters.sobel(img)

    # 2. Threshold + mask
    binary = gradient > filters.threshold_otsu(gradient)

    # 3. Distance transform
    distance = ndi.distance_transform_edt(binary)

    # 4. Maxima + markers
    local_maxi = local_maxima(distance)
    markers = label(local_maxi)

    # 5. Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)
    
    # Option 1: Sortie binaire : pixels segmentés avec label > 0
    img_out = (labels > 0) & img_mask
    img_out = remove_small_objects(img_out, min_size = 32)
    # Option 2: seuil appliqué à l'image initiale (selon ton besoin)
    #img_out = img_out & (img < seuil)

    return img_out


def evaluate(img_out, img_GT):
    GT_skel = skeletonize(img_GT) # On reduit le support de l'evaluation...
    img_out_skel = skeletonize(img_out) # ...aux pixels des squelettes
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs

    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

#Ouvrir l'image originale en niveau de gris
img =  np.asarray(Image.open('./images_IOSTAR/star02_OSC.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask,100)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_02.png')).astype(np.uint32)

ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU,', Recall =', RECALL)

plt.subplot(231)
plt.imshow(img,cmap = 'gray')
plt.title('Image Originale')
plt.subplot(232)
plt.imshow(img_out)
plt.title('Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel)
plt.title('Segmentation squelette')
plt.subplot(235)
plt.imshow(img_GT)
plt.title('Verite Terrain')
plt.subplot(236)
plt.imshow(GT_skel)
plt.title('Verite Terrain Squelette')
plt.show()

