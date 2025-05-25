import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk, ellipse
from skimage import segmentation, filters, exposure
from skimage.morphology import binary_erosion,binary_dilation,binary_opening
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, local_maxima

from scipy.ndimage import rotate

def plot_histogram(img, title="Histogram"):
    plt.figure()
    plt.hist(img.ravel(), bins=256, range=(0, 1), color='gray')
    plt.title(title)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def my_segmentation(img, img_mask):
    # Step 1: Preprocess image
    img = img.astype(np.float32)
    img /= 255.0  # Normalize to [0, 1]
    img = exposure.equalize_adapthist(img, clip_limit=3)  # CLAHE
    # Step 2: Apply Gaussian smoothing to reduce noise
    img = filters.gaussian(img, sigma=0.40)
    img = filters.rank.median(img, disk(3))
    # Step 3: Apply black hat transformation
    # Black hat highlights dark structures (ves
    # sels) on light background
    black_hat_img = black_tophat(img, ellipse(2,1))  # Apply black tophat with disk kernel
    black_hat_img2 = black_tophat(img, rectangle(1,10))
    


    # Step 4: Gradient image (vessel edges show up as high gradient)
    gradient = filters.sobel(black_hat_img)
    gradient2 = filters.sobel(black_hat_img2)
    # Step 5: Threshold to get binary foreground
    #print('Gradient min:', gradient.min(), 'max:', gradient.max(), "treshold:", filters.threshold_otsu(gradient))
    binary = gradient > 0.95*filters.threshold_otsu(gradient)
    binary2 = gradient2 > filters.threshold_otsu(gradient2)
    # Combine the two binary images
    binary = (binary | binary2) 
    binary = remove_small_objects(binary, min_size=32)
    binary = ndi.binary_fill_holes(binary)
    # Step 6: Distance transform
    distance = ndi.distance_transform_edt(binary)

    # Step 7: Local maxima and markers
    local_maxi = local_maxima(distance)
    markers = ndi.label(local_maxi)[0]

    # Step 8: Watershed segmentation
    labels = segmentation.watershed(-distance, markers, mask=binary)

    # Step 9: Post-process
    img_out = (labels > 0) 
    
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
img =  np.asarray(Image.open('./images_IOSTAR/star08_OSN.jpg')).astype(np.uint8)
print(img.shape)


nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 >= (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask)
#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_08.png')).astype(np.uint8)

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

