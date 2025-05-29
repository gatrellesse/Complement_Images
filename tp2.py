import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk, remove_small_objects, local_maxima
from skimage.measure import label
from skimage.filters import gaussian
from skimage.morphology import binary_erosion,binary_dilation,binary_opening
from skimage.feature import peak_local_max, hessian_matrix, hessian_matrix_eigvals
from skimage import  filters, exposure
from skimage import  segmentation
from skimage.util import invert




def my_segmentation2(img, img_mask, seuil):
    # Step 1: Normalize image
    img = img.astype(np.float32) / 255.0
    # Step 2: Smooth the image (optional but helps noise reduction)
    img = exposure.equalize_adapthist(img, clip_limit=3)  # CLAHE
    img = filters.rank.median(img, disk(3))

    # Step 3: Hessian matrix and eigenvalues
    Hxx, Hxy, Hyy = hessian_matrix(img, sigma=1.50, order='rc')
    lambda1, lambda2 = hessian_matrix_eigvals([Hxx, Hxy, Hyy])
    # Step 4: Ridge-like vessel detection
    
    ridge_response = (lambda2 < 0) & (np.abs(lambda1) < 0.995 * np.abs(lambda2))
    binary = (ridge_response & img_mask) 

    # Step 5: Post-processing
    img_out = remove_small_objects(binary, min_size=32)
    img_out = ndi.binary_fill_holes(img_out)
    img_out = np.logical_not(img_out)

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
img =  np.asarray(Image.open('./images_IOSTAR/star32_ODC.jpg')).astype(np.uint8)
print(img.shape)
# img = img.astype(np.float32) / 255.0
# plot_histogram(img, title="Original Image Histogram")

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 >= (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation2(img,img_mask,100)
img_out[invalid_pixels] = 0
#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_32.png')).astype(np.uint32)

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

