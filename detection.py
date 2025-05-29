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

from paper_segmentation import LoG_filter, path_opening, paper_segmentation
from frangi_segmentation import frangi_segmentation
from watershed_segmentation import watershed_segmentation

def plot_results(img, img_out_water, img_out_paper, img_out_Frangi, img_GT,
                 img_skel_water, img_skel_paper, img_skel_Frangi, GT_skel):
    # --- First Figure: Segmentations ---
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.imshow(img_GT, cmap='gray')
    plt.title('GT Segmentation')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(img_out_water, cmap='gray')
    plt.title('Watershed')
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.imshow(img_out_paper, cmap='gray')
    plt.title('Paper')
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.imshow(img_out_Frangi, cmap='gray')
    plt.title('Frangivalue')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # --- Second Figure: Skeletons ---
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(GT_skel, cmap='gray')
    plt.title('GT Skeleton')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.imshow(img_skel_water, cmap='gray')
    plt.title('Watershed Skeleton')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(img_skel_paper, cmap='gray')
    plt.title('Paper Skeleton')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(img_skel_Frangi, cmap='gray')
    plt.title('Frangi Skeleton')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


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
img =  np.asarray(Image.open('./images_IOSTAR/img_08.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 >= (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out_water = watershed_segmentation(img,img_mask)
img_out_water[invalid_pixels] = 0

img_out_paper = paper_segmentation(img,img_mask)
img_out_paper[invalid_pixels] = 0

img_out_frangi = frangi_segmentation(img,img_mask)
img_out_frangi[invalid_pixels] = 0

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_08.png')).astype(np.uint32)

ACCU_water, RECALL_water, img_skel_water, GT_skel = evaluate(img_out_water, img_GT)
ACCU_paper, RECALL_paper, img_skel_paper, GT_skel = evaluate(img_out_paper, img_GT)
ACCU_frangi, RECALL_frangi, img_skel_frangi, GT_skel = evaluate(img_out_frangi, img_GT)
print('Watershed: Accuracy =', ACCU_water, ', Recall =', RECALL_water)
print('Paper: Accuracy =', ACCU_paper, ', Recall =', RECALL_paper)
print('Frangi: Accuracy =', ACCU_frangi, ', Recall =', RECALL_frangi)

plot_results(img, img_out_water, img_out_paper, img_out_frangi, img_GT,
             img_skel_water, img_skel_paper, img_skel_frangi, GT_skel)

