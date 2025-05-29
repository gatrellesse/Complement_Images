import numpy as np
from PIL import Image
from pathlib import Path
import os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

from paper_segmentation import paper_segmentation
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
# Get the path of the current script
current_file = Path(__file__).resolve()
project_root = current_file.parent 
images_folder = project_root / 'images_IOSTAR'
water_precision, water_recall = [], []
paper_precision, paper_recall = [], []
frangi_precision, frangi_recall = [], []
for i in range(1,11):
    img_file = images_folder / f'img_{i}.jpg'
    img =  np.asarray(Image.open(img_file)).astype(np.uint8)
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
    gt_file = images_folder / f'GT_{i}.png'
    img_GT =  np.asarray(Image.open(gt_file)).astype(np.uint32)

    ACCU_water, RECALL_water, img_skel_water, GT_skel = evaluate(img_out_water, img_GT)
    ACCU_paper, RECALL_paper, img_skel_paper, GT_skel = evaluate(img_out_paper, img_GT)
    ACCU_frangi, RECALL_frangi, img_skel_frangi, GT_skel = evaluate(img_out_frangi, img_GT)
    water_precision.append(ACCU_water)
    water_recall.append(RECALL_water)
    paper_precision.append(ACCU_paper)
    paper_recall.append(RECALL_paper)
    frangi_precision.append(ACCU_frangi)
    frangi_recall.append(RECALL_frangi)
    print(f'Image {i} processed')
    # print('Watershed: Accuracy =', ACCU_water, ', Recall =', RECALL_water)
    # print('Paper: Accuracy =', ACCU_paper, ', Recall =', RECALL_paper)
    # print('Frangi: Accuracy =', ACCU_frangi, ', Recall =', RECALL_frangi)

    # plot_results(img, img_out_water, img_out_paper, img_out_frangi, img_GT,
    #              img_skel_water, img_skel_paper, img_skel_frangi, GT_skel)

# Precision Plot
x_vals = list(range(1, 11))

plt.figure(figsize=(10, 5))
plt.plot(x_vals, water_precision, label='Watershed', color='blue')
plt.plot(x_vals, paper_precision, label='Paper', color='green')
plt.plot(x_vals, frangi_precision, label='Frangi', color='red')
plt.xlabel('Image Index')
plt.ylabel('Precision')
plt.title('Precision per Image')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Recall Plot
plt.figure(figsize=(10, 5))
plt.plot(x_vals, water_recall, label='Watershed', color='blue')
plt.plot(x_vals, paper_recall, label='Paper', color='green')
plt.plot(x_vals, frangi_recall, label='Frangi', color='red')
plt.xlabel('Image Index')
plt.ylabel('Recall')
plt.title('Recall per Image')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
