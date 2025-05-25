import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, footprint_rectangle,rectangle, star, disk, remove_small_objects, local_maxima
from skimage.measure import label
from skimage.filters import gaussian, apply_hysteresis_threshold
from skimage.morphology import binary_erosion,binary_dilation,binary_opening
from skimage.feature import peak_local_max, hessian_matrix, hessian_matrix_eigvals
from skimage import  filters, exposure
from skimage import  segmentation
from skimage.util import invert

def LoG_filter(image, sigma, size=None):
    # Generate LoG kernel
    if size is None:
        size = int(6 * sigma + 1) if sigma >= 1 else 7

    if size % 2 == 0:
        size += 1

    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel
    # Perform convolution
    result = ndi.convolve(image, kernel)
    
    return result



def path_opening(img, length=20, angle_step=15):
    """
    Improved path opening for vessel detection
    
    Parameters:
    - img: input binary image
    - length: maximum length of paths to consider
    - angle_step: angular resolution (degrees) for directional structuring elements
    
    Returns:
    - Result of path opening operation
    """
    # Normalize input to [0,1]
    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    
    # Create structuring elements for multiple directions
    results = []
    angles = np.arange(0, 180, angle_step)
    
    for angle in angles:
        # Create line structuring element
        if angle == 0:
            se = footprint_rectangle((1, length))
        elif angle == 90:
            se = footprint_rectangle((length, 1))
        else:
            # For other angles, we'll create a rotated line
            theta = np.radians(angle)
            x = np.round(length/2 * np.cos(theta)).astype(int)
            y = np.round(length/2 * np.sin(theta)).astype(int)
            xx, yy = np.meshgrid(np.arange(-x, x+1), np.arange(-y, y+1))
            mask = np.abs(yy - xx * np.tan(theta)) < 0.5
            se = mask.astype(np.uint8)
        if np.sum(se) == 0:
            continue
        # Apply directional opening (erosion followed by dilation)
        opened = dilation(erosion(img, se), se)
        results.append(opened)
    
    # Combine results from all directions
    combined = np.maximum.reduce(results)
    
    # Post-processing to clean up small artifacts
    combined = opening(combined, disk(1))
    
    return combined

def compute_r(binary_image):
    labeled = label(binary_image)
    num_objects = np.max(labeled)
    num_pixels = np.sum(binary_image)
    return np.inf if num_pixels == 0 else num_objects / num_pixels

def find_Sopt(attribute_img, thresholds=np.linspace(0.01, 0.99, 100)):
    best_r = np.inf
    best_thresh = 0.5
    for t in thresholds:
        bin_img = attribute_img > t
        r = compute_r(bin_img)
        if r < best_r:
            best_r = r
            best_thresh = t
    return best_thresh

def hysteresis_thresholding(img, Sopt, extra_ratio=0.01):
    SH = Sopt
    target_count = np.sum(img > SH) * (1 + extra_ratio)
    sorted_vals = np.sort(img.ravel())[::-1]
    SL_index = np.searchsorted(np.cumsum(np.ones_like(sorted_vals)), target_count)
    SL = sorted_vals[min(SL_index, len(sorted_vals)-1)]
    return apply_hysteresis_threshold(img, SL, SH)

def paper_app(img, img_mask, seuil):
    dmax = 7
    img = filters.gaussian(img, sigma=0.5)
    g = opening(img, disk(1))
    
    plt.imsave('Gauss_Opening.png', img, cmap='gray')  # Use cmap='gray' for grayscale images
    g_opening = opening(g, disk(3))
    g_closing = closing(g, disk(4))

    # Toggle mapping filter
    diff_open = np.abs(g - g_opening)
    diff_close = np.abs(g_closing - g)
    
    # Create output image
    h = np.where(diff_close <= diff_open, g_closing, g_opening)
    
    # Attribute Extraction
    att0 = invert(h) - opening(invert(h), disk(dmax/2 + 2)) #background --> 0
    
    sigma1 = dmax/6
    sigma2 = dmax/3
    att1 = LoG_filter(att0, sigma1)
    att2 = LoG_filter(att0, sigma2)
    thresh1 = filters.threshold_otsu(att1)
    binary_att1 = att1 > thresh1
    thresh2 = filters.threshold_otsu(att2)
    binary_att2 = att2 > thresh2
    combined_att = np.maximum(att1, att2)
    rst1 = path_opening(combined_att, length=10, angle_step=45)
    threshrst = filters.threshold_otsu(rst1)
    binary_rst1 = rst1 > threshrst
    
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('original')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(g_opening, cmap='gray')
    plt.title('g_opening')
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.imshow(g_closing, cmap='gray')
    plt.title('g_closing')
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.imshow(g, cmap='gray')
    plt.title('g')
    plt.axis('off')

    plt.subplot(3, 4, 5)
    plt.imshow(h, cmap='gray')
    plt.title('h')
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(att0, cmap='gray')
    plt.title('att0')
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(np.logical_not(binary_att1), cmap='gray')
    plt.title('att1')
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.imshow(np.logical_not(binary_att2), cmap='gray')
    plt.title('att2')
    plt.axis('off')

    plt.subplot(3, 4, 9)
    plt.imshow(np.logical_not(binary_rst1), cmap='gray')
    plt.title('Rst1')
    plt.axis('off')
    binary_rst1 = np.logical_not(binary_rst1)
    Sopt1 = find_Sopt(att1)
    Sopt2 = find_Sopt(att2)
    seg_i_1 = hysteresis_thresholding(att1, Sopt1)
    seg_i_2 = hysteresis_thresholding(att2, Sopt2)
    Iseg0 = np.logical_or(seg_i_1, seg_i_2)
    
    # binary_rst1 = path_opening(Iseg0, length = dmax, angle_step=30)
    # binary_rst1 = ndi.binary_fill_holes(binary_rst1)
    # binary_rst1 = remove_small_objects(binary_rst1, min_size=16)
    binary_rst1 = remove_small_objects(np.logical_not(binary_att1), min_size=25)
    plt.subplot(3, 4, 10)
    plt.imshow(binary_rst1, cmap='gray')
    plt.title('Rst1_fill')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return binary_rst1


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

img_out = paper_app(img, img_mask, 100)
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

