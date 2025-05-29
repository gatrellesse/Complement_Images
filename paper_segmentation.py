import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import footprint_rectangle, disk, remove_small_objects
from skimage import  filters
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

def paper_segmentation(img, img_mask):
    dmax = 7
    img_original = img
    img = filters.gaussian(img, sigma=0.5)
    g = opening(img, disk(1))
    
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
    sigma2 = dmax/4
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
    
    
    plt.subplot(3, 4, 1)
    plt.imshow(img_original, cmap='gray')
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

    
    plt.tight_layout()
    plt.show()
    return np.logical_not(binary_rst1)