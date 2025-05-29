import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import black_tophat
from skimage.morphology import footprint_rectangle, disk, ellipse
from skimage import segmentation, filters, exposure
from skimage.morphology import remove_small_objects, local_maxima

def watershed_segmentation(img, img_mask):
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
    black_hat_img2 = black_tophat(img, footprint_rectangle((1,10)))
    
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

