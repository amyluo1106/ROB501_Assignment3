import numpy as np
from scipy.ndimage.filters import gaussian_laplace, median_filter

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---

    """
    This alternative matching algorithm first utilizes a Laplacian of Gaussians filter (sigma = 0.8) on both raw stereo images.
    This ensures that edges become more well defined, allowing for more contrast which aids in the SAD cost computation afterwards.
    This step is significant for the KITTI dataset as there is a lot of features in the image due to the busy street image.
    After the disparities are matched using SAD, a median filter (10x10) is passed over the disparity map in order to do smoothing and remove noise.
    Parameters were tuned using trial and error.
    """

    w, l = Il.shape

    Id = np.zeros((w, l))

    # Window size
    window_size = 14
    half_window = window_size // 2

    Il = gaussian_laplace(Il, 1)
    Ir = gaussian_laplace(Ir, 1)
    
    # Loop through image in bounding box region
    for y in range(bbox[1][0], bbox[1][1]):
        for x in range(bbox[0][0], bbox[0][1]):
            match = 0
            min_sad = float('inf')
            
            # Search under max disparity value
            for d in range(maxd+1):

                if x-d >= half_window and x+half_window < l:
                     left_window = Il[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
                     right_window = Ir[y - half_window:y + half_window + 1, x - d - half_window:x - d + half_window + 1]
                     
                     # Compute the SAD score for the current disparity
                     sad = np.sum(np.abs(left_window - right_window))

                     if sad < min_sad:
                         match = d
                         min_sad = sad

            Id[y, x] = match    

    # Apply a median filter to smooth the disparity map
    Id = median_filter(Id, 10)  

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id