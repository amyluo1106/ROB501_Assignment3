import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from stereo_disparity_score import stereo_disparity_score
from stereo_disparity_fast import stereo_disparity_fast

# Load the stereo images and ground truth.
Il = imread("../assignment_3/images/teddy/teddy_image_02.png", as_gray = True)
Ir = imread("../assignment_3/images/teddy/teddy_image_06.png", as_gray = True)

# The cones and teddy datasets have ground truth that is 'stretched'
# to fill the range of available intensities - here we divide to undo.
It = imread("../assignment_3/images/teddy/teddy_disp_02.png",  as_gray = True)/4.0

# Load the appropriate bounding box.
bbox = np.load("../assignment_3/data/teddy_02_bounds.npy")

Id = stereo_disparity_fast(Il, Ir, bbox, 55)
N, rms, pbad = stereo_disparity_score(It, Id, bbox)
print("Valid pixels: %d, RMS Error: %.2f, Percentage Bad: %.2f" % (N, rms, pbad))
plt.imshow(Id, cmap = "gray")
plt.show()