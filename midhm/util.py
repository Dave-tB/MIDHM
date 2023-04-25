import cv2
import os
import numpy as np

class Util:
  @staticmethod
  def load(image='data/obj1.tif'):
    '''
    '''
    image = cv2.imread(os.path.realpath(image), 0)

    return image

  @staticmethod
  def normalize(image):

    return cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  ### FROM https://stackoverflow.com/a/68253305
  @staticmethod
  def bwareaopen(img, min_size, connectivity=8):
    """Remove small objects from binary image (approximation of 
    bwareaopen in Matlab for 2D images).

    Args:
        img: a binary image (dtype=uint8) to remove small objects from
        min_size: minimum size (in pixels) for an object to remain in the image
        connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).

    Returns:
        the binary image with small objects removed
    """

    # Find all connected components (called here "labels")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity)
    
    # check size of all connected components (area in pixels)
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]
        
        # remove connected components smaller than min_size
        if label_size < min_size:
            img[labels == i] = 0
            
    return img

  @staticmethod
  def thickness(recon, ri_bead = 1.430, ri_med = 1.510, lambda_ = 0.488):
    # Find all connected components (called here "labels")
    ref_index_diff = abs(ri_bead - ri_med)
    factor = lambda_ / (2 * np.pi * ref_index_diff)
    thickness = factor * recon
    return thickness