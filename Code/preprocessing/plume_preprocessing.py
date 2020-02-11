#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:55:56 2020

@author: alec
"""

from skimage.filters import sobel, rank, threshold_otsu
from scipy.ndimage import median_filter, gaussian_filter, binary_fill_holes
from skimage import io, img_as_float, exposure, data, img_as_uint, img_as_ubyte


def dynamic_masking(image,keep='dark',filter_size=7,threshold=None):
    """ Dynamically masks out the objects in the PIV images
    
    Parameters
    ----------
    image: image
        a two dimensional array of uint16, uint8 or similar type
    keep: string
        'dark' or 'bright'
        'dark' means mask parts of the image brighter than the threshold
        'light' means mask parts of the image darker than the threshold

    
    filter_size: integer
        a scalar that defines the size of the Gaussian filter
    
    threshold: float
        a value of the threshold to segment the background from the object
        default value: None, replaced by sckimage.filter.threshold_otsu value
            
    Returns
    -------
    image : array of the same datatype as the incoming image with the object masked out
        as a completely black region(s) of zeros (integers or floats).
    
    
    Example
    --------
    frame_a  = openpiv.tools.imread( 'Camera1-001.tif' )
    imshow(frame_a) # original
    
    frame_a = dynamic_masking(frame_a,method='edges',filter_size=7,threshold=0.005)
    imshow(frame_a) # masked 
        
    """
    imcopy = np.copy(image)
    # stretch the histogram
    # blur the image, low-pass

    background = gaussian_filter(median_filter(image,filter_size),filter_size)
    if keep =='dark':
        if threshold is None:
            imcopy[background > threshold_otsu(background)] = 0
        else:
            imcopy[background > threshold] = 0
    if keep=='light':
        if threshold is None:
            imcopy[background < threshold_otsu(background)] = 0
        else:
            imcopy[background < threshold] = 0
    return imcopy #image
