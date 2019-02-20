# code to load in velocity & mask and apply as a NumPy masked array
# (necessary until numpy.ma.MaskedArray.tofile is implemented)

from __future__ import division
import numpy as np
import numpy.ma as nma


def apply_mask(piv_str,mask_str):
    piv_arr = np.load(piv_str)
    mask_arr = np.load(mask_str)
    masked_vel = nma.masked_array(piv_arr,mask=mask_arr)
    return masked_vel
