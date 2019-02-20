# code to load in velocity & mask and apply as a NumPy masked array
# (necessary until numpy.ma.MaskedArray.tofile is implemented)

from __future__ import division
import numpy as np
import numpy.ma as nma
from pio import load_npz


def apply_mask(piv_str,mask_str):
    piv_arr = np.load(piv_str)
    mask_arr = np.load(mask_str)
    masked_vel = nma.masked_array(piv_arr,mask=mask_arr)
    return masked_vel

def apply_maskz(piv_npz, mask_npz):
    piv_arr = load_npz.load_npz(piv_str)
    mask_arr = load_npz.load_npz(mask_str)
    masked_vel = nma.masked_array(piv_arr,mask=mask_arr)
    return masked_vel