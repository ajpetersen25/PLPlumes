# code to load in velocity & mask and apply as a NumPy masked array
# (necessary until numpy.ma.MaskedArray.tofile is implemented)


import numpy as np
import numpy.ma as nma
from Code.pio import load_npz


def apply_mask(piv_str,mask_str):
    piv_arr = np.load(piv_str)['arr_0']
    mask_arr = np.load(mask_str)['arr_0']
    masked_vel = nma.masked_array(piv_arr,mask=mask_arr)
    return masked_vel

def apply_maskz(piv_npz, mask_npz):
    piv_arr = load_npz.load_npz(piv_npz)
    mask_arr = load_npz.load_npz(mask_npz)
    masked_vel = nma.masked_array(piv_arr,mask=mask_arr)
    return masked_vel