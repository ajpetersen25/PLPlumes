# code to load in velocity & mask and apply as a NumPy masked array
# (necessary until numpy.ma.MaskedArray.tofile is implemented)



import numpy.ma as nma


def apply_mask(piv_arr,mask_arr):
    masked_vel = nma.masked_array(piv_arr,mask=~mask_arr.astype('bool'))
    return masked_vel