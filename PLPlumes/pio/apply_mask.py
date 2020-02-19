# code to load in velocity & mask and apply as a NumPy masked array
# (necessary until numpy.ma.MaskedArray.tofile is implemented)



import numpy.ma as nma
from PLPlumes.pio.piv_io import load_piv



def apply_masktxt(piv_txt_file,piv_arr_shape):
    x,y,u,v,mask_arr = load_piv(piv_txt_file,piv_arr_shape,full=True)
    masked_u = nma.masked_array(u,mask=mask_arr)
    masked_v = nma.masked_array(v,mask=mask_arr)
    return x,y,masked_u,masked_v,mask_arr