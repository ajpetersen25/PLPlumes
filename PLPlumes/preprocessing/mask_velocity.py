import numpy as np
#from .pio import pivio
from PLPlumes.pio.piv_io import load_piv
from PLPlumes.pio.image_io import load_tif
from openpiv.tools import save
import multiprocessing
import os
from itertools import repeat
import time
import argparse

# Creates mask arrays for sparse PIV fields (like in the case of particle plumes) where many interrogation windows
# capture nothing but noise since they are not centered on any part of the plume.
# masking is based off an intensity threshold, and what fraction of pixels within a PIV interrogation window are
# above that threshold.

def mask_vframe(params):
    """
    Parameters
    ----------
    params : img,piv,threshold,window_threshold,vector_spacing,piv_shape,overwrite
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # PIV field masking for a single frame
    # unload parameters
    img,piv,threshold,window_threshold,vector_spacing,piv_shape,overwrite = params
    # read image frame from .img file
    img_frame = load_tif(img)
    step = (vector_spacing[0],vector_spacing[1])
    x,y,u,v,mask = load_piv(piv,piv_shape,full=True)
    # loop through all interrogation windows, looking at which meet threshold criteria
    # elements in the frame mask array which meet this criteria are set to 0
    for i in range(0,mask.shape[0]):
        for j in range(0,mask.shape[1]):
            window = img_frame[i*step[0]:i*step[0]+step[0],j*step[1]:j*step[1]+step[1]]
            if np.sum(window>threshold)/(step[0]*step[1]) > window_threshold and mask[i,j]==1:
                mask[i,j] = 1
            else:
                mask[i,j] = 0
    if overwrite==True:
          save(x, y, u, v,mask,piv, delimiter='\t')
    else:
        return(x,y,u,v,mask)


def main():
    # parallelized version of frame masking, all frames are saved in the end as a .npy file
    # parse inputs
    parser = argparse.ArgumentParser(
             description='Program to calculate mask for PIV files based on raw image intensity & fraction of PIV interrogation above intensity threshold', 
             formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('threshold',default=100,type=int,nargs=1,help='image intensity threshold')
    parser.add_argument('window_threshold',default=0.05,type=float,nargs=1,help='number between 0 and 1 representing the number of pixels in a PIV interrogation window that must be above the threshold to be counted as a valid PIV vector')
    parser.add_argument('cores',type=int,nargs=1,default=1,help='number of cores to use')
    parser.add_argument('-i','--img_files',nargs='+',help='Path to .tif files')
    parser.add_argument('-p','--piv_files',nargs='+',help='Path to _piv.txt files')
    args = parser.parse_args()
    tic = time.time()
    fail = False
    # check if IMG file exists
    if os.path.exists(args.img_files[0]) == 0 or os.path.exists(args.piv_files[0])==0:
        print('[ERROR] file does not exist')
        fail = True
        

    if fail:
        print('exiting...')
        os.sys.exit(1)

    # start up parallel pool

    param1 = args.img_files
    param2 = args.piv_files
    param3 = args.threshold
    param4 = args.window_threshold
    param5 = (32,32)
    param6 = (49,79)
    param7 = True
    f_tot = len(args.piv_files)
    objList = list(zip(param1,param2,
                  repeat(param3,times=f_tot),
                  repeat(param4,times=f_tot),
                  repeat(param5,times=f_tot),
                  repeat(param6,times=f_tot),
                  repeat(param7,times=f_tot)))
    pool = multiprocessing.Pool(processes=args.cores[0])

    # process
    
    #all_frame_masks,all_U,all_W = pool.map(mask_vframe,objList)
    pool.map(mask_vframe,objList)

    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
   
if __name__ == "__main__":
    main()
