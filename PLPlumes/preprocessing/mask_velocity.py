#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PLPlumes.pio import imgio, pivio

import multiprocessing
import os
from itertools import repeat
import time
import argparse
import copy
from datetime import datetime 
import getpass

# Creates mask arrays for sparse PIV fields (like in the case of particle plumes) where many interrogation windows
# capture nothing but noise since they are not centered on any part of the plume.
# masking is based off an intensity threshold, and what fraction of pixels within a PIV interrogation window are
# above that threshold.

def mask_vframe(params):
    """
    Parameters
    ----------
    params : img,piv,threshold,window_threshold,frame_number
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # PIV field masking for a single frame
    # unload parameters
    img,piv,threshold,window_threshold,slices,frame = params
    # read image frame from .img file
    step = (piv.dy/2,piv.dx/2)
    img_frame = img.read_frame2d(frame)
    # initialize mask frame
    mask = piv.read_frame2d(frame)[0]
    mask[mask==4]=1
    mask[mask!=1]=0
    # loop through all interrogation windows, looking at which meet threshold criteria
    # elements in the frame mask array which meet this criteria are set to 0

    #for s in slices:
    #        window = img_frame[s]
    #        if np.sum(window>threshold)/(step[0]*step[1]) > window_threshold and mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] ==1:
    #            mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] = 1
    #        else:
    #            mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)]  = 0
    #for r in range(mask.shape[0]):
    #    for c in range(mask.shape[1]):
    #        if piv.read_frame2d(0)[1][r,c] < 1:
    #            mask[r,c] = 0

    return(np.flipud(mask))


def main():
    # parallelized version of frame masking, all frames are saved in the end as a .npy file
    # parse inputs
    parser = argparse.ArgumentParser(
             description='Program to calculate mask for PIV files based on raw image intensity & fraction of PIV interrogation above intensity threshold', 
             formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('img_file',type=str,help='Path to .img file')
    parser.add_argument('piv_file',type=str,help='Path to .piv file')
    parser.add_argument('threshold',default=100,type=int,nargs=1,help='image intensity threshold')
    parser.add_argument('window_threshold',default=0.05,type=float,nargs=1,help='number between 0 and 1 representing the number of pixels in a PIV interrogation window that must be above the threshold to be counted as a valid PIV vector')
    parser.add_argument('start_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to start the masking at')
    parser.add_argument('end_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to end the masking at')
    parser.add_argument('cores',type=int,nargs='?',default=1,help='number of cores to use')

    args = parser.parse_args()
    tic = time.time()
    fail = False
    # check if IMG file exists
    if os.path.exists(args.img_file) == 0 or os.path.exists(args.piv_file)==0:
        print('[ERROR] file does not exist')
        fail = True
        
    piv = pivio.pivio(args.piv_file)
    img = imgio.imgio(args.img_file)
    
    if fail:
        print('exiting...')
        os.sys.exit(1)
        
    if args.end_frame == 0:
        end_frame = piv.nt
    else:
        end_frame = args.end_frame
    # start up parallel pool

    x = np.arange(piv.dx,img.ix-piv.dx,piv.dx)
    y = np.arange(piv.dy,img.iy-piv.dy,piv.dy)
    slices = []
    for i in x:
       for j in y:
           slices.append((slice(int(j),int(j+piv.dy)),slice(int(i),int(i+piv.dx))))
           
    param1 = img
    param2 = piv
    param3 = args.threshold
    param4 = args.window_threshold
    param5 = slices
    param6 = np.arange(args.start_frame,end_frame)

    f_tot = len(param6)
    objList = list(zip(repeat(param1,times=f_tot),
                  repeat(param2,times=f_tot),
                  repeat(param3,times=f_tot),
                  repeat(param4,times=f_tot),
                  repeat(param5,times=f_tot),
                  param6))
    pool = multiprocessing.Pool(processes=args.cores)

    # process
    
    #all_frame_masks,all_U,all_W = pool.map(mask_vframe,objList)
    masks = pool.map(mask_vframe,objList)
    masks = np.array(masks)
    
    piv_root,piv_ext = os.path.splitext(args.piv_file)
    piv_m = copy.deepcopy(piv)
    piv_m.file_name = piv_root+'.msk.piv'
    d = datetime.now()
    piv_m.comment = piv.comment = "%s\n%s %d %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(),os.path.basename(piv.file_name), 1, piv, 
                                                     d.strftime("%a %b %d %H:%M:%S %Y")) + str(piv.comment,'utf-8')
    piv_m.nt = f_tot
    piv_m.write_header()
    
    for f in range(0,f_tot):
        data = [masks[f].flatten(),piv.read_frame(f)[1],piv.read_frame(f)[2]]
        piv_m.write_frame(data)
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
   
if __name__ == "__main__":
    main()
