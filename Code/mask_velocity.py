from __future__ import division
import numpy as np
from io import pivio
from io import imgio
import os,sys
import copy
import multiprocessing
import psutil
from itertools import repeat
import warnings
import time
import argparse
import getpass

# Creates mask arrays for sparse PIV fields (like in the case of particle plumes) where many interrogation windows
# capture nothing but noise since they are not centered on any part of the plume.
# masking is based off an intensity threshold, and what fraction of pixels within a PIV interrogation window are
# above that threshold.

def mask_vframe(params):
    # PIV field masking for a single frame
    # unload parameters
    img,piv_shape,step,threshold,window_threshold,frame = params
    # read image frame from .img file
    img_frame = img.read_frame2d(frame)
    # initialize mask frame
    frame_mask = np.ones((1,piv_shape[0],piv_shape[1]))
    # loop through all interrogation windows, looking at which meet threshold criteria
    # elements in the frame mask array which meet this criteria are set to 0
    for i in range(0,piv_shape[0]):
        for j in range(0,piv_shape[1]):
            window = img_frame[i*step[0]:i*step[0]+step[0],j*step[1]:j*step[1]+step[1]]
            if np.sum(window>threshold)/(step[0]*step[1]) > window_threshold:
                frame_mask[0,i,j] = 0
    return frame_mask


def main():
    # parallelized version of frame masking, all frames are saved in the end as a .npy file
    # parse inputs
    parser = argparse.ArgumentParser(
             description='Program to calculate mask for PIV files based on raw image intensity & fraction of PIV 
             interrogation above intensity threshold', 
             formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('img_file',type=str,nargs=1,help='Path to .img file')
    parser.add_argument('piv_file',type=str,nargs=1,help='Path to .piv file')
    parser.add_argument('threshold',type=int,nargs=1,help='image intensity threshold')
    parser.add_argument('window_threshold',type=float,nargs=1,help='number between 0 and 1 representing the number of
             pixels in a PIV interrogation window that must be above the threshold to be counted as a valid PIV vector')
    parser.add_argument('start_frame',type=int,nargs=1,default=0,help='Frame of PIV you want to start the masking at')
    parser.add_argument('start_frame',type=int,nargs=1,default=0,help='Frame of PIV you want to end the masking at')
    args = parser.parse_args()
    
    fail = False
    # check if IMG file exists
    if os.path.exists(args.img_file) == 0 or os.path.exists(args.piv_file)==0:
        print '[ERROR] file does not exist'
        fail = True
        
    piv_root, piv_ext = os.path.splitext(args.piv_file)
    piv = pivio.pivio(args.piv_file)
    img = imgio.imgio(args.img_file)
    
    if fail:
        print 'exiting...'
        os.sys.exit(1)
    if args.end_frame == 0:
        end_frame = piv.nt

    # start up parallel pool
    avail_cores = int(psutil.cpu_count() - (psutil.cpu_count()/2))

    param1 = img
    param2 = piv.read_frame2d(args.start_frame)[0].shape
    param3 = (piv.dx, piv.dy)
    param4 = args.threshold
    param5 = args.window_threshold
    param6 = range(args.start_frame,args.end_frame)
    f_tot = len(param6)
    objList = zip(repeat(param1,times=f_tot),
                  repeat(param2,times=f_tot),
                  repeat(param3,times=f_tot),
                  repeat(param4,times=f_tot),
                  repeat(param5,times=f_tot),
                  param6)
    pool = multiprocessing.Pool(processes=avail_cores)

    # process
    tic = time.time()
    all_frames = pool.map(mask_vframe,objList)
    print time.time() - tic, 's'
    results = np.concatenate(all_frames[:],axis=0)
    np.save(os.path.dirname(piv.file_name)+'/plume_velmask.npy',results)
    print
   
if __name__ == "__main__":
    main()
