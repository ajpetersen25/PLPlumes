import numpy as np
from .pio import pivio
from .pio import imgio
import os,sys
import copy
import multiprocessing
import psutil
from itertools import repeat
import time
import argparse

# Creates mask arrays for sparse PIV fields (like in the case of particle plumes) where many interrogation windows
# capture nothing but noise since they are not centered on any part of the plume.
# masking is based off an intensity threshold, and what fraction of pixels within a PIV interrogation window are
# above that threshold.

def mask_vframe(params):
    # PIV field masking for a single frame
    # unload parameters
    img,piv,threshold,window_threshold,frame = params
    # read image frame from .img file
    step = (piv.dy,piv.dx)
    img_frame = img.read_frame2d(frame)
    # initialize mask frame
    frame_mask = np.ones((piv.ny,piv.nx))
    # loop through all interrogation windows, looking at which meet threshold criteria
    # elements in the frame mask array which meet this criteria are set to 0
    for i in range(0,piv.ny):
        for j in range(0,piv.nx):
            window = img_frame[i*step[0]:i*step[0]+step[0],j*step[1]:j*step[1]+step[1]]
            if np.sum(window>threshold)/(step[0]*step[1]) > window_threshold:
                frame_mask[i,j] = 0
    # if piv status is bad (rejected, low SNR, etc) mask out automatically
    status = piv.read_frame2d(frame)[0]#.reshape(1,piv.read_frame2d(frame)[0].shape[0],piv.read_frame2d(frame)[0].shape[1])
    frame_mask[status !=1] = 1
    return frame_mask, piv.read_frame2d(frame)[1],piv.read_frame2d(frame)[2]#.reshape(1,piv.read_frame2d(0)[0].shape[0],piv.read_frame2d(0)[0].shape[1])


def main():
    # parallelized version of frame masking, all frames are saved in the end as a .npy file
    # parse inputs
    parser = argparse.ArgumentParser(
             description='Program to calculate mask for PIV files based on raw image intensity & fraction of PIV interrogation above intensity threshold', 
             formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('img_file',type=str,nargs=1,help='Path to .img file')
    parser.add_argument('piv_file',type=str,nargs=1,help='Path to .piv file')
    parser.add_argument('threshold',type=int,nargs=1,help='image intensity threshold')
    parser.add_argument('window_threshold',type=float,nargs=1,help='number between 0 and 1 representing the number of pixels in a PIV interrogation window that must be above the threshold to be counted as a valid PIV vector')
    parser.add_argument('start_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to start the masking at')
    parser.add_argument('end_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to end the masking at')
    args = parser.parse_args()
    
    fail = False
    # check if IMG file exists
    if os.path.exists(args.img_file[0]) == 0 or os.path.exists(args.piv_file[0])==0:
        print('[ERROR] file does not exist')
        fail = True
        
    piv_root, piv_ext = os.path.splitext(args.piv_file[0])
    piv = pivio.pivio(args.piv_file[0])
    img = imgio.imgio(args.img_file[0])
    
    if fail:
        print('exiting...')
        os.sys.exit(1)
    if args.end_frame == 0:
        end_frame = piv.nt
    # start up parallel pool
    avail_cores = int(psutil.cpu_count() - (psutil.cpu_count()/2))

    param1 = img
    param2 = piv
    param3 = args.threshold
    param4 = args.window_threshold
    param5 = list(range(args.start_frame,end_frame))
    f_tot = len(param5)
    objList = list(zip(repeat(param1,times=f_tot),
                  repeat(param2,times=f_tot),
                  repeat(param3,times=f_tot),
                  repeat(param4,times=f_tot),
                  param5))
    pool = multiprocessing.Pool(processes=avail_cores)

    # process
    tic = time.time()
    #all_frame_masks,all_U,all_W = pool.map(mask_vframe,objList)
    all= pool.map(mask_vframe,objList)
    all = np.array(all)
    #print all.shape
    mask_results = all[:,0,:,:]
    U_results = all[:,1,:,:]
    W_results = all[:,2,:,:]
    np.savez_compressed(os.path.splitext(piv.file_name)[0]+'.mask.npz',mask_results)
    np.savez_compressed(os.path.splitext(piv.file_name)[0]+'.U.npz',U_results)
    np.savez_compressed(os.path.splitext(piv.file_name)[0]+'.W.npz',W_results)
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
   
if __name__ == "__main__":
    main()
