import numpy as np
from PLPlumes.pio import imgio, pivio

from openpiv.tools import save
import multiprocessing
import os
from itertools import repeat
import time
import argparse
import copy

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
    img,piv,threshold,window_threshold,frame = params
    # read image frame from .img file
    step = (piv.dy,piv.dx)
    img_frame = img.read_frame2d(frame)
    # initialize mask frame
    mask = piv.read_frame2d(frame)[0]
    # loop through all interrogation windows, looking at which meet threshold criteria
    # elements in the frame mask array which meet this criteria are set to 0
    for i in range(0,piv.ny):
        for j in range(0,piv.nx):
            window = img_frame[i*step[0]:i*step[0]+step[0],j*step[1]:j*step[1]+step[1]]
            if np.sum(window>threshold)/(step[0]*step[1]) > window_threshold and mask[i,j]==1:
                mask[i,j] = 1
            else:
                mask[i,j] = 0

    return(mask)


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
    parser.add_argument('cores',type=int,nargs=1,default=1,help='number of cores to use')

    args = parser.parse_args()
    tic = time.time()
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

    param1 = img
    param2 = piv
    param3 = args.threshold
    param4 = args.window_threshold
    param5 = np.arange(args.start_frame,end_frame)

    f_tot = len(param5)
    objList = list(zip(repeat(param1,times=f_tot),
                  repeat(param2,times=f_tot),
                  repeat(param3,times=f_tot),
                  repeat(param4,times=f_tot),
                  param5))
    pool = multiprocessing.Pool(processes=args.cores)

    # process
    
    #all_frame_masks,all_U,all_W = pool.map(mask_vframe,objList)
    masks = pool.map(mask_vframe,objList)
    masks = np.array(masks)
    piv_m = copy.deepcopy(piv)
    piv_m.file_name = os.path.splitext(piv.file_name)[0]+'.msk.piv'
    d = datetime.now()
    piv_m.comment = piv.comment = "%s\n%s %d %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(),os.path.basename(piv.file_name), 1, piv, 
                                                     d.strftime("%a %b %d %H:%M:%S %Y")) + piv.comment
    piv_m.write_header()
    
    for f in range(0,f_tot):
        data = [masks[f].flatten(),piv.read_frame(f)[1],piv.read_frame(f)[2]]
        piv_m.write_frame(data)
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
   
if __name__ == "__main__":
    main()
