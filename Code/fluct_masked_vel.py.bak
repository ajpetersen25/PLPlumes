# %load fluct_masked_vel
# takes in masked numpy arrays and the fluctuating component
# save final fluctuating array as a separate .npy file

from __future__ import division
import numpy as np
import numpy.ma as nma
from avg_masked_vel import average_masked
import multiprocessing
import psutil
from itertools import repeat
import argparse
import os, sys
from apply_mask import apply_mask as am
import time


def fluct_masked(params):
    piv,ave_mvel,frame = params
    if isinstance(piv, basestring):
        piv = np.load(piv)['arr_0']
    fluct_frame = piv[frame,:,:] - ave_mvel.data
    return fluct_frame
    
    
    
def main():
  # calls average_masked and saves both average velocity field and mask for time-averaged field in separate
  # .npy files (necessary until saving masked numpy arrays is implemented)
    parser = argparse.ArgumentParser(
             description='Program to calculate fluctuating field of masked npy PIV files', 
             formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('piv_file',type=str,nargs=1,help='Path to velocity .npy file')
    parser.add_argument('mask_file',type=str,nargs=1,help='Path to masking .npy file')
    parser.add_argument('start_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to start the masking at')
    parser.add_argument('end_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to end the masking at')
    parser.add_argument('ncores',type=int,nargs='?',default=0,help='number of cores used, default = 3/4 cores on computer ')
    args = parser.parse_args() 
    
    fail = False
    # check if IMG file exists
    if os.path.exists(args.mask_file[0]) == 0 or os.path.exists(args.piv_file[0])==0:
        print '[ERROR] file does not exist'
        fail = True
    piv = np.load(args.piv_file[0])
    if fail:
        print 'exiting...'
        os.sys.exit(1)
    if args.end_frame == 0:
        temp = np.load(args.piv_file[0])
        end_frame = temp['arr_0'].shape[0]
    tic = time.time()
    ave_mvel = average_masked(args.piv_file[0],args.mask_file[0],args.start_frame,args.end_frame)
    # start up parallel pool
    if args.ncores==0:
        avail_cores = int(psutil.cpu_count() - (psutil.cpu_count()/2))
    else:
        avail_cores = args.ncores

    param1 = args.piv_file[0]
    param2 = ave_mvel
    param3 = range(args.start_frame,end_frame)
    f_tot = len(param3)
    objList = zip(repeat(param1,times=f_tot),
                  repeat(param2,times=f_tot),
                  param3)
    pool = multiprocessing.Pool(processes=avail_cores)
    
    # process

    fluct_mvel = pool.map(fluct_masked,objList)
    results = np.array(fluct_mvel)
    np.savez_compressed(os.path.splitext(args.piv_file[0])[0]+'.flct.npz',results)
    print '[FINISHED]: %f seconds elapsed' %(time.time()-tic)
    
if __name__ == "__main__":
    main()
