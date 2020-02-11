# takes in masked numpy arrays and calculates a spatial average
# save final averaged array as a separate .npy file


import numpy as np
import numpy.ma as nma
from .apply_mask import apply_mask as am
from .avg_masked_vel import average_masked
import argparse
import os
import time


def rms_masked(piv_file,mask_file,start_frame,end_frame):
    """ 
    Input: 
      piv   --str:   path to fluctuating .npy vel file
      mask  --str:   path to .npy mask file
    Output:
      rms_mvel  --array: masked numpy array"""
    masked_vel = am(piv_file,mask_file)
    avg_vel = nma.mean(masked_vel[start_frame:end_frame,:,:],axis=0)
    rms_mvel = np.sqrt(nma.mean((masked_vel[start_frame:end_frame,:,:]-avg_vel)**2,axis=0))
    return rms_mvel

def main():
    # calls rms_masked and saves rms velocity field in a
    # .npy files (necessary until saving masked numpy arrays is implemented)
    # mask file is the same as that produced by avg_mask_vel
    tic = time.time()
    parser = argparse.ArgumentParser(
             description='Program to calculate average field of masked npy PIV files (saves average mask as well)', 
             formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('piv_file',type=str,nargs=1,help='Path to fluctuating velocity .npy file')
    parser.add_argument('mask_file',type=str,nargs=1,help='Path to masking .npy file')
    parser.add_argument('start_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to start the masking at')
    parser.add_argument('end_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to end the masking at')
    args = parser.parse_args() 
    fail = False
    # check if IMG file exists
    if os.path.exists(args.mask_file[0]) == 0 or os.path.exists(args.piv_file[0])==0:
        print('[ERROR] file does not exist')
        fail = True
    
    if fail:
        print('exiting...')
        os.sys.exit(1)
    if args.end_frame == 0:
        temp = np.load(args.piv_file[0])
        end_frame = temp['arr_0'].shape[0]
        
    rms_mvel = rms_masked(args.piv_file[0],args.mask_file[0],args.start_frame,end_frame)
    np.savez_compressed(os.path.splitext(args.piv_file[0])[0]+'.rms.npz',rms_mvel.data)
    if os.path.exists(os.path.splitext(args.mask_file[0])[0]+'.tave_mask.npz') == 0:
        np.savez_compressed(os.path.splitext(args.mask_file[0])[0]+'.tave_mask.npz',rms_mvel.mask) # saves mask which will work for all time-averaged fields
    print('[FINISHED]: %f seconds elapsed' %(time.time()-tic))
if __name__ == "__main__":
    main()
