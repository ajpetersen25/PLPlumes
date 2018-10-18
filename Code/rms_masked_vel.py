# takes in masked numpy arrays and calculates a spatial average
# save final averaged array as a separate .npy file

from __future__ import division
import numpy as np
import numpy.ma as nma
from apply_mask import apply_mask as am
from avg_masked_vel import average_masked
import argparse
import os


def rms_masked(piv_file,mask_file,start_frame,end_frame):
    """ 
    Input: 
      piv   --str:   path to fluctuating .npy vel file
      mask  --str:   path to .npy mask file
    Output:
      rms_mvel  --array: masked numpy array"""
    masked_vel = am(piv_file,mask_file)
    rms_mvel = np.sqrt(nma.mean(masked_vel[start_frame:end_frame,:,:]**2,axis=0))
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
    n = np.load(args.piv_file[0]).shape[0]
    fail = False
    # check if IMG file exists
    if os.path.exists(args.mask_file[0]) == 0 or os.path.exists(args.piv_file[0])==0:
        print '[ERROR] file does not exist'
        fail = True
    
    if fail:
        print 'exiting...'
        os.sys.exit(1)
    if args.end_frame == 0:
        end_frame = n
        
    rms_mvel = rms_masked(args.piv_file[0],args.mask_file[0],args.start_frame,end_frame)
    np.save(os.path.splitext(args.piv_file[0])[0]+'.rms.npy',rms_mvel.data)
    if os.path.exists(os.path.splitext(args.mask_file[0])[0]+'.tave_mask.npy') == 0:
        np.save(os.path.splitext(args.mask_file[0])[0]+'.tave_mask.npy',rms_mvel.mask) # saves mask which will work for all time-averaged fields
    print '[FINISHED]: %f seconds elapsed' %(time.time()-tic)
if __name__ == "__main__":
    main()