# convergence of velocity profiles
# for specified height(s) z, calculated and save an nxt array where n is the number of piv vectors in the
# horizontal (x) direction, and t is the number of piv frames available. Each element is then the average
# velocity for that x & z position up to that moment t in time.

from __future__ import division
import numpy as np
from pio import pivio
import argparse
import numpy.ma as nma
import os
import apply_mask


def profile_convergence(masked_vel, height):
    # takes in a numpy masked array of velocities and a height, and returns an array of the same rows,
    # but with n columns (n=number of frames) each representing the average velocity profile up to that
    # point in time
    
    moving_avg = masked_vel[0][height,:].reshape(masked_vel[0][height,:].shape[0],1)
    rmse_series = np.zeros((masked_vel.shape[0]-1,))
    for f in range(1,masked_vel.shape[0]):
        prof = masked_vel[f][height,:].reshape(masked_vel[f][height,:].shape[0],1)
        moving_avg_new = nma.sum(nma.concatenate((moving_avg*f,prof),axis=1),axis=1).reshape(prof.shape[0],1)/(f+1)
        #print moving_avg_new.shape, prof.shape
        #print prof.shape, convergence_series[:,f-1].shape
        #prof_update = nma.mean(nma.concatenate((convergence_series[*f,prof),axis=1),axis=1).reshape(prof.shape[0],1)
        #print prof_update.shape, convergence_series.shape
        #convergence_series = nma.concatenate((convergence_series,prof_update),axis=1)
        rmse =  np.sqrt(np.mean((moving_avg_new-moving_avg)**2))
        moving_avg = moving_avg_new
        rmse_series[f-1] = rmse
    return rmse_series


def main():
    
    # parse inputs
    parser = argparse.ArgumentParser(
             description='Program to calculate convergence of velocity profiles for plume PIV', 
             formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('piv', type=str, nargs=1,help=' path to PIV array (.npy)')
    parser.add_argument('vel_mask',type=str,nargs=1,help='path to mask file (.npy)')
    args = parser.parse_args()
    masked_vel = apply_mask.apply_mask(args.piv[0],args.vel_mask[0])
    print '\n PIV frame height is %d windows\n\n' %(masked_vel.shape[1])
    
    heights = raw_input('Enter desired profile heights (separated by a space): ')
    heights = [int(i) for i in heights.split(' ')]
    
    
    for h in heights:
        prof_rmse = profile_convergence(masked_vel,h)
        np.save(os.path.splitext(args.piv[0])[0]+'_converge_rmse_h%d.npy' %h,prof_rmse)
    
    print 'calculated & saved convergence RMSE series'
    
    
    
if __name__ == "__main__":
    main()
    