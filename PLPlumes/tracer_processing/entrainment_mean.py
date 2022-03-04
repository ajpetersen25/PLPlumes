#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:36:33 2020

@author: ajp25
"""


import numpy as np
import h5py
import os
import time
import argparse

#def mean_entrainment(params):
    
    
def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program average entrainment vel hdf5 files',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('hdf5_file',type=str, help='Name of .hdf5 file')
    parser.add_argument('image_length',type=int, help='length of image file')
    args = parser.parse_args()
    file = h5py.File(args.hdf5_file,'r')
    root,ext = os.path.splitext(args.hdf5_file)
    frames = file['frames'].keys()
    
    param1 = file
    
    ue_sum = np.zeros((args.image_length,))
    ue_sum = np.full_like(ue_sum,np.nan)
    ue_num = np.full_like(ue_sum,np.nan)
    for f in frames:
        pts = file['frames'][f]['boundary_pts'][:]
        u_e = file['frames'][f]['u_e'][:]
        for c in range(0,len(ue_sum)):
            cols = np.where(pts[:,0]==c)[0]
            ue_sum[c] = np.nansum([ue_sum[c]]+[np.nansum(u_e[cols])])
            ue_num[c] = np.nansum([ue_num[c]] + [np.sum(~np.isnan(u_e[cols]))])
    
    
    ue_num[np.where(ue_num==0)[0]] = np.nan
    ue_mean = ue_sum/ue_num
    np.savetxt(root+'.mean.txt',ue_mean,delimiter='\t')
    

    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()    
