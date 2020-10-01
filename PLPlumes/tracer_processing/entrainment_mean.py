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
import multiprocessing
from itertools import repeat

  
def sum_ue(params):
    hdf5_file,frame = params
    h5 = h5py.File(hdf5_file,'r')
    arr = h5['frames'][frame][:]
    #h5.close()
    ue_sum = np.zeros((int(np.max(arr[:,1])+1),))
    ue_sum = np.full_like(ue_sum,np.nan)
    ue_num = np.full_like(ue_sum,np.nan)
    for f in frames:
        arr = h5['frames'][frame][:]
        for c in range(0,len(ue_sum)):
            cols = np.where(arr[:,1]==c)[0]
            u = arr[:,2][cols]
            ue_sum[c] = np.nansum([ue_sum[c]]+[np.nansum(u)])
            ue_num[c] = np.nansum([ue_num[c]] + [np.sum(~np.isnan(u))])
    return(ue_sum,ue_num)
    
def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program average entrainment vel h5py files',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('hdf5_file',type=str, help='Name of .hdf5 file')
    parser.add_argument('image_length',type=int, help='length of image file')
    args = parser.parse_args()
    h5 = h5py.File(args.hdf5_file,'r')
    root,ext = os.path.splitext(args.hdf5_file)
    frames = list(h5['frames'].keys())
    
    ue_sum = np.zeros((args.image_length,))
    ue_sum = np.full_like(ue_sum,np.nan)
    ue_num = np.full_like(ue_sum,np.nan)
    if args.image_length==2560:
        row=1
    elif args.image_length==1600:
        row=0
    for frame in frames:
        arr = h5['frames'][frame][:]
        for c in range(0,len(ue_sum)):
            cols = np.where(arr[:,row]==c)[0]
            u = arr[:,2][cols]
            ue_sum[c] = np.nansum([ue_sum[c]]+[np.nansum(u)])
            ue_num[c] = np.nansum([ue_num[c]] + [np.sum(~np.isnan(u))])
    ue_num[np.where(ue_num==0)[0]] = np.nan
    ue_mean = ue_sum/ue_num
    h5.close()
    np.savetxt(root+'.mean.txt',ue_mean,delimiter='\t')
    

    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()    
