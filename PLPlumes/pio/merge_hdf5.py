#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:53:32 2020

@author: ajp25
"""


import numpy as np
import h5py
import argparse
from datetime import datetime
import tables

def main():
    '''Join hdf5 files together'''
    parser = argparse.ArgumentParser(description='Program to join HDF5 files', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('output_file', type=str, help='Name of output HDF5 file')
    parser.add_argument('input_files', nargs='+', help='Name of HDF5 files to join')
    args = parser.parse_args()
    #files = sort(args.input_files)
    #print(files)
    h5 = h5py.File(args.output_file,'a')
    try:
        frame_group = h5.create_group('frames')
    except:
        pass
    for i in range(0,len(args.input_files)):
        h_temp = h5py.File(args.input_files[i],'r')
        for key in h_temp.keys():
            for k in h_temp[key].keys():
                #h5[key].create_dataset(k+'/frame_mask',data=h_temp[key][k]['frame_masks'][:])
                h5[key].create_dataset(k+'/boundary_pts',data=h_temp[key][k]['boundary_pts'][:])
                h5[key].create_dataset(k+'/uf_at_p',data=h_temp[key][k]['uf_at_p'][:])
                h5[key].create_dataset(k+'/u_p',data=h_temp[key][k]['u_p'][:])
                h5[key].create_dataset(k+'/n',data=h_temp[key][k]['n'][:])
        h_temp.close()
        del h_temp
    h5.close()
    tables.file._open_files.close_all()
    #results = []
    #for file in files:
    #    results.append(np.load(file))
        
    #results = np.array(results)
    #np.save(args.output_file,results)
    
if __name__ == "__main__":
  main() 
