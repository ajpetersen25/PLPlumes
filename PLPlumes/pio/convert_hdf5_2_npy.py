#!/usr/bin/env python3

import numpy as np
import h5py
import os
import argparse
import glob



def main():
    '''
    convert hdf5 file to npy
    '''
    
    parser = argparse.ArgumentParser(description='Program to convert hdf5 files to npy', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_files',type=str, nargs='+', help='input .hdf5')
    args = parser.parse_args()
    
    # handle input file if glob
    if len(args.input_files)==1:
        file_list = glob.glob(args.input_files[0])
        file_list.sort()
    else:
        file_list = args.input_files
    
    for i in range(0,len(file_list)):
        np_file = []
        frames_used = []
        h5 = h5py.File(file_list[i],'r')
        keys = np.array(h5['frames'])
        frames = h5['frames']
        for key in keys:
            pts = frames[key]['boundary_pts'][:]
            if len(pts) > 0:
                uf = frames[key]['uf_at_p'][:]
                up = frames[key]['u_p'][:]
                n = frames[key]['n'][:]
                frame = int(key.split('_')[1]),pts.shape[0]
                frames_used.append(frame)
                together = np.concatenate((pts,uf,up,n),axis=1)
                np_file.append(together)
    
        np_file = np.array(np_file,dtype='object')
        frames_used = np.array(frames_used)
        path = os.path.splitext(file_list[i])[0]
        np.save(path+'.npy',np_file)
        np.save(path+'.frames.npy',frames_used)
    
if __name__ == "__main__":
    main()
