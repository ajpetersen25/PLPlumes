#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:50:18 2020

@author: alec
"""

import os,sys
from os.path import splitext
import argparse
import numpy as np

import time
from PLPlumes.pio.image_io import load_tif,write_tif

import multiprocessing
from itertools import repeat

def backsub_t0(params):
    frame,d,save_name = params
    write_tif(save_name,frame-d)

def backsub_t1(params):
    frame1,frame2,d1,d2,save_name1,save_name2 = params
    write_tif(save_name1,frame1-d1)
    write_tif(save_name2,frame2-d2)

    

def main():
    ''' Background subtract images. '''
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program to perform background subtract and normalise img files.\n\nThe input file is first normalised, then minimum values are calculated and subtracted.\n\nOutputs are min.piv and .bsub.piv files',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('save_path',type=str,nargs=1,default='./',help='path to directory you want to save _bsub images')
    parser.add_argument('type', type=int,
               help='[0] all [1] pulsed')
    parser.add_argument('frame_increment', type=int, nargs='?',
                      default=1,help='Frames to increment for backsub for subsampling.\nOnly valid toy type 0')
    parser.add_argument('cores',type=int,nargs=1,default=1,help='number of cores to use')
    parser.add_argument('files', nargs='+',
               help='Name of files as inputs')
    args = parser.parse_args()
    
    for file in args.files:
        # skip files that don't exist
        if os.path.isfile(file) == 0:
            print('WARNING: File does not exist: %s' % file)
            print('skipping...')
            continue
        else:
            print('Handling File : %s' % file)
    
    img_files = args.files#glob.glob(args.path_to_image_files)   
    save_list = [args.save_path+splitext(e)[0] +'_bsub' +splitext(e)[1] for e in img_files]
    
    
    img = load_tif(img_files[0])
    if args.type == 0:
        print('Continuous background subtraction')
        print('Finding  minimum')
    
        d = img.max*np.ones(img.shape[0]*img.shape[1],)
        # generate frame numbers to average (excluding first frame above)
        frames = np.arange(0,len(img_files),args.frame_increment);
        for i in frames:
            d = np.minimum(d,load_tif(img_files[i]))
            
        sys.stdout.write('\nPerforming background subtraction\n')
        f_tot = len(img_files)
        objList = list(zip(img_files,
           repeat(d,times=f_tot),
           save_list))
        
        pool = multiprocessing.Pool(processes=args.cores)
        
        pool.map(backsub_t0,objList)
    
    elif args.type == 1:
        print('Paired background subtraction')
        print('Finding  minimum')
        
        d1 = img.max*np.ones(img.shape[0]*img.shape[1],)
        d2 = img.max*np.ones(img.shape[0]*img.shape[1],)
        for i in range(0,len(img_files),2):
            d1 = np.minimum(d1,img.read_frame(i))
            d2 = np.minimum(d2,img.read_frame(i+1))

        sys.stdout.write('\nPerforming background subtraction\n')
        img_files1 = img_files[::2]
        img_files2 = img_files[1::2]
        slist1 = save_list[::2]
        slist2 = save_list[1::2]
        f_tot = len(img_files1)
        objList = list(zip(img_files1,img_files2,
           repeat(d1,times=f_tot),
           repeat(d2,times=f_tot),
           slist1,slist2))
    
        pool = multiprocessing.Pool(processes=args.cores)
        
        pool.map(backsub_t1,objList)
        
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()
    
    
