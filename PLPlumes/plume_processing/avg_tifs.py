#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:52:32 2020

@author: alec
"""
import os
import argparse
import numpy as np
import time
from PLPlumes.pio.image_io import load_tif,write_tif

def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program to perform spatial light intensity normalization of img files.\n\nOutputs an image of mean brightness across frames. \n\nIndividual images are then normalized by the mean bright frame: F_adjusted = (F_original * mean(F_mean)/F_mean)*max_intensity/max(F_original)\n',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('bitdepth',type=int,
               help='bit depth (2^n), input n')
    parser.add_argument('frame_increment', type=int, nargs='?',
                        default=1,help='Frames to increment for backsub for subsampling.')
    parser.add_argument('files', nargs='+',
               help='tif files as inputs')
    parser.add_argument('-s','--save_name',type=str,nargs='?',default=None,help='path to directory you want to save piv files')
    args = parser.parse_args()
    tic = time.time()

    files = args.files
    files.sort()
    for file in files:
        # skip files that don't exist
        if os.path.isfile(file) == 0:
            print('WARNING: File does not exist: %s' % file)
            print('skipping...')
            continue
        else:
            continue
            

    print('Continuous image normalization (for time-resolved data)')
    print('Finding average')
    
    d = load_tif(files[0])
    frames = np.arange(2,len(files),args.frame_increment);
    for i in frames:
        d = (d*(i-1)+np.double(load_tif(files[i])))/i
            
    if args.save_name is None:
        save_name = os.path.join(os.path.dirname(files[0]),'avg_plume_img.tif')
    else:
        save_name = args.save_name[0]
        
    write_tif(save_name,d)

    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()
    

    