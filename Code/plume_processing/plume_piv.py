#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:09:27 2020

@author: alec
"""
from openpiv.windef import first_pass, multipass_img_deform
from openpiv.validation import local_median_val, global_val, replace_outliers,sig2noise_val
from openpiv.tools import save
import argparse
import os
import numpy as np

import time
from Code.io.image_io import load_tif

import multiprocessing
from itertools import repeat


class Settings(object):
    pass


def plume_piv(params):
    f1,f2,settings = params
    frame_a = load_tif(f1)
    frame_b = load_tif(f2)
    'first pass'
    x, y, u, v, sig2noise_ratio = first_pass(frame_a,frame_b,settings.window_sizes[0], settings.overlap[0],settings.iterations,
                                  correlation_method=settings.correlation_method, subpixel_method=settings.subpixel_method, 
                                  do_sig2noise=settings.extract_sig2noise,sig2noise_method=settings.sig2noise_method, 
                                  sig2noise_mask=settings.sig2noise_mask,)
    
    mask=np.full_like(x,False)
    
    u, v, mask_m = local_median_val( u, v, u_threshold=settings.median_threshold, v_threshold=settings.median_threshold, 
                                               size=settings.median_size)
    u, v, mask_g = global_val( u, v, settings.MinMax_U_disp, settings.MinMax_V_disp)
    u,v, mask_s2n = sig2noise_val( u, v, sig2noise_ratio, threshold = settings.sig2noise_threshold)
    mask=mask+mask_g+mask_m+mask_s2n
    
    u, v = replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, 
                                    kernel_size=settings.filter_kernel_size)
    
    i = 1
    for i in range(2, settings.iterations+1):
        x, y, u, v, sig2noise_ratio, mask = multipass_img_deform(frame_a, frame_b, settings.windowsizes[i-1], settings.overlap[i-1],
                    settings.iterations,i,x, y, u, v, correlation_method=settings.correlation_method,
                    subpixel_method=settings.subpixel_method, do_sig2noise=settings.extract_sig2noise,
                    sig2noise_method=settings.sig2noise_method, sig2noise_mask=settings.sig2noise_mask,
                    MinMaxU=settings.MinMax_U_disp,
                    MinMaxV=settings.MinMax_V_disp,std_threshold=settings.std_threshold,
                    median_threshold=settings.median_threshold,median_size=settings.median_size,filter_method=settings.filter_method,
                    max_filter_iteration=settings.max_filter_iteration, filter_kernel_size=settings.filter_kernel_size,
                    interpolation_order=settings.interpolation_order)    
        
    u,v, mask_s2n = sig2noise_val( u, v, sig2noise_ratio, threshold = settings.sig2noise_threshold)
    mask=mask+mask_s2n
    u, v = replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration, 
                                    kernel_size=settings.filter_kernel_size)
    'pixel/frame->pixel/sec'
    #u=u/settings.dt
    #v=v/settings.dt
    save(x, y, u, v,sig2noise_ratio, mask ,os.path.join(settings.save_path,
                                                        os.path.splitext(os.path.basename(frame_a))[0]+'.txt'), delimiter='\t')

def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for parallel plume piv',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('save_path',type=str,nargs=1,default='./',help='path to directory you want to save _bsub images')
    parser.add_argument('cores',type=int,nargs=1,default=1,help='number of cores to use')
    parser.add_argument('files', nargs='+', help='Name of files as inputs')
    args = parser.parse_args()
    
    img_files = args.files#glob.glob(args.path_to_image_files)   
    
    # Settings
    settings = Settings()
    settings.window_sizes = (256,128,64)
    settings.overlap = (128,64,32)
    settings.iterations = 3
    settings.correlation_method = 'circular'
    settings.subpixel_method = 'gaussian'
    settings.do_sig2noise = True
    settings.sig2noise_method = 'peak2peak'
    settings.sig2noise_mask = 2
    settings.sig2noise_threshold = 1.01
    settings.MinMax_U_disp = (-200,200)
    settings.MinMax_V_disp = (-100,100)
    settings.median_threshold = 2
    settings.median_size = 3
    settings.filter_method = 'localmean'
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 3
    settings.std_threshold = 10
    settings.interpolation_order= 3
    #settings.dt = (1/600)
    #settings.scaling_factor
    settings.save_path = args.save_path
    
    img_files1 = img_files[::2]
    img_files2 = img_files[1::2]

    f_tot = len(img_files1)-1
    objList = list(zip(img_files1,img_files2,
       repeat(settings,times=f_tot)))

    pool = multiprocessing.Pool(processes=args.cores)
    
    pool.map(plume_piv,objList)
    
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()
    