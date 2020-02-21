#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:01:20 2020

@author: alec
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:09:27 2020

@author: alec
"""
from openpiv.windef import first_pass, multipass_img_deform
from openpiv.validation import local_median_val, global_val,sig2noise_val
from openpiv.filters import replace_outliers
import argparse
import os
import numpy as np

import time
from PLPlumes.pio import imgio, pivio

import multiprocessing
from itertools import repeat

from datetime import datetime 
import getpass


class Settings(object):
    pass


def tracer_piv(params):
    img,f1,f2,settings = params
    frame_a = img.read_frame2d(f1)
    frame_b = img.read_frame2d(f2)
    'first pass'
    x, y, u, v, sig2noise_ratio = first_pass(frame_a,frame_b,settings.window_sizes[0], settings.overlap[0],settings.iterations,
                                  correlation_method=settings.correlation_method, subpixel_method=settings.subpixel_method, 
                                  do_sig2noise=settings.extract_sig2noise,sig2noise_method=settings.sig2noise_method, 
                                  sig2noise_mask=settings.sig2noise_mask,)
    mask=np.full_like(x,False)
    
    # Commented out sections should be uncommented for all Plumes except the dn45 View1 plumes
    #u, v, mask_m = local_median_val( u, v, u_threshold=settings.median_threshold+1, v_threshold=settings.median_threshold+1, 
    #                                          size=1)
    u,v, mask_s2n = sig2noise_val( u, v, sig2noise_ratio, threshold = settings.sig2noise_threshold)
    u, v, mask_g = global_val( u, v, settings.MinMax_U_disp, settings.MinMax_V_disp)
    mask=mask+mask_g+mask_s2n#+mask_m
    #mask=mask+mask_g
    
    u, v = replace_outliers( u, v, method=settings.filter_method, max_iter=settings.max_filter_iteration,
                            kernel_size=1)
    
    i = 1
    for i in range(2, settings.iterations+1):
        x, y, u, v, sig2noise_ratio, mask = multipass_img_deform(frame_a, frame_b, settings.window_sizes[i-1], settings.overlap[i-1],
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
    #save_file = os.path.join(settings.save_path,os.path.splitext(os.path.basename(f1))[0]+'_piv.txt')
    #save(x, y, u, v,mask,save_file, delimiter='\t')
    return mask,u,v
    
def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for parallel plume piv',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('img_file',type=str, help='Name of .img file')
    parser.add_argument('start_frame',nargs='?',default=0,type=int, help='Frame to start separation from')
    parser.add_argument('end_frame',nargs='?', default=-1,type=int, help='Number of frames to separate')
    parser.add_argument('piv_increment',nargs='?', default=1,type=int, help='increment between piv cross-correlations')
    parser.add_argument('cores',type=int,nargs='?',default=1,help='Optional - Force number of cores for node_separate')
    args = parser.parse_args()
    img = imgio.imgio(args.img_file)
    if args.end_frame == 0:
        end_frame = img.it
    else:
        end_frame = args.end_frame
    # Settings
    settings = Settings()
    settings.window_sizes = (128,64,32)
    settings.overlap = (64,32,16)
    settings.iterations = 3
    settings.correlation_method = 'circular'
    settings.subpixel_method = 'gaussian'
    settings.extract_sig2noise = True 
    settings.sig2noise_method = 'peak2peak'
    settings.sig2noise_mask = 2
    settings.sig2noise_threshold = 1.1
    settings.MinMax_U_disp = (-30,30)
    settings.MinMax_V_disp = (-30,30)
    settings.median_threshold = 2
    settings.median_size = 3
    settings.filter_method = 'localmean'
    settings.max_filter_iteration = 3
    settings.filter_kernel_size = 3
    settings.std_threshold = 10
    settings.interpolation_order= 3
    #settings.dt = (1/600)
    #settings.scaling_factor
    

    piv_imgs = []
    for i in range(args.start_frame,end_frame-1,args.piv_increment):
        piv_imgs.append(i)
        piv_imgs.append(i+1)
    piv_imgs1 = piv_imgs[::2]
    piv_imgs2 = piv_imgs[1::2]

    f_tot = len(piv_imgs1)
    objList = list(zip(repeat(img,times=f_tot),
                       piv_imgs1,piv_imgs2,
                       repeat(settings,times=f_tot)))

    pool = multiprocessing.Pool(processes=args.cores)
    results = pool.map(tracer_piv,objList)
    results = np.array(results)
    piv = pivio.pivio(os.path.splitext(img.file_name)[0]+'.piv')
    piv.ix=img.ix
    piv.iy=img.iy
    piv.nx = results[0][0].shape[1]
    piv.ny = results[0][0].shape[0]
    piv.nt = f_tot
    piv.dx = settings.window_sizes[-1]
    piv.dy = settings.window_sizes[-1]
    d = datetime.now()
    piv.comment = piv.comment = "%s\n%s %d %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(),os.path.basename(piv.file_name), 1, piv, 
                                                     d.strftime("%a %b %d %H:%M:%S %Y")) + piv.comment
    piv.write_header()
    
    for f in range(0,f_tot):
        data = [results[f][0].flatten(),results[f][1].flatten(),results[f][2].flatten()]
        piv.write_frame(data)
    
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()
    
