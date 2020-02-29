#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:47:21 2020

@author: alec
"""

import numpy as np
from PLPlumes.pio import imgio,pivio
import time
import argparse
import multiprocessing
from itertools import repeat
import copy
from datetime import datetime 
import getpass
import os


def mp_flux_map(params):
    rhob_img,piv,slices,frame = params
    rhob_img_frame = rhob_img.read_frame2d(frame)
    piv_frame = piv.read_frame2d(frame)
    mpf_frame = np.zeros(piv_frame[0].shape)
    for s in slices:
        mpf_frame[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] = np.mean(rhob_img_frame[s])
        
    return np.flipud(piv.read_frame2d(frame)[0]), np.flipud(piv.read_frame2d(frame)[1]*mpf_frame),np.flipud(piv.read_frame2d(frame)[2]*mpf_frame)


def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for parallel plume piv',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('rho_b_file',type=str, help='Name of bulk density .img file')
    parser.add_argument('piv_file',type=str, help='Name of .piv file')
    parser.add_argument('start_frame',nargs='?',default=0,type=int, help='Frame to start separation from')
    parser.add_argument('end_frame',nargs='?', default=0,type=int, help='Number of frames to separate')
    parser.add_argument('cores',type=int,nargs='?',default=1,help='Optional - Force number of cores for node_separate')
    args = parser.parse_args()
    img = imgio.imgio(args.rho_b_file)
    piv = pivio.pivio(args.piv_file)
    if args.end_frame == 0:
        end_frame = img.it
    else:
        end_frame = args.end_frame

    x = np.arange(piv.dx/2,img.ix-piv.dx/2,piv.dx)
    y = np.arange(piv.dy/2,img.iy-piv.dy/2,piv.dy)
    slices = []
    for i in x:
       for j in y:
           slices.append((slice(int(j),int(j+piv.dy)),slice(int(i),int(i+piv.dx))))
           
    frames = np.arange(args.start_frame,end_frame)
    f_tot = len(frames)
    objList = list(zip(repeat(img,times=f_tot),
                       repeat(piv,times=f_tot),
                       repeat(slices,times=f_tot),
                       frames))
    
    pool = multiprocessing.Pool(processes=args.cores)
    results = pool.map(mp_flux_map,objList)
    results = np.array(results)
    
    piv_root,piv_ext = os.path.splitext(args.piv_file)
    piv2 = copy.deepcopy(piv)
    piv2.file_name = piv_root+'.mpf.piv'
    d = datetime.now()
    piv2.comment = "%s\n%s %d %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(),os.path.basename(__file__), 1, args.piv_file, 
                                                     d.strftime("%a %b %d %H:%M:%S %Y")) + str(piv.comment,'utf-8')
    piv2.nt = f_tot
    piv2.write_header()
    for f in range(0,f_tot):
        data = [results[f][0].flatten(),results[f][1].flatten(),results[f][2].flatten()]
        piv2.write_frame(data)
        
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()    
