#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:24:44 2020

@author: alec
"""

import os
import numpy as np
from PLPlumes.pio import imgio
from PLPlumes.plume_processing.plume_functions import phi_to_rho
import argparse
import time
import copy
import multiprocessing
from itertools import repeat

from datetime import datetime 
import getpass

def quadratic(x,a):
    return a*x**2


def linear_off(x,a,b,x0):
    return b+a*(x-x0)


def convert_frame(params):
    img,pq,pl,frame = params
    knot = pl[2]
    img_frame = img.read_frame2d(frame)
    mask = img_frame > knot
    phi_frame = ~mask*quadratic(img_frame,pq[0]) + mask*linear_off(img_frame,pl[0],pl[1],pl[2])
    rho_frame = phi_to_rho(phi_frame,2500,1.225)
    return rho_frame


def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program to save concentration .imgs',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('img_file',type=str, help='Name of .img file')
    parser.add_argument('p_quad',type=str,help='path to .txt file containing fit parameters for initial quadratic function')
    parser.add_argument('p_lin',type=str,help='path to .txt file containing fit parameters for linear function')
    parser.add_argument('start_frame',nargs='?',default=0,type=int, help='Frame to start separation from')
    parser.add_argument('end_frame',nargs='?', default=0,type=int, help='Number of frames to separate')
    parser.add_argument('cores',type=int,nargs='?',default=1,help='Optional - Force number of cores for node_separate')
    args = parser.parse_args()
    img = imgio.imgio(args.img_file)
    if args.end_frame == 0:
        end_frame = img.it
    else:
        end_frame = args.end_frame
    frames = np.arange(args.start_frames,end_frame)
    pq = np.loadtxt(args.p_quad)
    pl = np.loadtxt(args.p_lin)
    f_tot = len(frames)
    objList = list(zip(repeat(img,times=f_tot),
                       repeat(pq,times=f_tot),
                       repeat(pl,times=f_tot),
                       frames))
    pool = multiprocessing.Pool(processes=args.cores)
    results = pool.map(convert_frame,objList)
    results = np.array(results)
    
    img_root,img_ext = os.path.splitext(args.img_file)
    img2 = copy.deepcopy(img)
    img2.file_name = "%s.rho_b.img" % img_root
    d = datetime.now()
    img2.comment = "%s\n%s %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__), args.img_file, d.strftime("%a %b %d %H:%M:%S %Y")) + img.comment

    img2.write_header()
    
    for f in range(0,f_tot):
        img2.write_frame(results[f,:,:])
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()        