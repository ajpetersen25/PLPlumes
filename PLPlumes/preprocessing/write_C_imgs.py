#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:24:44 2020

@author: alec
"""

import os
import numpy as np
from PLPlumes.pio import imgio
from PLPlumes.plume_processing.plume_functions import phi_to_rho,windowed_average
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
    img,pq,pl,start_height,frame,orientation = params
    img_frame = img.read_frame2d(frame).astype('float32')
    phi_frame = np.zeros(img_frame.shape).astype('float32')
    pq = windowed_average(np.array(pq),50)
    pl[:,0] = windowed_average(np.array(pl)[:,0],50)
    pl[:,1] = windowed_average(np.array(pl)[:,1],50)
    pl[:,2] = windowed_average(np.array(pl)[:,2],50)
    if orientation == 'vert':
        h = img.iy
        for c in range(0,h):
            if c<start_height:
                pass
            else:
                knot = pl[c,2]
                mask = img_frame[c,:] > knot
                phi_frame[c,:] = ~mask*quadratic(img_frame[c,:],pq[c]) + mask*linear_off(img_frame[c,:],pl[c,0],pl[c,1],pl[c,2])
            
    elif orientation =='horz':
        h = img.ix
        for c in range(0,h):
            if c<start_height:
                pass
            else:
                knot = pl[c,2]
                mask = img_frame[:,c] > knot
                phi_frame[:,c] = ~mask*quadratic(img_frame[:,c],pq[c]) + mask*linear_off(img_frame[:,c],pl[c,0],pl[c,1],pl[c,2])
            
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
    parser.add_argument('start_height',type=int,nargs='?',default=0,help='frame column to start conversion at')
    parser.add_argument('start_frame',nargs='?',default=0,type=int, help='Frame to start separation from')
    parser.add_argument('end_frame',nargs='?', default=0,type=int, help='Number of frames to separate')
    parser.add_argument('cores',type=int,nargs='?',default=1,help='Optional - Force number of cores for node_separate')
    parser.add_argument('orientation',type=str,nargs='?',default='horz',help='orientation of images, input horz or vert as str as appropriate')
    args = parser.parse_args()
    img = imgio.imgio(args.img_file)
    if args.end_frame == 0:
        end_frame = img.it
    else:
        end_frame = args.end_frame
    frames = np.arange(args.start_frame,end_frame)
    print(args.p_quad,args.p_lin)
    pq = np.loadtxt(args.p_quad)
    pl = np.loadtxt(args.p_lin)
    f_tot = len(frames)
    objList = list(zip(repeat(img,times=f_tot),
                       repeat(pq,times=f_tot),
                       repeat(pl,times=f_tot),
                       repeat(args.start_height,times=f_tot),
                       frames,
		               repeat(args.orientation,times=f_tot)))
    pool = multiprocessing.Pool(processes=args.cores)
    results = pool.map(convert_frame,objList)
    results = np.array(results)
    img_root = os.path.splitext(img.file_name)[0]
    img2 = copy.deepcopy(img)
    img2.file_name = "%s" % (img_root+'.rho_b.img')
    img2 = imgio.imgio(os.path.splitext(img.file_name)[0]+'.rho_b.img')
    img2.ix = img.ix
    img2.iy = img.iy
    img2.it = img.it
    img2.it = f_tot
    img2.type='float32'
    img2.bytes = 4
    img2.min = 0 
    img2.max = np.max(results[:,:,:])
    img2.unsigned = 1
    img2.channels = 1
    d = datetime.now()
    img2.comment = "%s\n%s %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__), args.img_file, d.strftime("%a %b %d %H:%M:%S %Y")) + str(img.comment,'utf-8')

    img2.write_header()
    for f in range(0,f_tot):
        img2.write_frame(np.flipud(results[f,:,:]))
    #img2.it = f_tot
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()        
