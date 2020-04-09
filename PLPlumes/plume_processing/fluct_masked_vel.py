#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:21:19 2020

@author: ajp25
"""

import numpy as np
import numpy.ma as nma
from PLPlumes.pio import pivio
from datetime import datetime
import getpass
from PLPlumes.plume_processing.avg_masked_vel import average_masked_vel
import time
import argparse
import multiprocessing
from itertools import repeat
import copy
import os


def fluct_masked(params):
    piv,ave_mvel,f = params

    fluct_u = (piv.read_frame2d(f)[1] - ave_mvel[1])
    fluct_v = (piv.read_frame2d(f)[2] - ave_mvel[2])
    mask = piv.read_frame2d(f)[0].astype('bool')


    return(np.flipud(mask),np.flipud(fluct_u),np.flipud(fluct_v))
    
  
def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for parallel plume piv',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('piv_file',type=str, help='Name of .piv file')
    parser.add_argument('start_frame',nargs='?',default=0,type=int, help='Frame to start separation from')
    parser.add_argument('end_frame',nargs='?', default=0,type=int, help='Number of frames to separate')
    parser.add_argument('cores',type=int,nargs='?',default=1,help='Optional - Force number of cores for node_separate')
    args = parser.parse_args()
    piv = pivio.pivio(args.piv_file)
    if args.end_frame == 0:
        end_frame = piv.nt
    else:
        end_frame = args.end_frame

    avg = average_masked_vel(piv)
           
    frames = np.arange(args.start_frame,end_frame)
    f_tot = len(frames)
    objList = list(zip(repeat(piv,times=f_tot),
                       repeat(avg,times=f_tot),
                       frames))
    
    pool = multiprocessing.Pool(processes=args.cores)
    results = pool.map(fluct_masked,objList)
    results = np.array(results)
    
    piv_root,piv_ext = os.path.splitext(args.piv_file)
    piv2 = copy.deepcopy(piv)
    piv2.file_name = piv_root+'.flct.piv'
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