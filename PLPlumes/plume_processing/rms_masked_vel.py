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


def rms_masked(piv,frames):

    u = []
    v = []
    masks = []
    for f in frames:
        piv_frame = piv.read_frame2d(f)
        u.append(piv_frame[1]**2)
        v.append(piv_frame[2]**2)
        masks.append(piv_frame[0])
    masks = np.array(masks).astype('bool')
    masked_u = nma.masked_array(u,mask=~masks)
    masked_v = nma.masked_array(v,mask=~masks)


    u_m_rms = np.sqrt(nma.mean(masked_u,axis=0))
    v_m_rms = np.sqrt(nma.mean(masked_v,axis=0))
    return(np.flipud(~u_m_rms.mask),np.flipud(u_m_rms),np.flipud(v_m_rms))
    
  
def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for parallel plume piv',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('piv_file',type=str, help='Name of fluctuating .piv file')
    parser.add_argument('start_frame',nargs='?',default=0,type=int, help='Frame to start separation from')
    parser.add_argument('end_frame',nargs='?', default=0,type=int, help='Number of frames to separate')
    args = parser.parse_args()
    piv = pivio.pivio(args.piv_file)
    if args.end_frame == 0:
        end_frame = piv.nt
    else:
        end_frame = args.end_frame
    frames = np.arange(args.start_frame,end_frame)
           
    rms_mask, u_m_rms,v_m_rms = rms_masked(piv,frames)

    piv_root,piv_ext = os.path.splitext(args.piv_file)
    piv2 = copy.deepcopy(piv)
    piv2.file_name = piv_root+'.rms.piv'
    d = datetime.now()
    piv2.comment = "%s\n%s %d %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(),os.path.basename(__file__), 1, args.piv_file, 
                                                     d.strftime("%a %b %d %H:%M:%S %Y")) + str(piv.comment,'utf-8')
    piv2.nt = 1
    piv2.write_header()
    data = [rms_mask.flatten(),u_m_rms.flatten(),-v_m_rms.flatten()]
    piv2.write_frame(data)
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()    