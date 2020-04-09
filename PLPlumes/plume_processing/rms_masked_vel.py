#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:55:39 2020

@author: alec
"""

import numpy as np
import numpy.ma as nma
from PLPlumes.pio import pivio
#from PLPlumes.pio import apply_mask
import time
import argparse
import os
import copy
from datetime import datetime
import getpass


def rms_masked_vel(piv):
    u = []
    v = []
    masks = []
    for f in range(0,piv.nt):
        piv_frame = piv.read_frame2d(f)
        u.append(piv_frame[1])
        v.append(piv_frame[2])
        masks.append(piv_frame[0])
    masks = np.array(masks).astype('bool')
    masked_u = nma.masked_array(u,mask=~masks)
    masked_v = nma.masked_array(v,mask=~masks)


    u_m_rms = np.sqrt(nma.mean(masked_u**2,axis=0))
    v_m_rms = np.sqrt(nma.mean(masked_v**2,axis=0))
    return(np.flipud(~u_m_rms.mask),np.flipud(u_m_rms),np.flipud(v_m_rms))

def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for calculating rms from fluctuating piv file',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('piv_file',type=str,help='.flct.piv file')
    args = parser.parse_args()
    piv = pivio.pivio(args.piv_file)
    
    rms_mask, u_m_rms,v_m_rms = rms_masked_vel(piv)

    piv_root,piv_ext = os.path.splitext(args.piv_file)
    piv2 = copy.deepcopy(piv)
    piv2.file_name = piv_root+'.rms.piv'
    d = datetime.now()
    piv2.comment = "%s\n%s %d %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(),os.path.basename(__file__), 1, args.piv_file, 
                                                     d.strftime("%a %b %d %H:%M:%S %Y")) + str(piv.comment,'utf-8')
    piv2.nt = 1
    piv2.write_header()
    data = [rms_mask.flatten(),u_m_rms.flatten(),v_m_rms.flatten()]
    piv2.write_frame(data)
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()