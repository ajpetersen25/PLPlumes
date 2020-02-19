#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:55:39 2020

@author: alec
"""

import numpy as np
import numpy.ma as nma
from PLPlumes.pio.piv_io import load_piv
from PLPlumes.pio.apply_mask import apply_masktxt
import time
from openpiv.tools import save
import argparse



def average_masked_vel(piv_files,piv_shape):
    u_ms = []
    v_ms = []
    masks = []
    for i in range(0,len(piv_files)):
        _,_,u_m,v_m,mask = apply_masktxt(piv_files[i],piv_shape)
        u_ms.append(u_m)
        v_ms.append(v_m)
        masks.append(mask.astype('bool'))
    masks=~np.array(masks)
    u_ms = nma.masked_array(u_ms,mask=masks)
    v_ms = nma.masked_array(v_ms,mask=masks)
    u_m_avg = nma.mean(u_ms,axis=0)
    v_m_avg = nma.mean(v_ms,axis=0)
    return(u_m_avg,v_m_avg)

def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for averaging piv files',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('save_name',type=str,nargs=1,help='path to directory you want to save piv files')
    parser.add_argument('cores',type=int,nargs=1,default=1,help='number of cores to use')
    parser.add_argument('nx',default=79,type=int,nargs=?,help='piv cols')
    parser.add_argument('ny',default=49,type=int,nargs=?,help='piv rows')
    parser.add_argument('files', nargs='+', help='Name of files as inputs')
    args = parser.parse_args()
    piv_files = args.files
    piv_files.sort()
    u_m_avg,v_m_avg = average_masked_vel(piv_files, (args.ny[0],args.nx[0]))
    x,y,_,_,_ = load_piv(piv_files[0],(args.ny[0],args.nx[0]),full=True)
    
    save(x,y,u_m_avg,v_m_avg,u_m_avg.mask,args.save_name[0], delimiter='\t')
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()