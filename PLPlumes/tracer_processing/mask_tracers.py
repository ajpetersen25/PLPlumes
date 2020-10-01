#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:34:15 2020

@author: ajp25
"""
import time
import numpy.ma as nma
import numpy as np
from PLPlumes.pio import pivio,imgio
#from PLPlumes.pio import apply_mask
import time
import argparse
import os
import copy
from datetime import datetime
import getpass



def mask_tracer_piv(param):
    
    step = (piv.dx,piv.dy)
    masks = []
    u = []
    v = []
    f = frame
    piv_frame = piv.read_frame2d(f)
    img_frame = img.read_frame2d(f)
    frame_mask = masks[f]
    mask = piv.read_frame2d(f)[0]
    mask[mask==4]=1
    mask[mask!=1]=0
    # loop through all interrogation windows, looking at which meet threshold criteria
    # elements in the frame mask array which meet this criteria are set to 0
    for s in slices:
            window = img_frame[s]
            if np.sum(window==threshold)/(step[0]*step[1]) < window_threshold and mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] ==1:
                mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] = 1
            else:
                mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)]  = 0
    u.append(piv_frame[1])
    v.append(piv_frame[2])
    masks.append(mask)
    
    masks = np.array(masks).astype('bool')
    masked_u = nma.masked_array(u,mask=~masks)
    masked_v = nma.masked_array(v,mask=~masks)          
    u_m_avg = nma.mean(masked_u,axis=0)
    v_m_avg = nma.mean(masked_v,axis=0)
    return(np.flipud(~u_m_avg.mask),np.flipud(u_m_avg),np.flipud(v_m_avg))


def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for averaging piv files',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('piv_file',type=str,help='.piv file you want to average')
    args = parser.parse_args()
    piv = pivio.pivio(args.piv_file)
    
    
    x = np.arange(piv.dx,img.ix-piv.dx,piv.dx)
    y = np.arange(piv.dy,img.iy-piv.dy,piv.dy)
    slices = []
    for i in x:
       for j in y:
           slices.append((slice(int(j),int(j+piv.dy)),slice(int(i),int(i+piv.dx))))
    avg_mask, u_m_avg,v_m_avg = average_masked_vel(piv)

    piv_root,piv_ext = os.path.splitext(args.piv_file)
    piv2 = copy.deepcopy(piv)
    piv2.file_name = piv_root+'.ave.piv'
    d = datetime.now()
    piv2.comment = "%s\n%s %d %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(),os.path.basename(__file__), 1, args.piv_file, 
                                                     d.strftime("%a %b %d %H:%M:%S %Y")) + str(piv.comment,'utf-8')
    piv2.nt = 1
    piv2.write_header()
    data = [avg_mask.flatten(),u_m_avg.flatten(),-v_m_avg.flatten()]
    piv2.write_frame(data)
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()