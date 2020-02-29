#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:55:39 2020

@author: alec
"""

import numpy as np
import numpy.ma as nma
from PLPlumes.pio import pivio
from PLPlumes.pio import apply_mask
import time
from openpiv.tools import save
import argparse



def average_masked_vel(piv):
    u = []
    v = []
    masks = []
    for f in range(0,piv.nt):
        piv_frame = piv.read_frame2d(f)
        u.append(piv_frame[1])
        v.append(piv_frame[2])
        masks.append(piv_frame[0])
    masks = np.array(masks).astype('bool')
    masked_ u = nma.masked_array(u,mask=~masks)
    masked_ v = nma.masked_array(v,mask=~masks)


    u_m_avg = nma.mean(masked_u,axis=0)
    v_m_avg = nma.mean(masked_v,axis=0)
    return(u_m_avg.mask,u_m_avg,v_m_avg)

def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for averaging piv files',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('piv_file',type=str,help='.piv file you want to average')
    args = parser.parse_args()
    piv = pivio.pivio(args.piv_file)
    
    avg_mask, u_m_avg,v_m_avg = average_masked_vel(piv)

    piv_root,piv_ext = os.path.splitext(args.piv_file)
    piv2 = copy.deepcopy(piv)
    piv2.file_name = piv_root+'.ave.piv'
    d = datetime.now()
    piv2.comment = "%s\n%s %d %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(),os.path.basename(__file__), 1, args.piv_file, 
                                                     d.strftime("%a %b %d %H:%M:%S %Y")) + str(piv.comment,'utf-8')
    piv2.nt = 1
    piv2.write_header()
    data = [avg_mask.flatten(),u_m_avg.flatten(),v_m_mask.flatten()]
    piv2.write_frame(data)
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()