#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:39:16 2020

@author: alec
"""

import numpy as np
from PLPlumes.pio import pivio as pivio
import copy
import os,sys
import argparse
import getpass
from datetime import datetime

def main():
    '''Join piv files together'''
    parser = argparse.ArgumentParser(description='Program to join PIV files', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('output_file', type=str, help='Name of output PIV file')
    parser.add_argument('piv_files', nargs='+', help='Name of PIV files to average')
    args = parser.parse_args()

    # Total number of frames
    nt = 0

    # list of files in list to process
    do_piv  = np.ones(len(args.piv_files),)
    
    # first file
    piv = pivio.pivio(args.piv_files[0])
    if piv.nt < 1:
        print('[WARNING] piv %s has no frames... skipping' % (piv.file_name))
        do_piv[0] = 0
    else:
        nt = piv.nt

    # first loop checks for total frames and error checking
    for i in range(1,len(args.piv_files)):
        t = pivio.pivio(args.piv_files[i])
        if t.nx != piv.nx or t.ny != piv.ny:
            print('[ERROR] piv %s and %s files have different number of vectors' % (piv.file_name,t.file_name))
            print('QUITTING')
            os.sys.exit(1)

        if t.cols != piv.cols:
            print('[ERROR] piv %s and %s files have different number of columns' % (piv.file_name,t.file_name))
            print('QUITTING')
            os.sys.exit(1)
            
        if t.nt < 1:
            print('[WARNING] piv %s has no frames... skipping' % (t.file_name))
            do_piv[i] = 0
        else:
            nt = nt + t.nt

    print('Total frames to join : %d' % nt)
    
    # # base all information on first piv file
    piv2 = copy.deepcopy(piv)
    piv2.file_name = "%s" % (args.output_file)
    piv2.nt = nt
    d = datetime.now()
    piv2.comment = "%s\n%s %s %s\npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__),args.output_file, ' '.join(args.piv_files), d.strftime("%a %b %d %H:%M:%S %Y")) + str(piv.comment,'utf-8')
    piv2.write_header()

    f2 = open("%s" % piv2.file_name, "ab")
    
    # second loop writes frames
    kk = 0
    for i in range(len(args.piv_files)):
        if do_piv[i]:
            t = pivio.pivio(args.piv_files[i])
            f = open("%s" % t.file_name,'rb')
            f.seek(t.header_length)
            for k in range(t.nt):
                kk = kk+1
                f2.write(f.read(4*t.nx*t.ny*t.cols))
                sys.stdout.write('\r' + 'Joining frame %04d/%04d' % (kk,piv2.nt))
                sys.stdout.flush()
        

        
if __name__ == "__main__":
    main()
