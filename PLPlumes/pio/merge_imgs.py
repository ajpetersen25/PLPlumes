#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:39:16 2020

@author: alec
"""

import numpy as np
from PLPlumes.pio import imgio
import copy
import os,sys
import argparse
import getpass
from datetime import datetime

def main():
    '''Join piv files together'''
    parser = argparse.ArgumentParser(description='Program to join PIV files', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('output_file', type=str, help='Name of output PIV file')
    parser.add_argument('img_files', nargs='+', help='Name of PIV files to join')
    args = parser.parse_args()

    # Total number of frames
    nt = 0

    # list of files in list to process
    do_img  = np.ones(len(args.img_files),)
    
    # first file
    img = imgio.imgio(args.img_files[0])
    if img.it < 1:
        print('[WARNING] piv %s has no frames... skipping' % (img.file_name))
        do_img[0] = 0
    else:
        it = img.it

    # first loop checks for total frames and error checking
    for i in range(1,len(args.img_files)):
        t = imgio.imgio(args.img_files[i])
        if t.ix != img.ix or t.iy != img.iy:
            print('[ERROR] img %s and %s files have different shapes' % (img.file_name,t.file_name))
            print('QUITTING')
            os.sys.exit(1)

            
        if t.it < 1:
            print('[WARNING] piv %s has no frames... skipping' % (t.file_name))
            do_img[i] = 0
        else:
            it = it + t.it

    print('Total frames to join : %d' % it)
    
    # # base all information on first piv file
    img2 = copy.deepcopy(img)
    img2.file_name = "%s" % (args.output_file)
    img2.it = it
    d = datetime.now()
    img2.comment = "%s\n%s %s %s\npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__),args.output_file, ' '.join(args.img_files), d.strftime("%a %b %d %H:%M:%S %Y")) + str(img.comment,'utf-8')
    img2.write_header()

    f2 = open("%s" % img2.file_name, "ab")
    
    # second loop writes frames
    kk = 0
    for i in range(len(args.img_files)):
        if do_img[i]:
            t = imgio.imgio(args.img_files[i])
            for k in range(t.it):
                kk = kk+1
                img2.write_frame(t.read_frame(k))
                sys.stdout.write('\r' + 'Joining frame %04d/%04d' % (kk,img2.it))
                sys.stdout.flush()
            #f = open("%s" % t.file_name,'rb')
            #f.seek(t.header_length)
            #for k in range(t.it):
            #    kk = kk+1
            #    f2.write(f.read(4*t.nx*t.ny*t.cols))
            #    sys.stdout.write('\r' + 'Joining frame %04d/%04d' % (kk,piv2.nt))
            #    sys.stdout.flush()
        
