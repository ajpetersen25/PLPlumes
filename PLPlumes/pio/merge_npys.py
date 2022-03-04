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
    parser = argparse.ArgumentParser(description='Program to join npy files', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('output_file', type=str, help='Name of output npz file')
    parser.add_argument('npy_files', nargs='+', help='Name of npy files to join')
    args = parser.parse_args()

    # Total number of files
    nt = 0

    # list of files in list to process
    do_img  = np.ones(len(args.npy_files),)
    
    # first file
    f = args.npy_files[0]
    it = len(np.load(args.npy_files[0],allow_pickle=True))

    # first loop checks for total frames and error checking
    for i in range(1,len(args.npy_files)):
        t = np.load(args.npy_files[i],allow_pickle=True)
            
        if len(np.load(args.npy_files[i],allow_pickle=True)) < 1:
            print('[WARNING] piv %s has no frames... skipping' % (args.npy_files[i]))
            do_img[i] = 0
        else:
            it = it + len(t)

    print('Total frames to join : %d' % it)
    
    # # base all information on first piv file
    #img2 = copy.deepcopy(img)
    #img2.file_name = "%s" % (args.output_file)
    #img2.it = it
    d = datetime.now()
    #img2.comment = "%s\n%s %s %s\npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__),args.output_file, ' '.join(args.img_files), d.strftime("%a %b %d %H:%M:%S %Y")) + str(img.comment,'utf-8')
    #img2.write_header()

    #f2 = open("%s" % img2.file_name, "ab")
    
    # second loop writes frames
    kk = 0
    all_files = []
    for i in range(len(args.npy_files)):
        if do_img[i]:
            file = np.load(args.npy_files[i],allow_pickle=True)
            all_files.extend(np.squeeze(file))
    #all_files = np.array(all_files,dtype=object)
    #all_files = all_files.reshape(it,3)
    all_files = np.array(all_files)
    np.savez(args.output_file,all_files)
            #f = open("%s" % t.file_name,'rb')
            #f.seek(t.header_length)
            #for k in range(t.it):
            #    kk = kk+1
            #    f2.write(f.read(4*t.nx*t.ny*t.cols))
            #    sys.stdout.write('\r' + 'Joining frame %04d/%04d' % (kk,piv2.nt))
            #    sys.stdout.flush()
if __name__ == "__main__":
  main() 

        
