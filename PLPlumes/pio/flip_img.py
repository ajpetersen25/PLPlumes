#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:39:29 2021

@author: apetersen
"""


import numpy as np
from PLPlumes.pio import imgio
import os
import argparse
import getpass
from datetime import datetime
import copy
import sys

def main():
    '''
    Join image files into a IMG file
    Uses PIL fork pillow for image handling and so can handle any input
    Currently only handles .img output. 
    TODO: support .btf files
    '''
    
    parser = argparse.ArgumentParser(description='Program to flip images in an img file', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file',type=str, nargs=1, help='input .img')
    args = parser.parse_args()

    img = imgio.imgio(args.input_file[0])
    
    ##### FAST METHOD: NO ERROR CHECKING #####
    # base all information on first piv file
    # first file
    img2 = copy.deepcopy(img)
    img2 = imgio.imgio(os.path.splitext(img.file_name)[0]+'.flip.img')
    img2.ix = img.ix
    img2.iy = img.iy
    img2.it = img.it
    img2.type='uint16'
    img2.bytes = 2
    img2.min = 0 
    img2.max = np.max(65535)
    img2.unsigned = 1
    img2.channels = 1
    d = datetime.now()
    img2.comment = "%s\n%s %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__), ' ', d.strftime("%a %b %d %H:%M:%S %Y")) + str(img.comment,'utf-8')


    print('Total frames to flip : %d\n' % img.it)
    
    
    
    img2.write_header()
    for i in range(0,img.it):
        #data = np.fliplr(np.flipud(img.read_frame2d(i)))
        data = img.read_frame2d(i)
        img2.write_frame(np.flipud(data))
        sys.stdout.write('\r' + 'Frame %04d/%04d' % (i+2,img.it))
        sys.stdout.flush()

        
    print('Finished: written %s' % img.file_name)
        
if __name__ == "__main__":
    main()
