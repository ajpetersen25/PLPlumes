#!/usr/bin/env python3
"""
Created on Wed Feb 19 14:31:46 2020

@author: alec
"""

import numpy as np
from PLPlumes.pio import imgio
import os
import argparse
import getpass
from datetime import datetime
from PIL import Image as p
import glob
from clint.textui import progress

def main():
    '''
    Join image files into a IMG file
    Uses PIL fork pillow for image handling and so can handle any input
    Currently only handles .img output. 
    TODO: support .btf files
    '''
    
    parser = argparse.ArgumentParser(description='Program to join image files into a single file\nInput files can be any standard image file', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('output_file', type=str, help='Output .img file')
    parser.add_argument('input_files', nargs='+', help='Input image files')
    args = parser.parse_args()

    # handle input file if glob
    if len(args.input_files)==1:
        file_list = glob.glob(args.input_files[0])
        file_list.sort()
    else:
        file_list = args.input_files
    first = file_list[0]
    
    ##### FAST METHOD: NO ERROR CHECKING #####
    img = imgio.imgio()
    img.file_name = args.output_file
    # base all information on first piv file
    # first file
    deets = p.open(first)
    if deets.mode != 'I':
        deets = deets.convert(mode='I')

    img.ix,img.iy = deets.size
    img.it = len(file_list)
    # force 8bit uint
    img.unsigned = 1
    img.bytes = 2
    img.channels = 1
    img.min = 0
    img.max = 65535

    print('Total frames to join : %d' % img.it)
    
    d = datetime.now()
    
    until = np.minimum(len(args.input_files),4)
    img.comment = "%s\n%s %s %s...\npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__),args.output_file, ' '.join(args.input_files[0:until]), d.strftime("%a %b %d %H:%M:%S %Y"))
    img.write_header()
    print(img.comment)

    f = open("%s" % img.file_name, "ab")
    f.seek(img.header_length)

    for i in progress.bar(range(0,len(file_list))):
        deets = p.open(file_list[i])

        if deets.mode != 'I;16':
            deets = deets.convert(mode='I;16')
        #if deets.mode != 'I':
        #    deets = deets.convert(mode='L')        
    
        f.write(deets.tobytes())
          
        #sys.stdout.write('\r' + 'Joining frame %04d/%04d' % (i,img.it))
        #sys.stdout.flush()
        
    print('Finished: written %s' % img.file_name)
        
if __name__ == "__main__":
    main()