#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:58:08 2020

@author: andras nemes
"""


import os,sys
import argparse
import numpy as np
from Code.pio import imgio
import copy
import getpass
from datetime import datetime

def main():
  ''' Background subtract images. '''

  # parse inputs
  parser = argparse.ArgumentParser(
             description='Program to perform background subtract and normalise img files.\n\nThe input file is first normalised, then minimum values are calculated and subtracted.\n\nOutputs are min.piv and .bsub.piv files',
             formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('type', type=int,
             help='[0] all [1] pulsed [2] pulsed2')
  parser.add_argument('frame_increment', type=int, nargs='?',
                      default=1,help='Frames to increment for backsub for subsampling.\nOnly valid toy type 0')
  # parser.add_argument('method', type=int,
  #            help='[0] min [1] mean [2] median')
  parser.add_argument('files', nargs='+',
             help='Name of img files as inputs')
  args = parser.parse_args()

  for file in args.files:
    # skip files that don't exist
    if os.path.isfile(file) == 0:
      print('WARNING: File does not exist: %s' % file)
      print('skipping...')
      continue
    else:
      print('Handling File : %s' % file)

    # start piv instance
    img = imgio.imgio('%s' % file)
    img_root,img_ext = os.path.splitext(file)

    # # copy piv instance
    img2 = copy.deepcopy(img)
    img2.file_name = "%s.min.img" % img_root

    # # generate comment addition.
    # # TODO: move to pivio class
    d = datetime.now()
    img2.comment = "%s\n%s %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__), file, d.strftime("%a %b %d %H:%M:%S %Y")) + img.comment

    img2.write_header()

    # sys.stdout.write('Calculating histogram equalisation\n')
    # for i in xrange(img.it):
    #     img2.write_frame(histeq(img.read_frame(i)))
    # 	sys.stdout.write('\r' + 'Frame %04d/%04d' % (i+1,img2.it))
    #     sys.stdout.flush()

    # copy piv instance
    img3 = copy.deepcopy(img)
    img3.file_name = "%s.bsub.img" % img_root

    # generate comment addition.
    # TODO: move to pivio class
    d = datetime.now()
    img3.comment = "%s\n%s %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__), img.file_name, d.strftime("%a %b %d %H:%M:%S %Y")) + img.comment

    img3.write_header()

    if args.type == 0:
      print('Continuous background subtraction')
      print('Finding  minimum')

      d = img.max*np.ones(img.iy*img.ix,)
      # generate frame numbers to average (excluding first frame above)
      frames = np.arange(0,img.it,args.frame_increment);
      for i in frames:
        d = np.minimum(d,img.read_frame(i))
        sys.stdout.write('\r' + 'Frame %04d/%04d' % (i+2,img.it))
        sys.stdout.flush()

      img2.it = 1
      img2.write_header()
      img2.write_frame(d)

      sys.stdout.write('\nPerforming background subtraction\n')
      for i in range(img.it):
          img3.write_frame(img.read_frame(i)-d)
          sys.stdout.write('\r' + 'Frame %04d/%04d' % (i+1,img.it))
          sys.stdout.flush()

    elif args.type == 1:
      print('Paired background subtraction')
      print('Finding  minimum')

      d1 = img.max*np.ones(img.iy*img.ix,)
      d2 = img.max*np.ones(img.iy*img.ix,)
      for i in range(0,img.it,2):
          sys.stdout.write('\r' + 'Frame %04d/%04d' % (i+2,img.it))
          sys.stdout.flush()

          d1 = np.minimum(d1,img.read_frame(i))
          d2 = np.minimum(d2,img.read_frame(i+1))

      img2.it = 2
      img2.write_header()
      img2.write_frame(d1)
      img2.write_frame(d2)

      sys.stdout.write('\nPerforming background subtraction\n')
      for i in range(0,img.it,2):
          img3.write_frame(img.read_frame(i)-d1)
          img3.write_frame(img.read_frame(i+1)-d2)
          sys.stdout.write('\r' + 'Frame %04d/%04d' % (i+2,img.it))
          sys.stdout.flush()

    else:
      print('Individual background subtraction')
      sys.stdout.write('\nPerforming background subtraction\n')

      img2.it = img.it/2
      img2.write_header()

      for i in range(0,img.it,2):
          d = np.minimum(img.read_frame(i+1),img.read_frame(i))
          img2.write_frame(d)
          img3.write_frame(img.read_frame(i)-d)
          img3.write_frame(img.read_frame(i+1)-d)
          sys.stdout.write('\r' + 'Frame %04d/%04d' % (i+1,img.it))
          sys.stdout.flush()

    sys.stdout.write('\n')

def histeq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = np.histogram(im,nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(im,bins[:-1],cdf)

   return im2

if __name__ == "__main__":
  main()