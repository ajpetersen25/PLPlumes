#!/usr/bin/env python3

import os,sys
import argparse
import numpy as np
from PLPlumes.pio import imgio
import copy
import getpass
from datetime import datetime

def main():
  ''' Find average intensity in images. '''

  # parse inputs
  parser = argparse.ArgumentParser(
             description='Program to perform spatial light intensity normalization of img files.\n\nOutputs an image of mean brightness across frames. \n\nIndividual images are then normalized by the mean bright frame: F_adjusted = (F_original * mean(F_mean)/F_mean)*max_intensity/max(F_original)\n\nOutputs are avg.img and .spnorm.piv files',
             formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('type', type=int,
             help='[0] time-resolved [1] double-pulsed')
  parser.add_argument('bitdepth',type=int,
             help='bit depth (2^n), input n')
  parser.add_argument('frame_increment', type=int, nargs='?',
                      default=1,help='Frames to increment for backsub for subsampling.')
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
    img2.file_name = "%s.avg.img" % img_root

    # # generate comment addition.
    # # TODO: move to pivio class
    d = datetime.now()
    img2.comment = "%s\n%s %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__), file, d.strftime("%a %b %d %H:%M:%S %Y")) + str(img.comment,'utf-8')

    img2.write_header()

    if args.type == 0:
      print('Continuous image normalization (for time-resolved data)')
      print('Finding average')

      d = np.double(img.read_frame(0))
      # generate frame numbers to average (excluding first frame above)
      frames = np.arange(2,img.it,args.frame_increment);
      for i in frames:
        d = (d*(i-1)+np.double(img.read_frame(i)))/i
        sys.stdout.write('\r' + 'Frame %04d/%04d' % (i+2,img.it))
        sys.stdout.flush()

      img2.it = 1
      img2.write_header()
      img2.write_frame(d)
      sys.stdout.flush()

    elif args.type == 1:
      print('Double-pulsed image normalization')
      print('Part 1: spatial normalization for frames A and B')

      d1 = np.double(img.read_frame(0))
      d2 = np.double(img.read_frame(1))
      ct = 2
      for i in range(2,img.it-1,2):
        sys.stdout.write('\r' + 'Frame %04d/%04d' % (i,img.it))
        sys.stdout.flush()

        d1 = (d1*(ct-1)+np.double(img.read_frame(i)))/ct
        d2 = (d2*(ct-1)+np.double(img.read_frame(i+1)))/ct
        ct = ct+1

      img2.it = 2
      img2.write_header()
      img2.write_frame(d1)
      img2.write_frame(d2)
      sys.stdout.flush()

    else:
      print('Do nothing')

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
