#!/usr/bin/env python3

import os,sys
import argparse
import numpy as np
from PLPlumes.pio import imgio
import copy
import getpass
from datetime import datetime

def main():
  ''' Find fluctuating intensity in images. '''

  # parse inputs
  parser = argparse.ArgumentParser(
             description='Program to perform spatial light intensity normalization of img files.\n\nOutputs an image of mean brightness across frames. \n\nIndividual images are then normalized by the mean bright frame: F_adjusted = (F_original * mean(F_mean)/F_mean)*max_intensity/max(F_original)\n\nOutputs are avg.img and .spnorm.piv files',
             formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('bitdepth',type=int,
             help='bit depth (2^n), input n')
  parser.add_argument('frame_increment', type=int, nargs='?',
                      default=1,help='Frames to increment for backsub for subsampling.')
  parser.add_argument('avg_file',nargs='+',help='average file')
  parser.add_argument('files', nargs='+',
             help='Name of img files as inputs')
  args = parser.parse_args()

  for fnum,file in enumerate(args.files):
    avg_img = imgio.imgio('%s' %args.avg_file[fnum])
    # skip files that don't exist
    if os.path.isfile(file) == 0:
      print('WARNING: File does not exist: %s' % file)
      print('skipping...')
      continue
    else:
      print('Handling File : %s' % file)

    # start piv instance
    img = imgio.imgio('%s' % file)
    img_root = os.path.splitext(img.file_name)[0]
    img2 = copy.deepcopy(img)
    img2.file_name = "%s" % (img_root+'.flct.img')
    img2 = imgio.imgio(os.path.splitext(img.file_name)[0]+'.flct.img')
    img2.ix = img.ix
    img2.iy = img.iy
    img2.it = img.it
    img2.type='float32'
    img2.bytes = 4
    img2.min = 0 
    img2.max = np.max(4095)
    img2.unsigned = 1
    img2.channels = 1
    d = datetime.now()
    img2.comment = "%s\n%s %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__), file, d.strftime("%a %b %d %H:%M:%S %Y")) + str(img.comment,'utf-8')



    print('Continuous image normalization (for time-resolved data)')
    print('Finding average')
    
    avg = avg_img.read_frame(0)
    img2.write_header()
    print('Subtracting average')
    for i in range(0,img.it):
        data = img.read_frame(i) - avg
        img2.write_frame(data)
        sys.stdout.write('\r' + 'Frame %04d/%04d' % (i+2,img.it))
        sys.stdout.flush()
    
    sys.stdout.flush()



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
