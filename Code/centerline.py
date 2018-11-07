from __future__ import division
import numpy as np
import numpy.ma as nma
import imgio
from scipy.signal import find_peaks
import cv2

import os,sys
import copy
import multiprocessing
import psutil
from itertools import repeat
import warnings
import time
import argparse
import getpass

def find_centerline(params):
    # Method options:
    # moment, outline, mixture
    # returns a Nx2 array where first column is the vertical pixel location and the second
    # column is horizontal location of the plume centerline

    img,method,kernel,threshold,frame = params;
    imf = img.read_frame2d(frame)
    rc = getattr(sys.modules[__name__],'centerline_%s'% method)(imf, kernel, threshold)
    #rc = globals()["centerline_%s" %method](imf, kernel, threshold)
    return rc

def centerline_moment(frame,kernel,threshold):
    center = []
    heights = []
    for h in np.arange(frame.shape[0]-1,0,-kernel):
        M2 = cv2.moments(frame[h-kernel:h,:].astype('uint16'))
        cx = int(M2['m10']/M2['m00'])
        center.append(cx)
        heights.append(h-(kernel-1)/2)
    heights = np.array(heights).reshape(len(heights),1)
    center = np.array(center).reshape(len(center),1)
    return np.concatenate((heights,center),axis=1)
        
def centerline_outline(frame,kernel,threshold):
    img_outline = img>threshold
    if kernel < 2:
        kernel = 2;
    center = []
    heights = []
    for h in np.arange(0,frame.shape[0],kernel):
        image, contours, hierarchy =   cv2.findContours(img_outline[h:h+kernel2,:].copy().astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        c_size = ([c.size for c in contours])
        lc = np.where(c_size == np.max(c_size))[0][0]
        y = contours[lc]
        M = cv2.moments(y)
        cx = int(M['m10']/M['m00'])
        center.append(cx)
        heights.append(h-(kernel-1)/2)
    heights = np.array(heights).reshape(len(heights),1)
    center = np.array(center).reshape(len(center),1)
    return np.concatenate((heights,center)).reshape(len(heights),2)

def centerline_mixture(frame,kernel,threshold):
    center = []
    heights = []
    for h in np.arange(frame.shape[0]-1,0,-kernel):
        y = frame[h,:]
        ps = find_peaks(y,200,prominence=50)
        w = y[ps[0]]/np.sum(y[ps[0]])
        u = ps[0]
        center.append(np.sum(w*u))
        heights.append(h-(kernel-1)/2)
    heights = np.array(heights).reshape(len(heights),1)
    center = np.array(center).reshape(len(center),1)
    return np.concatenate((heights,center)).reshape(len(heights),2)

def main():
    """
    
Parallelized function call to find the centerline of a plume for multiple images.
There are 3 implemented methods:
1) moment  ---  uses cv2 python library to calculate moment of image at a certain height, which is used to find the horizontal component of the centroid
    inputs: frame --  numpy array representing the image
            kernel--  int representing vertical interval over which to calculate the centroid
2) outline --- binarizes images using the inputted threshold. cv2 python library takes in the binarized image and creates outlines, the largest of which
                is assumed to be the plume outline. The centroid of the outline for each height according to the kernel size is then computed. 
               MINIMUM KERNEL SIZE = 2
    inputs: frame     --  numpy array representing the image
            kernel    --  int representing vertical interval over which to calculate the centroid
            threshold --  int threshold for binarizing image frame
3) mixture --- assumes plume intensity profile can be modeled using a Gaussian Mixture Model. Weighs each peak according to it's amplitude and uses it
                to find a weighted average of the horizontal locations of each peak
    inputs: frame --  numpy array representing the image
            kernel--  int representing vertical interval over which to calculate the centroid
            
            
    """
    parser = argparse.ArgumentParser(
             description='Program to calculate the centerline of a plume based on the raw image', 
             formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('img_file',type=str,nargs=1,help='Path to .img file')
    parser.add_argument('method',type=str,nargs=1,help='Method choice: moment, outline or mixture')
    parser.add_argument('kernel',type=int,nargs=1,help='vertical window over which to calculate the centroid')
    parser.add_argument('threshold',type=int,nargs=1,help='image intensity threshold')
    parser.add_argument('start_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to start the masking at')
    parser.add_argument('end_frame',type=int,nargs='?',default=0,help='Frame of PIV you want to end the masking at')
    args = parser.parse_args()
    
    fail = False
    # check if IMG file exists
    if os.path.exists(args.img_file[0]) == 0:
        print '[ERROR] file does not exist'
        fail = True
        
    img = imgio.imgio(args.img_file[0])
    
    if fail:
        print 'exiting...'
        os.sys.exit(1)
    if args.end_frame == 0:
        end_frame = img.it
    # start up parallel pool
    avail_cores = int(psutil.cpu_count() - (psutil.cpu_count()/2))
    
    param1 = img
    param2 = args.method[0]
    param3 = args.kernel[0]
    param4 = args.threshold[0]
    param5 = range(args.start_frame,end_frame)
    f_tot = len(param5)
    objList = zip(repeat(param1,times=f_tot),
                  repeat(param2,times=f_tot),
                  repeat(param3,times=f_tot),
                  repeat(param4,times=f_tot),
                  param5)
    pool = multiprocessing.Pool(processes=avail_cores)
    # process
    tic = time.time()
    results = pool.map(find_centerline,objList)
    results = np.asarray(results)
    np.savez_compressed(os.path.splitext(img.file_name)[0]+'.img_centerline.npz',results)
    print '[FINISHED]: %f seconds elapsed' %(time.time()-tic)
    
if __name__ == "__main__":
    if sys.argv[1]=='-help' or sys.argv[1]=='-h':
        print(main.__doc__)
        main()
    else:
        main()
