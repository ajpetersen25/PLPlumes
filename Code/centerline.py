from __future__ import division
import numpy as np
import numpy.ma as nma
from pio import imgio
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
    frame[frame<threshold] = 0
    for h in np.arange(0,frame.shape[0]-1,kernel):
        M2 = cv2.moments(frame[h:h+kernel,:].astype('uint16'))
        cx = int(M2['m10']/M2['m00'])
        center.append(cx)
        heights.append(h+(kernel-1)/2 + .5)
    heights = np.array(heights).reshape(len(heights),1)
    center = np.array(center).reshape(len(center),1)
    return np.concatenate((heights,center),axis=1)
        
def centerline_outline(frame,kernel,threshold):
    img_outline = frame>threshold
    image, contours, hierarchy =   cv2.findContours(img_outline.copy().astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    c_size = ([c.size for c in contours])
    lc = np.where(c_size == np.max(c_size))[0][0]
    plume_contour =cv2.drawContours(image,contours[lc],-1, 255, 1)
    center = []
    heights = []
    for h in np.arange(0,frame.shape[0]-1,kernel):
        c = np.where(plume_contour[h:h+kernel,:]==1)[1]
        if len(c)!=0:
            cx=c.mean()
            center.append(cx)
        else:
            center.append(0)
        heights.append(h+(kernel-1)/2 + .5)
    heights = np.array(heights).reshape(len(heights),1)
    center = np.array(center).reshape(len(center),1)
    return np.concatenate((heights,center),axis=1)

def centerline_mixture(frame,kernel,threshold):
    center = []
    heights = []
    for h in np.arange(0,frame.shape[0]-1,kernel):
        y = np.mean(frame[h:h+kernel,:],axis=0)
        ps = find_peaks(y,200,prominence=50)
        w = y[ps[0]]/np.sum(y[ps[0]])
        u = ps[0]
        center.append(np.sum(w*u))
        heights.append(h+(kernel-1)/2 +.5)
    heights = np.array(heights).reshape(len(heights),1)
    center = np.array(center).reshape(len(center),1)
    return np.concatenate((heights,center),axis=1)

def main():
    """
    
Parallelized function call to find the centerline of a plume for multiple images.
There are 3 implemented methods:
1) moment  ---  uses cv2 python library to calculate moment of image at a certain height, which is used to find the horizontal component of the centroid
    inputs: frame --  numpy array representing the image
            kernel--  int representing vertical interval over which to calculate the centroid
            threshold --  int threshold for masking areas of the image you don't want to include in the centroid weighting
2) outline --- binarizes images using the inputted threshold. cv2 python library takes in the binarized image and creates outlines, the largest of which
                is assumed to be the plume outline. The centroid of the outline for each height according to the kernel size is then computed. 
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
        args.end_frame = img.it
    # start up parallel pool
    avail_cores = int(psutil.cpu_count() - (psutil.cpu_count()/2))
    
    param1 = img
    param2 = args.method[0]
    param3 = args.kernel[0]
    param4 = args.threshold[0]
    param5 = np.arange(args.start_frame,args.end_frame)
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
    results = np.array(results)
    results = np.concatenate((results[0:1,:,0],results[:,:,1]),axis=0)
    np.savez_compressed(os.path.splitext(img.file_name)[0]+'.img_centerline_%s.npz' %(args.method[0]),results)
    print '[FINISHED]: %f seconds elapsed' %(time.time()-tic)
    
if __name__ == "__main__":
    try:
        if sys.argv[1]=='-help' or sys.argv[1]=='-h':
            print(main.__doc__)
            main()
        else:
            main()
    except:
        main()
