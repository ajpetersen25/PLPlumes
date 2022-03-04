#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:29:53 2020

@author: ajp25
"""
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, binary_fill_holes
from skimage.morphology import dilation,disk,square,binary_erosion, binary_dilation,erosion
import scipy.ndimage.measurements as measurements
from PLPlumes.pio import pivio,imgio
from scipy import interpolate
from scipy.ndimage import map_coordinates
from scipy.stats import kurtosis
from scipy.spatial.distance import cdist
import time
import argparse
import multiprocessing as mp
from itertools import repeat
import copy
import getpass
from datetime import datetime
import cv2
#from scipy.ndimage import sobel
import os

import h5py
import tables

#import numba
#from numba import cfunc, carray
#from numba.types import intc, CPointer, float64, intp, voidptr
#from scipy import LowLevelCallable

"""

from numba import njit,prange


@njit
def padding(img,pad):
    padded_img = np.zeros((img.shape[0]+2*pad,img.shape[1]+2*pad))
    padded_img[pad:-pad,pad:-pad] = img
    return padded_img

@njit(parallel=True)
def AdaptiveMedianFilter(img,s=3,sMax=7):
    if len(img.shape) == 3:
        raise Exception ("Single channel image only")

    H,W = img.shape
    a = sMax//2
    padded_img = padding(img,a)

    f_img = np.zeros(padded_img.shape)

    for i in prange(a,H+a+1):
        for j in range(a,W+a+1):
            value = Lvl_A(padded_img,i,j,s,sMax)
            f_img[i,j] = value

    return f_img[a:-a,a:-a] 

@njit
def Lvl_A(mat,x,y,s,sMax):
    window = mat[x-(s//2):x+(s//2)+1,y-(s//2):y+(s//2)+1]
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)

    A1 = Zmed - Zmin
    A2 = Zmed - Zmax

    if A1 > 0 and A2 < 0:
        return Lvl_B(window, Zmin, Zmed, Zmax)
    else:
        s += 2 
        if s <= sMax:
            return Lvl_A(mat,x,y,s,sMax)
        else:
             return Zmed

@njit
def Lvl_B(window, Zmin, Zmed, Zmax):
    h,w = window.shape

    Zxy = window[h//2,w//2]
    B1 = Zxy - Zmin
    B2 = Zxy - Zmax

    if B1 > 0 and B2 < 0 :
        return Zxy
    else:
        return Zmed
"""
def fill_nans(x, y, z, xp, yp, q_at_p):
    """"
    Fill in the NaNs or masked values on interpolated points using nearest
    neighbors.

    .. warning::import getpass


        Operation is performed in place. Replaces the NaN or masked values of
        the original array!

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the original data points (not
        interpolated).
    * v : 1D array
        Array with the scalar value assigned to the data points (not
        interpolated).
    * xp, yp : 1D arrays
        Points where the data values were interpolated.
    * vp : 1D array
        Interpolated data values (the one that has NaNs or masked values to
        replace).

    """
    if np.ma.is_masked(q_at_p):
        nans = q_at_p.mask
    else:
        nans = np.isnan(q_at_p)
    q_at_p[nans] = interpolate.griddata(((x.ravel()), y.ravel()), z, (xp[nans], yp[nans]), method='nearest').ravel()
    return q_at_p

def q_at_p(points, q, dstep):
    """
    Inputs:
    points --- (n,2) array of positions in pixel units
    q      --- 2d array of fluid field values
    dstep  --- int, pixel spacing between piv vectors
    Returns:
    q_at_ps --- n,1 array of interpolated values at points
    """
    X,Y = np.meshgrid(dstep * np.arange(0,q.shape[1]) + dstep,dstep * np.arange(0,q.shape[0]) + dstep)
    xi=X.ravel()/dstep
    yi=Y.ravel()/dstep
    z = q.ravel()
    #xi=xi[~np.isnan(z)]
    #yi=yi[~np.isnan(z)]
    xi=xi.reshape(len(xi),1)
    yi=yi.reshape(len(yi),1)
    #z = z[~np.isnan(z)]
    q_at_pts = interpolate.griddata(np.concatenate((xi,yi),axis=1),z,points/dstep,method='linear')
    #print(q[int(points[0]/dstep),int(points[1]/dstep)])
    if np.any(np.isnan(q_at_pts)):
        q_at_pts = fill_nans(xi,yi,z,points[:,0]/dstep,points[:,1]/dstep,q_at_pts)
    return q_at_pts




def plume_outline(img_frame,dilation_size,threshold,med_filt_size,cutoff=1600):
    #img_frame,dilation_size,threshold,med_filt_size,cutoff,orientation = params
    """
    img_outline = frame>threshold
    contours, hierarchy =   cv2.findContours(img_outline.copy().astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    c_size = ([c.size for c in contours])
    lc = np.where(c_size == np.max(c_size))[0][0]
    plume_contour = cv2.drawContours(image,contours[lc],-1, 255, 1)

    return plume_contour"""
    frame_d = img_frame.astype('float')
    kernel = square(int(dilation_size))
    frame_d = dilation(frame_d,kernel)
    frame_d = median_filter(frame_d,med_filt_size)
    
    #kernel = square(int(dilation_size))
    #frame_d = erosion(frame_d,kernel)
    #kernel = square(int(dilation_size/2))
    #frame_d = dilation(frame_d,kernel)
    #frame_d = gaussian_filter(frame_d,dilation_size)
    #blur = gaussian_filter(frame_d, sigma=gaussian_sigma)
    
    frame_d = gaussian_filter(erosion(frame_d,square(int(10))),24)
    frame_d = gaussian_filter(dilation(frame_d,disk(int(dilation_size*1.5))).astype('float32'),24)
    frame_mask = frame_d > threshold
    #frame_mask = gaussian_filter(dilation(frame_mask,disk(int(dilation_size))).astype('float32'),24)>0.1
    #f1lab, f1num = measurements.label(frame_mask)
    #f1slices = measurements.find_objects(f1lab)
    #obj_sizes = []
    #img_objects = []
    #for s, f1slice in enumerate(f1slices):
    #    # remove overlapping label objects from current slice
    #    img_object = ((f1lab==(s+1)))
    #    obj_size = (np.count_nonzero(img_object))
    #    obj_sizes.append(obj_size)
    #    img_objects.append(img_object)
    #img_objects = np.array(img_objects)[np.where(np.array(obj_sizes)>50000)[0]]
    #frame_mask = np.zeros(img_frame.shape)
    #for i in range(0,img_objects.shape[0]):
    #    frame_mask += img_objects[i]
    #frame_mask = binary_fill_holes(frame_mask)
    #contours, hierarchy = cv2.findContours(frame_mask.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #c = np.squeeze(contours)
    #frame_mask[:,np.min(c[:,0])] = 1
    #frame_mask[:,np.max(c[:,0])] = 1

    set1 = np.where(frame_mask[:,0]==0)[0]
    set2 = np.where(frame_mask[:,2559]==0)[0]
    set3 = np.where(frame_mask[0,:]==0)[0]
    frame_mask[set1,0] = 1
    frame_mask[set2,2559] = 1
    frame_mask[0,set3] = 1
    frame_mask = binary_fill_holes(frame_mask)
    frame_mask[set1,0] = 0
    frame_mask[set2,2559] = 0
    frame_mask[0,set3] = 0 
    
    contours, hierarchy = cv2.findContours(frame_mask.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #c = np.squeeze(contours)
    # Initialize empty list
    lst_intensities = []

    # For each list of contour points...
    for i in range(len(contours)):
    # Create a mask image that contains the contour filled in
        cimg = np.zeros_like(img_frame)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        lst_intensities.append(img_frame[pts[0], pts[1]])
    bad = []
    for c in range(0,len(contours)):
        if np.sum(lst_intensities[c]>700)/len(lst_intensities[c]) < 0.07 or len(contours[c])<100: #abs(kurtosis(lst_intensities[c])) > 5 or 
            bad.append(c)

            
    for b in sorted(bad,reverse=True):
        del contours[b]
    c2 = []
    for c in contours:
        try:
            c = np.squeeze(c)
            idx = []
            #if orientation == 'horz':
            idx.extend(np.where(c[:,0]==0)[0])
            idx.extend(np.where(c[:,0]==1)[0])
            idx.extend(np.where(c[:,1]==0)[0])
            idx.extend(np.where(c[:,1]==1)[0])
            idx.extend(np.where(c[:,0]==frame_mask.shape[1]-1)[0])
            idx.extend(np.where(c[:,0]==frame_mask.shape[1]-2)[0])
        except:
    	    pass
        c2.append(np.delete(c,idx,axis=0))


    c3 = []
    for c in c2:
        c3.extend(c)
    plume_outline_pts = np.array(c3)
   
    
    """plume_outline_pts = []
    if orientation == 'horz':
        for col in range(0,frame_mask.shape[1]):
            rows = np.where(np.diff(frame_mask[:,col])==1)[0]
            rows = rows[rows<cutoff]
            for row in rows:
                plume_outline_pts.append(np.array([row,col]))
    elif orientation == 'vert':
        for row in range(0,frame_mask.shape[0]):
            cols = np.where(np.diff(frame_mask[row,:])==1)[0]
            cols = cols[cols>cutoff]
            for col in cols:
                plume_outline_pts.append(np.array([row,col]))"""
        
    return plume_outline_pts, frame_mask, frame_d


"""def plume_outline2(img_frame,blur_strength=32,orientation='horz',piv_window_size=32,cutoff=1000):
    #img_frame = img_frame.astype('float')
    kernel = square(dilation_size)
    frame_d = dilation(img_frame,kernel)
    frame_d = gaussian_filter(frame_d, sigma=blur_strength)
    frame_d = median_filter(frame_d,med_filt_size)
    frame_mask = frame_d > threshold
    f1lab, f1num = measurements.label(frame_mask)
    f1slices = measurements.find_objects(f1lab)
    obj_sizes = []
    img_objects = []
    for s, f1slice in enumerate(f1slices):
        # remove overlapping label objects from current slice
        img_object = ((f1lab==(s+1)))
        obj_size = (np.count_nonzero(img_object))
        obj_sizes.append(obj_size)
        img_objects.append(img_object)
    frame_mask = img_objects[np.where(obj_sizes==np.max(obj_sizes))[0][0]]
    del img_objects, img_object, f1lab,f1slices,f1slice
    kernel = square(dilation_size*5)
    sobel_map = sobel(frame_d.astype('float'))*dilation(frame_mask,kernel)
   
    #eroded_map = cv2.erode(sobel_map,kernel,iterations=2)
    blurred_sobel2 = gaussian_filter(sobel_map, sigma=blur_strength)
    plume_outline_pts2 = []
    missing = []
    too_many_peaks = []
    if orientation == 'horz':
        for col in range(0,frame_d.shape[1]):
            profile = np.gradient(windowed_average(frame_d[0:cutoff,col],10))
            #all_peaks = find_peaks(profile,height=10,prominence=1)
            #try:
            #    if len(all_peaks[0]>3):
            #        idx = all_peaks[0][np.argpartition(-all_peaks[0],3)[:3]]
            #        peaks = find_peaks(profile,height=np.min(profile[idx]),prominence=1)
            #    else:
            peaks = find_peaks(abs(profile),height=np.max(abs(profile)),prominence=1)
            plume_outline_pts2.append(np.array([peaks[0][0]+piv_window_size,col]))
            if len(peaks[0])>1:
                too_many_peaks.append(col)

            if len(peaks[0])==0:
                missing.append(col)
            elif len(peaks[0])==1:
                row = peaks[0][0]
                plume_outline_pts2.append(np.array([row+piv_window_size,col]))
            else:
                if len(peaks[0]) % 2 == 0:
                    row = peaks[0][0]
                    plume_outline_pts2.append(np.array([row+piv_window_size,col]))
                else:
                    for i in range(0,len(peaks[0])):
                        row = peaks[0][i]
                        if (i+1) % 2 ==0:
                            plume_outline_pts2.append(np.array([row-piv_window_size,col]))
                        else:
                            plume_outline_pts2.append(np.array([row+piv_window_size,col]))
            #except:
            #    pass
            

    return(plume_outline_pts)
 """   
def euclideanDistance(coordinate1, array):
    return pow(pow(coordinate1[0] - array[:,0], 2) + pow(coordinate1[1] - array[:,1], 2), .5)

def image_gradient(image, sigma):
    image = np.asfarray(image)
    gx = gaussian_filter(image, sigma, order=[0, 1])
    gy = gaussian_filter(image, sigma, order=[1, 0])
    return gx, gy

def subpixel_pts(plume_pts,frame_d,nx,ny,threshold,interp_kernel=5):
    pts = np.empty(plume_pts.shape)
    for i in range(0,len(pts)):
        pts[i] = plume_pts[i]
        if abs(nx[i]) >= abs(ny[i]):
            x0 = int(plume_pts[i][0])
            edgemin = 0#np.min(pts[:,0])
            edgemax = 2559#np.max(pts[:,0])
            if plume_pts[i,0]-interp_kernel < edgemin:
                interp_kernel_min = plume_pts[i,0] - edgemin
                x_interp = np.arange(x0-interp_kernel_min,x0+interp_kernel)
            elif plume_pts[i,0]+interp_kernel>edgemax:
                interp_kernel_max = edgemax - pts[i,0]
                x_interp = np.arange(x0-interp_kernel,x0+interp_kernel_max)
            else:
                x_interp = np.arange(x0-interp_kernel,x0+interp_kernel)
            Is = [frame_d[int(plume_pts[i,1]),int(xi)] for xi in x_interp]
            I_interp = interpolate.interp1d(x_interp,Is)
            x = np.arange(x_interp.min(),x_interp.max(),.0001)
            idx = (np.abs(I_interp(x)-threshold)).argmin()
            pts[i,0] = x[idx]
        else:
            y0 = int(plume_pts[i][1])
            edgemin = 0#np.min(pts[:,0])
            edgemax = 1599#np.max(pts[:,0])
            if plume_pts[i,1]-interp_kernel < edgemin:
                interp_kernel_min = plume_pts[i,1] - edgemin
                y_interp = np.arange(y0-interp_kernel_min,y0+interp_kernel)
            elif plume_pts[i,1]+interp_kernel>edgemax:
                interp_kernel_max = edgemax - plume_pts[i,1]
                y_interp = np.arange(y0-interp_kernel,y0+interp_kernel_max)
            else:
                y_interp = np.arange(y0-interp_kernel,y0+interp_kernel)
            Is = [frame_d[int(yi),int(plume_pts[i,0])] for yi in y_interp]
            I_interp = interpolate.interp1d(y_interp,Is)
            y = np.arange(y_interp.min(),y_interp.max(),.0001)
            idx = (np.abs(I_interp(y)-threshold)).argmin()
            pts[i,1] = y[idx]
            
    return(pts)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y          

def on_segment(p1, p2, p):
    return min(p1.x, p2.x) <= p.x <= max(p1.x, p2.x) and min(p1.y, p2.y) <= p.y <= max(p1.y, p2.y)

def direction(p1, p2, p3):
	return  cross_product(p3.subtract(p1), p2.subtract(p1))
    
def cross_product(p1, p2):
	return p1.x * p2.y - p2.x * p1.y

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def subtract(self, p):
    	return Point(self.x - p.x, self.y - p.y)

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

def intersect(p1, p2, p3, p4):
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    elif d1 == 0 and on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and on_segment(p1, p2, p4):
        return True
    else:
        return False
          
def boundary_normal_disp(pts1,pts2,nx_norm,ny_norm,max_move=20,interp_kernel=5):
    #tic = time.time()
    ub = []
    for i in range(0,len(pts1)):
        lim1 = pts1[i] - np.array([nx_norm[i]*max_move,ny_norm[i]*max_move])
        lim2 = pts1[i] + np.array([nx_norm[i]*max_move,ny_norm[i]*max_move])
        n_range = np.vstack((lim1,lim2))
        
        nn = euclideanDistance(pts1[i],pts2).argmin()
        if nn-interp_kernel < 0:
            y_interp = np.arange(0,nn+interp_kernel)
        elif nn + interp_kernel > len(pts2):
            interp_kernel_max = len(pts2) - nn - 1
            y_interp = np.arange(nn-interp_kernel,nn+interp_kernel_max)
        else:
            y_interp = np.arange(nn-interp_kernel,nn+interp_kernel)
            
        p1 = Point(n_range[0,0],n_range[0,1])
        p2 = Point(n_range[1,0],n_range[1,1])
        q1 = Point(pts2[y_interp[0]][0],pts2[y_interp[0]][1])
        q2 = Point(pts2[y_interp[-1]][0],pts2[y_interp[-1]][1])
        if intersect(p1,p2,q1,q2):
            try:
                int1 = line_intersection((np.array([p1.x,p1.y]),np.array([p2.x,p2.y])),(np.array([q1.x,q1.y]),np.array([q2.x,q2.y])))
                nn2 = euclideanDistance(int1,pts2).argmin()
                upordown = np.array([pts2[nn2-1],pts2[nn2+1]])
                intersection = line_intersection((np.array([p1.x,p1.y]),np.array([p2.x,p2.y])),(pts2[nn2],upordown[euclideanDistance(pts2[nn2] + (int1 - pts2[nn2]), upordown).argmin()]))
                ub.append(intersection - pts1[i])
            except:
                ub.append(np.array([np.nan,np.nan]))
        else:
            ub.append(np.array([np.nan,np.nan]))

        """if abs(nx_norm[i]) >= abs(ny_norm[i]):
            n_seg_x = np.arange(n_range[:,0].min(),n_range[:,0].max(),.01)
            n_interp = interpolate.interp1d(n_range[:,0],n_range[:,1])
            b_seg_x = np.arange(pts2[y_interp][:,1].min(),pts2[y_interp][:,1].max(),.01)
            b_interp = interpolate.interp1d(pts2[y_interp][:,1],pts2[y_interp][:,0])
            l1 = np.concatenate((n_seg_x.reshape(len(n_seg_x),1),n_interp(n_seg_x).reshape(len(n_seg_x),1)),axis=1)  
            l2 = np.flip(np.concatenate((b_seg_x.reshape(len(b_seg_x),1),b_interp(b_seg_x).reshape(len(b_seg_x),1)),axis=1))          
            dist = cdist(l1,l2)
            idxs = np.where(dist == dist.min())
            idx_l1 = idxs[0]
            idx_l2 = idxs[1]
            if idx_l1 == len(l1)-1 or idx_l2 == len(l2)-1:
                ub.append(np.array([np.nan,np.nan]))
                
            elif idx_l1 == len(l1)-2 or idx_l2 == len(l2)-2:
                p1 = Point(l1[idx_l1-2][0,1],l1[idx_l1-2][0,0])
                p2 = Point(l1[idx_l1+1][0,1],l1[idx_l1+1][0,0])
                
                q1 = Point(l2[idx_l2-2][0,1],l2[idx_l2-2][0,0])
                q2 = Point(l2[idx_l2+1][0,1],l2[idx_l2+1][0,0])
        
                if intersect(p1,p2,q1,q2):
                    intersection = line_intersection((l1[idx_l1-2][0],l1[idx_l1+1][0]),(l2[idx_l2-2][0],l2[idx_l2+1][0]))
                    ub.append(intersection - pts1[i])
                else:
                    ub.append(np.array([np.nan,np.nan]))
            else:
                p1 = Point(l1[idx_l1-2][0,1],l1[idx_l1-2][0,0])
                p2 = Point(l1[idx_l1+2][0,1],l1[idx_l1+2][0,0])
                
                q1 = Point(l2[idx_l2-2][0,1],l2[idx_l2-2][0,0])
                q2 = Point(l2[idx_l2+2][0,1],l2[idx_l2+2][0,0])
        
                if intersect(p1,p2,q1,q2):
                    intersection = line_intersection((l1[idx_l1-2][0],l1[idx_l1+2][0]),(l2[idx_l2-2][0],l2[idx_l2+2][0]))
                    ub.append(intersection - pts1[i])
                else:
                    ub.append(np.array([np.nan,np.nan]))
        else:
            n_seg_x = np.arange(n_range[:,1].min(),n_range[:,1].max(),.01)
            n_interp = interpolate.interp1d(n_range[:,1],n_range[:,0])
            b_seg_x = np.arange(pts2[y_interp][:,0].min(),pts2[y_interp][:,0].max(),.01)
            b_interp = interpolate.interp1d(pts2[y_interp][:,0],pts2[y_interp][:,1])    
            l1 = np.concatenate((n_seg_x.reshape(len(n_seg_x),1),n_interp(n_seg_x).reshape(len(n_seg_x),1)),axis=1)  
            l2 = np.flip(np.concatenate((b_seg_x.reshape(len(b_seg_x),1),b_interp(b_seg_x).reshape(len(b_seg_x),1)),axis=1))          
            dist = cdist(l1,l2)
            idxs = np.where(dist == dist.min())
            idx_l1 = idxs[0]
            idx_l2 = idxs[1]
            if idx_l1 == len(l1)-1 or idx_l2 == len(l2)-1:
                ub.append(np.array([np.nan,np.nan]))
            elif idx_l1 == len(l1)-2 or idx_l2 == len(l2)-2:
                p1 = Point(l1[idx_l1-2][0,1],l1[idx_l1-2][0,0])
                p2 = Point(l1[idx_l1+1][0,1],l1[idx_l1+1][0,0])
                
                q1 = Point(l2[idx_l2-2][0,1],l2[idx_l2-2][0,0])
                q2 = Point(l2[idx_l2+1][0,1],l2[idx_l2+1][0,0])
                if intersect(p1,p2,q1,q2):
                    intersection = line_intersection((l1[idx_l1-2][0],l1[idx_l1+1][0]),(l2[idx_l2-2][0],l2[idx_l2+1][0]))
                    ub.append(intersection - pts1[i])
                else:
                    ub.append(np.array([np.nan,np.nan]))
            else:
                p1 = Point(l1[idx_l1-2][0,1],l1[idx_l1-2][0,0])
                p2 = Point(l1[idx_l1+2][0,1],l1[idx_l1+2][0,0])
                
                q1 = Point(l2[idx_l2-2][0,1],l2[idx_l2-2][0,0])
                q2 = Point(l2[idx_l2+2][0,1],l2[idx_l2+2][0,0])
        
                if intersect(p1,p2,q1,q2):
                    intersection = line_intersection((l1[idx_l1-2][0],l1[idx_l1+2][0]),(l2[idx_l2-2][0],l2[idx_l2+2][0]))
                    ub.append(intersection[::-1] - pts1[i])
                else:
                    ub.append(np.array([np.nan,np.nan]))"""
    #print('%0.2f seconds elapsed' %(time.time()-tic))
    return(ub)
        
def entrainment_vel(params):

    img,piv,dilation_kernel,threshold,med_filt_size,cutoff,masking,frame = params

    if masking is True:
        _,frame_mask,_ = plume_outline(img.read_frame2d(frame),dilation_kernel,
                                                     threshold,med_filt_size,cutoff)
        return(frame_mask)
    else:
        #all_outlines = []
        #all_masks = []
        #for frame in range(11904,12027):
        try:
            plume_outline_pts,frame_mask,frame_d = plume_outline(img.read_frame2d(frame),dilation_kernel,
                                                         threshold,med_filt_size,cutoff)
            plume_outline_pts_2,_,frame_d2 = plume_outline(img.read_frame2d(frame+1),dilation_kernel,
                                                     threshold,med_filt_size,cutoff)
        
            #if len(plume_outline_pts) > 0 and len(plume_outline_pts_2) > 0:
            grads = image_gradient(frame_d,5.0)
            nx = grads[0][plume_outline_pts[:,1],plume_outline_pts[:,0]]
            ny = grads[1][plume_outline_pts[:,1],plume_outline_pts[:,0]]   
            pts1 = subpixel_pts(plume_outline_pts,frame_d,nx,ny,threshold)    
    
            #all_outlines.append(plume_outline_pts)
            #all_masks.append(frame_mask)
    
    
            grads2 = image_gradient(frame_d2,5.0)
            nx2 = grads2[0][plume_outline_pts_2[:,1],plume_outline_pts_2[:,0]]
            ny2 = grads2[1][plume_outline_pts_2[:,1],plume_outline_pts_2[:,0]]
            pts2 = subpixel_pts(plume_outline_pts_2,frame_d2,nx2,ny2,threshold)
            
            
            #grads = image_gradient(frame_mask,5.0)
            #nx = grads[0][plume_outline_pts[:,1],plume_outline_pts[:,0]]
            #ny = grads[1][plume_outline_pts[:,1],plume_outline_pts[:,0]]
            n_vecs = np.concatenate((nx.reshape(len(nx),1),ny.reshape(len(ny),1)),axis=1) 
            n_vecs = n_vecs[:,:]/(np.linalg.norm(n_vecs,axis=1)).reshape(len(nx),1)
            bound_vecs = np.array(boundary_normal_disp(pts1,pts2,n_vecs[:,0],n_vecs[:,1],max_move=20,interp_kernel=5))

            piv_root,piv_ext = os.path.splitext(piv.file_name)
            ### masking###
            #img = imgio.imgio(img)
            #piv = pivio.pivio(piv)
            #x = np.arange(piv.dx,img.ix-piv.dx,piv.dx)
            #y = np.arange(piv.dy,img.iy-piv.dy,piv.dy)
            #slices = []
            #for i in x:
            #   for j in y:
            #       slices.append((slice(int(j),int(j+piv.dy)),slice(int(i),int(i+piv.dx))))
                   
            #mask = np.zeros(piv.read_frame2d(frame)[0].shape)
        
            #for s in slices:
            #    mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] = np.mean(frame_mask[s])
        
            #mask[mask>=0.1]=1
            #mask[mask<0.1]=0
            #mask[-1,:]=1
            #np.savetxt(piv_root[:-5]+'_frame_c_%06d_mask.txt' %frame,mask)
            #return(np.flipud(mask))
            ######
            #pts = np.array([np.array([r[0],r[1]]) for r in pts1])    
            
            #X,Y = np.meshgrid(piv.dx * np.arange(0,piv.read_frame2d(frame)[0].shape[1]) + piv.dx,
            #                  piv.dx * np.arange(0,piv.read_frame2d(frame)[0].shape[0]) + piv.dx)
            
            #points = np.vstack([Y.ravel(),X.ravel()])
            #vf_mask = np.fliplr(piv.read_frame2d(frame)[0]) #(map_coordinates(frame_mask,points).reshape(piv.read_frame2d(frame)[0].shape))
            #u = -np.fliplr(np.flipud((piv.read_frame2d(frame)[1])))
            #v = -np.fliplr(np.flipud((piv.read_frame2d(frame)[2])))
            vf_mask = (piv.read_frame2d(frame-1)[0]) #(map_coordinates(frame_mask,points).reshape(piv.read_frame2d(frame)[0].shape))
            u = piv.read_frame2d(frame-1)[1]
            v = piv.read_frame2d(frame-1)[2]

            
            #piv_mask = piv.read_frame2d(frame)[0].astype('bool')
            #u[piv_mask!=T]=np.nan
            #v[piv.read_frame2d(frame)[0]!=1]=np.nan
            u[vf_mask!=1] = np.nan
            v[vf_mask!=1] = np.nan
            #u[np.abs(u)>3] = np.nan
            #v[np.abs(v)>3] = np.nan
            u_at_p = q_at_p(pts1,u,piv.dx)
            v_at_p = q_at_p(pts1,v,piv.dx)
            upiv = np.concatenate((u_at_p.reshape(len(u_at_p),1),v_at_p.reshape(len(v_at_p),1)),axis=1)
            u_e_vec = (upiv/3 - bound_vecs)
            """proj = (np.sum(u_e_vec*n_vecs,axis=1))/(np.linalg.norm(n_vecs,axis=1))
            angles = (np.arccos((np.sum(u_e_vec*n_vecs,axis=1))/(np.linalg.norm(u_e_vec,axis=1)*np.linalg.norm(n_vecs,axis=1)))*180/np.pi)
            sign = np.ones(angles.shape)
            with np.errstate(invalid='ignore'):
                sign[np.abs(angles)<90] = -1
            u_at_boundary = proj*sign
            u_at_boundary[np.abs(u_at_boundary)>5] = np.nan"""
            """
            d = []
            vec = []
            #for f in range(len(all_outlines)-1):
            #    plume_outline_pts = all_outlines[f]
            #    plume_outline_pts_2 = all_outlines[f+1]
            for i in range(len(plume_outline_pts)):
                #distances = []
                #for j in range(len(plume_outline_pts2)):
                distances = euclideanDistance(np.array(plume_outline_pts)[i],np.array(plume_outline_pts_2))
                min_loc = np.where(distances==np.min(distances))[0][0]
                vec.append(plume_outline_pts_2[min_loc] - plume_outline_pts[i])
                d.append(np.min(distances))
             
            if orientation=='horz':
                for c in range(0,img.ix):
                    cols = np.where(pts[:,0]==c)[0]
                    for c2 in cols:
                        vel_vec = [u_at_p[c2]/3-vec[c2][0],v_at_p[c2]/3-vec[c2][1]]
                        #vel_vec = [u_at_p[c2],v_at_p[c2]]
                        try:
                            if np.isnan(vel_vec).any():
                                u_e.append(np.nan)
                            else:
                                fit = np.polyfit(pts[c2-7:c2+8,0],pts[c2-7:c2+8,1],1)
                                line= np.array([1,fit[0]])
                                l_norm = np.sqrt(np.sum(np.array(line)**2))
                                proj = (np.dot(vel_vec,line)/l_norm**2)*np.array(line)
                                perp_vel = np.linalg.norm(vel_vec - proj)*np.sign(vel_vec-proj)[1]
                                u_e.append(perp_vel)
                        except:
                            u_e.append(np.nan)
            elif orientation=='vert':
                for r in range(0,img.iy):
                    rows = np.where(pts[:,0]==r)[0]
                    for r2 in rows:
                        vel_vec = [v_at_p[r2]/3,u_at_p[r2]/3]
                        try:
                            if np.isnan(vel_vec).any():
                                u_e.append(np.nan)
                            else:
                                fit = np.polyfit(pts[r2-2:r2+3,0],pts[r2-2:r2+3,1],1)
                                if fit[0] < 1e-6:
                                    fit[0]=0
                                line = np.array([1,fit[0]])
                                l_norm = np.sqrt(np.sum(np.array(line)**2))
                                proj = (np.dot(vel_vec,line)/l_norm**2)*np.array(line)
                                perp_vel = np.linalg.norm(vel_vec - proj)*np.sign(vel_vec)[1]
                                u_e.append(perp_vel)
                        except:
                            u_e.append(np.nan)
        
            try:
                u_at_boundary = np.array(u_e).reshape(len(u_e),1)
            except:
                print('frame %d ERROR' %frame)"""
    
            return(frame_mask,pts1,np.concatenate((u_at_p.reshape(len(u_at_p),1),v_at_p.reshape(len(v_at_p),1)),axis=1),bound_vecs,n_vecs)
        except Exception as e:
        #else:
            piv_root,piv_ext = os.path.splitext(piv.file_name)
            error_file = piv_root+'.error.txt'
            if os.path.exists(error_file):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
            
            err = open(error_file,append_write)
            err.write(str(e) + '  \n')
            err.close()
            
            piv_root,piv_ext = os.path.splitext(piv.file_name)
            missing_file = piv_root+'.missing_frames.txt'
            if os.path.exists(missing_file):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
            
            missing = open(missing_file,append_write)
            missing.write('frame_%06d' %(frame) + '\n')
            missing.close()
            return(frame_mask,[],([],[]),[],[])

def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for parallel plume piv',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('img_file',type=str, help='path to .img file')
    parser.add_argument('piv_file',type=str, help='Name of .piv file')
    parser.add_argument('dilation_kernel',type=int,help='size of dilation filter')
    parser.add_argument('threshold',type=int,help='intensity threshold for binarizing image')
    parser.add_argument('med_filt_size',type=int,help='size of median filter')
    parser.add_argument('cutoff',type=int,help='cutoff row/column')
    parser.add_argument('masking',type=str,nargs='?', default=False,help='masking or full entrainment calc')
    parser.add_argument('start_frame',nargs='?',default=0,type=int, help='Frame to start separation from')
    parser.add_argument('end_frame',nargs='?', default=0,type=int, help='Number of frames to separate')
    parser.add_argument('cores',type=int,nargs='?',default=1,help='Optional - Force number of cores for node_separate')
    args = parser.parse_args() 
    img = imgio.imgio(args.img_file)
    piv = pivio.pivio(args.piv_file)
    if args.end_frame == 0:
        end_frame = img.it
    else:
        end_frame = args.end_frame
    frames = np.arange(args.start_frame,end_frame)
    f_tot = len(frames)
    if args.masking == 'False' or args.masking == 'false':
        masking = False
    elif args.masking == 'True' or args.masking == 'true':
        masking = True
    else:
        print('%s is not a valid option, please enter True or False' %args.masking)
        exit()
    
    ### masking
    
    objList = list(zip(repeat(img,times=f_tot),
                   repeat(piv,times=f_tot),
                   repeat(args.dilation_kernel,times=f_tot),
                   repeat(args.threshold,times=f_tot),
                   repeat(args.med_filt_size,times=f_tot),
                   repeat(args.cutoff,times=f_tot),
                   repeat(masking,times=f_tot),
                   frames))
    
    """objList = list(zip([img.read_frame2d(f) for f in frames],
               repeat(args.dilation_kernel,times=f_tot),
               repeat(args.threshold,times=f_tot),
               repeat(args.med_filt_size,times=f_tot),
               repeat(args.cutoff,times=f_tot),
               repeat(args.orientation,times=f_tot)))"""

    pool = mp.Pool(processes=args.cores)
    results = pool.map(entrainment_vel,objList)

    if masking is True:
        masks = np.array(results)
        img_root = os.path.splitext(img.file_name)[0]
        img2 = imgio.imgio(os.path.splitext(img.file_name)[0]+'.mask.img')
        
        #img2.file_name = "%s" % (img_root+'.mask.img')
        #img2 = imgio.imgio(os.path.splitext(img.file_name)[0]+'.mask.img')
        img2.ix = img.ix
        img2.iy = img.iy
        img2.it = f_tot
        img2.unsigned = 1
        img2.bytes = 2
        img2.channels = 1
        img2.min = 0
        img2.max = 65535
        img2.type = 'uint16'
        d = datetime.now()
        img2.comment = "%s\n%s %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__), args.img_file, d.strftime("%a %b %d %H:%M:%S %Y")) + str(img.comment,'utf-8')
    
        img2.write_header()
        for f in range(0,f_tot):
            img2.write_frame(np.flipud(masks[f,:,:]))
    else:
    #print(results.shape)
        results = np.array(results)
        #masks = results[:,0]
        pts = results[:,1]
        uf_at_p = results[:,2]
        u_p = results[:,3]
        u_e = results[:,4]
        piv_root,piv_ext = os.path.splitext(args.piv_file)
        h5 = h5py.File(piv_root+'.u_e.hdf5','a')
        frame_group = h5.create_group('frames')
        for ff,f in enumerate(frames):
            #h5['frames'].create_dataset('frame_%06d/frame_masks' %f,data=masks[ff])
            h5['frames'].create_dataset('frame_%06d/boundary_pts' %f,data=pts[ff])
            h5['frames'].create_dataset('frame_%06d/uf_at_p' %f,data=uf_at_p[ff])
            h5['frames'].create_dataset('frame_%06d/u_p' %f,data=u_p[ff])
            h5['frames'].create_dataset('frame_%06d/n' %f,data=u_e[ff])
            
        h5.close()
        tables.file._open_files.close_all()
    #np.save(piv_root+'.interp_vel.npy',results[1:3])

    #results = ~np.array(results).astype('bool')
    ###
    
    
    """
    results = []
    for f in frames:
        params = img,piv,args.dilation_kernel,args.med_filt_size,args.threshold,args.cutoff,args.orientation,f
        results.append(entrainment_vel(params))
    results = np.array(results)
    #np.save(piv_root+'.u_e.npy', results)
    for ff,f in enumerate(frames):
        h5['frames'].create_dataset('frame_%06d' %f,data=results[ff])
        
    h5.close()
    tables.file._open_files.close_all()"""
    ###masking
    #piv_root,piv_ext = os.path.splitext(args.piv_file)
    #piv2 = copy.deepcopy(piv)
    #piv2.file_name = piv_root+'.msk.piv'
    #d = datetime.now()
    #piv2.comment = "%s\n%s %d %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(),os.path.basename(__file__), 1, args.piv_file, 
    #                                                 d.strftime("%a %b %d %H:%M:%S %Y")) + str(piv.comment,'utf-8')
    #piv2.nt = f_tot
    #piv2.write_header()
    #for f in range(0,f_tot):
    #    data = [results[f].flatten(),piv.read_frame(f)[1],piv.read_frame(f)[2]]
    #    piv2.write_frame(data)
    ###
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()    
    

#%%

'''img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.img')

piv = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.t.bsub.0032.def.piv')

f=11000
u_at_boundary = np.zeros((12000,2560))
for f in range(0,12000):
    plume_outline_pts,frame_mask = plume_outline(img.read_frame2d(f),40,2,2600,100,'horz')
      
    pts = np.array([np.array([r[0],r[1]]) for r in plume_outline_pts])    
    
    X,Y = np.meshgrid(piv.dx * np.arange(0,piv.read_frame2d(f)[0].shape[1]) + piv.dx,
                      piv.dx * np.arange(0,piv.read_frame2d(f)[0].shape[0]) + piv.dx)
    
    points = np.vstack([Y.ravel(),X.ravel()])
    vf_mask = binary_erosion(map_coordinates(frame_mask,points).reshape(piv.read_frame2d(f)[0].shape))
    u = (piv.read_frame2d(f)[1])
    v = (piv.read_frame2d(f)[2])
    
    u[piv.read_frame2d(f)[0]!=1]=np.nan
    v[piv.read_frame2d(f)[0]!=1]=np.nan
    u[vf_mask!=0] = np.nan
    v[vf_mask!=0] = np.nan
    
    plt.figure(31);plt.pcolormesh(img.read_frame2d(f),cmap='gray');  
    plt.figure(31);plt.quiver(X,Y,u,v,color='b')
    plt.figure(31);plt.plot(pts[:,1],pts[:,0],'r.')       
    
    u_at_p = q_at_p(pts,u.transpose(),piv.dx)
    v_at_p = q_at_p(pts,v.transpose(),piv.dx)
    for c in range(2,2560-3):
        vel_vec = [u_at_p[c],v_at_p[c]]
        if np.isnan(vel_vec).any():
            u_at_boundary[c] = np.nan
        else:
            fit = np.polyfit(pts[c-2:c+3,1],pts[c-2:c+3,0],1)
            line= np.array([1,fit[0]])
            l_norm = np.sqrt(np.sum(np.array(line)**2))
            proj = (np.dot(vel_vec,line)/l_norm**2)*np.array(line)
            perp_vel = np.linalg.norm(vel_vec - proj)
            u_at_boundary[c] = perp_vel
    u_at_boundary[u_at_boundary==0]=np.nan
    plt.figure(31);plt.quiver(pts[:,1],pts[:,0],u_at_p[:],v_at_p[:],color='r',scale=100)
    '''
