#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:16:38 2020

@author: ajp25
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.io
import scipy.ndimage.measurements as measurements
import os
import glob
import copy
import argparse
import time
from PIL import Image



#directory = os.fsencode(directory_in_str)

#for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".asm") 
    
    
def load_image2(img_file):
    img_bgr = cv2.imread(img_file)
    img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)     
    return img

def load_image(img_file):

    img = skimage.io.imread(img_file,plugin='tifffile')
    return img
    
def crop(img):
    fromCenter = False
    # select crop area
    r = cv2.selectROI('Image',img, fromCenter)
    # Crop image
    imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    # Display cropped image
    # cv2.imshow("Image", imCrop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return imCrop,r
    
def find_points(img,threshold,min_size=9):
    img[img<threshold]=0
    flab,fnum = measurements.label(img)
    fslices = measurements.find_objects(flab)
    cal_points = []
    for s, fslice in enumerate(fslices):
        imCopy = np.zeros(img.shape)
        img_object = img[fslice]*(flab[fslice]==(s+1))
        obj_size = np.count_nonzero(img_object)
        if obj_size > min_size:
            imCopy[fslice] = img_object
            centroid = measurements.center_of_mass(imCopy)
            cal_points.append(np.array([s,fslice,centroid[1]+0.5,centroid[0]+0.5]))
    cal_points = np.asarray(cal_points)
    return cal_points
    
def pick_shared_pts(imCrop,cal_points,pts):
    plt.figure()
    plt.pcolormesh((imCrop),cmap='gray')
    plt.plot(cal_points[:,2],cal_points[:,3],'rx')
    print('Choose shared points between images')
    x=[]
    for i in range(0,pts):
        x0 = plt.ginput(1)[0]
        dist = (cal_points[:,3]-x0[1])**2+(cal_points[:,2]-x0[0])**2
        (mindist, i0) = (dist.min(),np.where(dist==dist.min())[0])
        plt.plot(cal_points[i0,2],cal_points[i0,3],'bo')
        x.append(np.array([cal_points[i0,2][0],cal_points[i0,3][0]]))
        i+=1
    x = np.asarray(x).astype('float32')
    return x

def translation_matrix(x1,x2,pts,img_shape_in_stack_dir):
    T = np.zeros((3,3))

    T[0,0]=1;T[1,1]=1;T[2,2]=1
    T[0,2] = np.mean([x1[i][0]+img_shape_in_stack_dir - x2[i][0] for i in range(0,pts)])
    T[1,2] = np.mean([x1[i][1]- x2[i][1] for i in range(0,pts)])
    return T

def create_alpha(pano_h,pano_w,img_shape_in_stack_dir,T,border):
    lr = img_shape_in_stack_dir-T[0,2]
    ll = img_shape_in_stack_dir

    alpha1 = np.zeros((pano_h,pano_w))
    for c in range(int(lr+1),ll):
        alpha1[:,c] = np.abs(c-ll)/(np.abs(c-lr)+np.abs(c-ll))

#    if T[1,2]<0:
#        alpha1[0:int(border/2),:] = 1
#        alpha1[int(np.ceil(border/2-T[1,2]+pano_h)):,:] = 0
#    elif T[1,2]>0:
#        alpha1[0:int(border/2),:] = 0
#        alpha1[int(np.ceil(border/2-T[1,2]+pano_h)):,:] = 1
    alpha1[:,0:int(lr+1)]=1

    alpha2 = np.ones((pano_h,pano_w))
    for c in range(int(lr+1),ll):
        alpha2[:,c] = np.abs(c-lr)/(np.abs(c-lr)+np.abs(c-ll))
    alpha2[:,0:int(lr+1):]=0
#    if T[1,2]<0:
#        alpha2[0:int(border/2),:] = 1
#        alpha2[int(np.ceil(border/2-T[1,2]+pano_h)):,:] = 0
#    elif T[1,2]>0:
#        alpha2[0:int(border/2),:] = 0
#        alpha2[int(np.ceil(border/2-T[1,2]+pano_h)):,:] = 1
        
    return alpha1,alpha2
    
def create_pano(img1,img2,alpha1,alpha2,T,border=200):
    border = int(border)
    height_panorama = img1.shape[0]+border
    width_panorama = img1.shape[1]*2
    panorama1 = np.zeros((height_panorama,width_panorama))
    panorama1[int(border/2)+int(T[1,2]):height_panorama-int(border/2)+int(T[1,2]), img1.shape[1]-int(T[0,2]):-int(T[0,2])] = img1
    #ii2 = np.zeros((img2.shape[0]+border,img2.shape[1]))
    #ii2[int(border/2):ii2.shape[0]-int(border/2),:] = img2
    #panorama2 = cv2.warpPerspective(ii2, T, (width_panorama, height_panorama))
    panorama2 = np.zeros((height_panorama,width_panorama))
    panorama2[int(border/2):height_panorama-int(border/2),0:img1.shape[1]] = img2
    return np.floor(alpha2*panorama2+(1-alpha1)*panorama1)

def check_threshold(im):
    good = 0
    while good == 0:
        img = copy.deepcopy(im)
        threshold = input('Enter intensity threshold for calibration points: \n')
        validt = 0
        while validt == 0:
            try:
                t = int(threshold)
                validt = 1
            except:
                threshold = input('Enter VALID intensity threshold for calibration points: \n')
                validt = 0
        img[img<t] = 0
        plt.figure();
        plt.pcolormesh(img,cmap='gray')
        plt.show()
        g = input('Accept thresholding? (y/n) \n')
        to_continue = 0
        while to_continue == 0:
            if g == 'y' or g == 'Y':
                good = 1
                to_continue = 1
            elif g =='n' or g =='N':
                to_continue = 1
            else:
                g = input('Please enter (y/n): \n')
        
    plt.close()
    return t

def calc_stitch(T=None):

    if T is None:
        cal_img = input('Enter path to right/top calibration image: ')
        cal_img1 = load_image2(cal_img)
        del cal_img
        #### remove hardcoding ####
        pts=6
        img_shape_in_stack_dir = cal_img1.shape[1]
        border=100
        pano_h = cal_img1.shape[0]+border
        pano_w = cal_img1.shape[1]*2
        ##############################################
        cal1 = copy.deepcopy(cal_img1)
        threshold1 = check_threshold(cal1)
        cal_img = input('Enter path to left/bottom calibration image: ')
        cal_img2 = load_image2(cal_img)
        del cal_img
        cal2 = copy.deepcopy(cal_img2)
        threshold2 = check_threshold(cal2)
        imCrop1,r1 = crop(cal_img1)
        imCrop2,r2 = crop(cal_img2)
        cal_points1 = find_points(imCrop1,threshold1,min_size=9)
        cal_points2 = find_points(imCrop2,threshold2,min_size=9)
        del cal1,cal2
        x1 = pick_shared_pts(imCrop1,cal_points1,pts) + np.array([r1[0],r1[1]])
        x2 = pick_shared_pts(imCrop2,cal_points2,pts) + np.array([r2[0],r2[1]])
        T = translation_matrix(x1,x2,pts,img_shape_in_stack_dir)
        alpha1,alpha2 = create_alpha(pano_h,pano_w,img_shape_in_stack_dir,T,border)

    else:
        cal_img = cal_img = input('Enter path to a single example image: ')
        cal_img1 = load_image(cal_img)
        del cal_img
        #### remove hardcoding ####
        img_shape_in_stack_dir = cal_img1.shape[1]
        border=0
        pano_h = cal_img1.shape[0]+border
        pano_w = cal_img1.shape[1]*2
        ##############################################
        alpha1,alpha2 = create_alpha(pano_h,pano_w,img_shape_in_stack_dir,T,border)
        
    return T, (alpha1,alpha2), border

def main():
    
    parser = argparse.ArgumentParser(description='Program to calculate stitching & blending transform and apply to input images', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-l','--top_cam',type=str,nargs='+',help = 'Input path to image files for top camera')
    parser.add_argument('-r','--bottom_cam',type=str,nargs='+',help = 'Input path to image files for bottom camera')
    parser.add_argument('-s','--save_path',type=str,nargs=1,help = 'directory in which to save panorama images')
    parser.add_argument('-t','--transform', type=str,nargs=1,help = 'path to transform matrix .txt file')
    args = parser.parse_args()
    if args.transform is None:
        T,alphas,border = calc_stitch()
        np.savetxt(args.save_path[0]+'T_matrix.txt',T)
        #np.savetxt(args.save_path[0]+'alpha_matrix.txt',alphas)
    else:
        T = np.loadtxt(args.transform[0])
        T,alphas,border = calc_stitch(T) 
    top_cam_files = glob.glob(args.top_cam[0]+'*.tif')
    bottom_cam_files = glob.glob(args.bottom_cam[0]+'*.tif')
    
    if len(top_cam_files)!=1:
        top_list = top_cam_files
        top_list.sort()
    else:
        top_list = args.top_cam_files
    if len(bottom_cam_files)!=1:
        bottom_list = bottom_cam_files
        bottom_list.sort()
    else:
        bottom_list = bottom_cam_files

    d = time.time()
    for i in range(0,len(top_list)):
        c1 = load_image(top_list[i])
        c2 = load_image(bottom_list[i])
        panorama = create_pano(c2,c1,alphas[1],alphas[0],T,border)
        panorama = panorama.astype('uint16')
        #cv2.imwrite(args.save_path[0]+'pano_%06d.tif' %i,panorama)
        skimage.io.imsave(args.save_path[0]+'pano_%06d.tiff' %i, panorama, plugin='tifffile')
    print('Finished in %0.3f s' %(time.time()-d))
if __name__ == "__main__":
    main()