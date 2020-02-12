#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:50:55 2020

@author: alec
"""
import numpy as np
import cv2
import skimage.io

def load_npz(file_name):
    arr = np.load(file_name)
    array = arr['arr_0']
    return array


def load_image(img_file):
    img_bgr = cv2.imread(img_file)
    img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    return img

def load_tif(img_file):

    img = skimage.io.imread(img_file,plugin='tifffile')
    return img

def write_tif(name,array):
    skimage.io.imsave(name,array.astype('uint16'),plugin='tifffile')

