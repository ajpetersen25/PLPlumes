#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plume PIV Processing

Created on Mon Feb  3 16:34:04 2020

@author: alec
"""
# Imports
import glob
from skimage import data
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import feature
from PIL import Image

from skimage.color import rgb2gray
from skimage import img_as_uint
# Data base path
base_path = '/media/cluster/msi/pet00105/Coletti/Data_2019/Plumes/zoomed/'
#%%
# Plume v. Tracer Separation
# run separation.py from command line (hopefully on a node with many cores)

#%% PIV

from openpiv import tools, pyprocess, scaling, filters, validation, process, preprocess

                    
#%% Plume PIV
# View1 image path
v1_ip = base_path + 'View1/Plume_dn32/'                  
frame_list = sorted(glob.glob(v1_ip+'*.tif'))


# Dynamic plume masking



#%% Tracer PIV



# Dynamic plume masking


# background subtraction
