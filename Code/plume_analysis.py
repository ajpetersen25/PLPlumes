#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plume PIV Processing

Created on Mon Feb  3 16:34:04 2020

@author: alec
"""
# Imports
from openpiv import tools, pyprocess, scaling, filters, \
                    validation, process
import numpy as np
from skimage import data
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import feature
from PIL import Image

from skimage.color import rgb2gray
from skimage import img_as_uint

#%%
# Plume v. Tracer Separation
