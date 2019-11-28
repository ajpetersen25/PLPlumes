#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:17:35 2019

@author: alec
"""

from __future__ import division
import numpy as np
import numpy.ma as nma
from pio import imgio
import apply_mask
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from matplotlib import colors as clr
rc('text', usetex=True)
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
from scipy.signal import correlate2d
from lmfit import Model
import cv2
import cmap_sunset
#%% 