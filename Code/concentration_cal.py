#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 14:10:38 2019

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
from scipy import stats

from copy import deepcopy
import cv2
import cmap_sunset
from datetime import datetime
#%%
path = '/media/cluster/msi/pet00105/Coletti/Data_2019/Plumes/whole_plume/'
img_file = path+'plume_30um_dn32_D019/pano/Analysis/pano.img'
deltat = 1/500 #s
cal = 3581 #pix/m
D_0 = 1.905e-2 #m
dx = 24
outlet_z_pix = 197
image = imgio.imgio(img_file)
#%% concentration calibration images
c1 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2019/Plumes/whole_plume/concentration/dn_235_2in/Raw/pano.avg.img').read_frame2d(0)
c2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2019/Plumes/whole_plume/concentration/dn_32_2in/Raw/pano.avg.img').read_frame2d(0)
c3 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2019/Plumes/whole_plume/concentration/dn_235_1in/Raw/pano.avg.img').read_frame2d(0)
#c4 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2019/Plumes/whole_plume/concentration/dn_465_1in/Raw/pano.avg.img').read_frame2d(0)
#c5 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2019/Plumes/whole_plume/concentration/dn_58_1in/Raw/pano.avg.img').read_frame2d(0)
#%%
c_imgs = [c1,c2,c3]
phis = [1.6e-3,3.6e-3,6.5e-3]
centerlines= []
for c in c_imgs:
    cc = np.zeros((2,c.shape[1]))
    for r in range(0,c.shape[1]):
        cc[0,r] = int(r)#((r*dx+dx+0.5)-outlet_z_pix)
        try:
            cc[1,r] = int(np.where(c[:,r] == nma.max(c[:,:],axis=0)[r])[0])
        except:
            cc[1,r] = np.nan
    cc = cc[:,~np.isnan(cc[1])].astype('int')
    centerlines.append(cc[:,cc[0]>=outlet_z_pix])
#%%
def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2
#%%
nbins=100
#nbins=10
to_calc = []
for i,img in enumerate(c_imgs):
    a = []
    for i in range(0,len(centerlines[i][1])):
        a.append(np.mean(img[100:150,centerlines[0][0][i]]))
    to_calc.append(np.array(a))
c1_stats = stats.binned_statistic(centerlines[0][0],to_calc[0],statistic='mean',bins=nbins)
c2_stats = stats.binned_statistic(centerlines[1][0],to_calc[1],statistic='mean',bins=nbins)
c3_stats = stats.binned_statistic(centerlines[2][0],to_calc[2],statistic='mean',bins=nbins)
#c4_stats = stats.binned_statistic(centerlines[3][0],to_calc[3],statistic='mean',bins=nbins)
#c5_stats = stats.binned_statistic(centerlines[4][0],to_calc[4],statistic='mean',bins=nbins)
plot_bins = c1_stats.bin_edges+(np.diff(c1_stats.bin_edges)[0]/2)
c_stats = [c1_stats.statistic,c2_stats.statistic,c3_stats.statistic]
plt.figure();plt.plot([c[10] for c in c_stats],phis,'ko')
alphas = []
r_sqr = []
for i in range(0,nbins):
    p=np.polyfit([c[i] for c in c_stats],phis,1)[0]
    alphas.append(p)
    r_sqr.append(rsquared(phis,[p*c[i] for c in c_stats]))
alphas = np.array(alphas)
#%% apply conversion
cmap_bins = c1_stats.bin_edges.astype('int')
test = image.read_frame2d(4000).astype('float')
ctest = np.zeros(test.shape)
for i in range(1,len(cmap_bins)):
    ctest[:,cmap_bins[i-1]-1:cmap_bins[i]+1] = test[:,cmap_bins[i-1]-1:cmap_bins[i]+1]*alphas[i-1]