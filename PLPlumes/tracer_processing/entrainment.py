#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:19:34 2020

@author: ajp25
"""


import numpy as np
#from PLPlumes.tracer_processing.interpolate_at_boundary import plume_outline,q_at_p
from PLPlumes.pio import pivio,imgio
from scipy.ndimage import median_filter
from skimage.morphology import dilation,square,binary_erosion
import scipy.ndimage.measurements as measurements
from scipy import interpolate
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import rc
#from matplotlib.ticker import *
#import matplotlib.gridspec as gridspec
#import matplotlib.patches as patches
#from matplotlib.lines import Line2D
#from matplotlib.font_manager import FontProperties
#from mpl_toolkits.axes_grid1 import make_axes_locatable
rc('text', usetex=True)

from PLPlumes.plume_processing.plume_functions import windowed_average, plume_outline
from PLPlumes import cmap_sunset
#%%
v1_cal = 6.64e-5
v2_cal = 6.76e-5
v3_cal = 7.689e-5
v4_cal = 7.75e-5
v5_cal = 7.52e-5
v6_cal = 7.587e-5
v7_cal = 7.86e-5
v8_cal = 7.9-6e-5
v1_range = (1.25/39.37,7.9375/39.37)
v2_range = (.201,.373)
v3_range = (.357,.553)
v4_range = (.549,.746)
v5_range = (29.25/39.37,34/39.37)
v6_range = (34.3125/39.37,39.125/39.37)
v7_range = (39.25/39.37,44.25/39.37)
v8_range = (44.3125/39.37, 49.3125/39.37)
deltat = 1/(300) #s
#%% load in tracer imgs & piv 
#%% Plume dn32 files
pdn32_v1 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/plume_dn32_v1.img')
pdn32_v2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/plume_dn32_v2.img')
pdn32_v3 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View3/processed/quiescent/plume_dn32_v3.img')
pdn32_v4 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/plume_dn32_v4.img')
pdn32_v5 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View5/processed/quiescent/plume_dn32_v5.img')
pdn32_v6 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View6/processed/quiescent/plume_dn32_v6.img')
pdn32_v7 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View7/processed/quiescent/plume_dn32_v7.img')
pdn32_v8 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View8/processed/quiescent/plume_dn32_v8.img')

pivdn32_v1 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/plume_dn32_v1.0064.def.msk.ave.piv')
pivdn32_v2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/plume_dn32_v2.0064.def.msk.ave.piv')
pivdn32_v3 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View3/processed/quiescent/plume_dn32_v3.0064.def.msk.ave.piv')
pivdn32_v4 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/plume_dn32_v4.0064.def.msk.ave.piv')
pivdn32_v5 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View5/processed/quiescent/plume_dn32_v5.0064.def.msk.ave.piv')
pivdn32_v6 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View6/processed/quiescent/plume_dn32_v6.0064.def.msk.ave.piv')
pivdn32_v7 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View7/processed/quiescent/plume_dn32_v7.0064.def.msk.ave.piv')
pivdn32_v8 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View8/processed/quiescent/plume_dn32_v8.0064.def.msk.ave.piv')

#%%
#%%
#%%
img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/tracers_dn32_v4.img')

piv = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/tracers_dn32_v4.tracers.bsub.0032.def.piv')

f=1100
plume_outline_pts,frame_mask = plume_outline(img.read_frame2d(f),40,2,2600,100)
  
pts = np.array([np.array([r[0]-piv.dx,r[1]]) for r in plume_outline_pts])    

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
#plt.figure(31);plt.plot(pts[:,1],pts[:,0],'r.')       

u_at_p = q_at_p(pts,u.transpose(),piv.dx)
v_at_p = q_at_p(pts,v.transpose(),piv.dx)
plt.figure(31);plt.quiver(pts[:,1],pts[:,0],u_at_p[:],v_at_p[:],color='r',scale=100)
plt.figure(31);plt.plot(pts[:,1],pts[:,0],'r.')  

file = '/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/entrainment_v1.hdf5'
h5 = h5py.File(file,'r')
dset = np.zeros((f,img.ix))
results = []
for f in h5['frames'].keys():
    results.append(h5['frames'][f][:])
for c in range(0,img.ix):
    count = 0
    col_mean = 0
    for i in results[:,c]:
        for ii in i:
            col_mean+=ii
            count+=1
    col_mean = col_mean/count
    dset[c] = col_mean