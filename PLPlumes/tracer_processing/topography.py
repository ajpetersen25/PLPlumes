#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:32:50 2021

@author: apetersen
"""

import numpy as np
from scipy.stats import binned_statistic as bs
import h5py
from PLPlumes.pio import pivio, imgio
from PLPlumes.plume_processing.plume_functions import windowed_average as windowed_average
import os
import time 
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.signal import fftconvolve
from scipy.interpolate import griddata
def fftcorrelate2d(a,b,save=False,savepath=None,trim=None):
    ac = fftconvolve(a, b[::-1,::-1])
    cut = np.ceil(np.array(ac.shape)/2).astype('int')
    ac = ac[0:cut[0],cut[1]:]
    if trim:
        ac = ac[:,0:trim]
    if save is True and savepath:
        np.save(savepath,ac)
    return(ac)


from PLPlumes.plume_processing.plume_functions import orientation as orientation
from PLPlumes.plume_processing.plume_functions import curvature as curvature
#%%
v1_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View1/*u_e.*npy')
v2_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View2/*u_e.npy')
v3_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View3/*u_e.npy')
v4_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View4/*u_e.npy')
v5_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View5/*u_e.npy')
v6_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View6/*u_e.npy')
v7_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View7/*u_e.new.npy')
v8_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View8/*u_e.npy')

"""v1_list = glob.glob('/media/apetersen/Photography/P2_entrainment/View1/*u_e.npy')
v2_list = glob.glob('/media/apetersen/Photography/P2_entrainment/View2/*u_e.npy')
v3_list = glob.glob('/media/apetersen/Photography/P2_entrainment/View3/*u_e.npy')
v4_list = glob.glob('/media/apetersen/Photography/P2_entrainment/View4/*u_e.npy')
v5_list = glob.glob('/media/apetersen/Photography/P2_entrainment/View5/*u_e.npy')
v6_list = glob.glob('/media/apetersen/Photography/P2_entrainment/View6/*u_e.npy')
v7_list = glob.glob('/media/apetersen/Photography/P2_entrainment/View7/*u_e.npy')
v8_list = glob.glob('/media/apetersen/Photography/P2_entrainment/View8/*u_e.npy')"""



v1_ppi = 412.0012 #pix per in
v1_range_in = np.array([190/v1_ppi,190/v1_ppi+2560/v1_ppi ])#inches
v1_range = v1_range_in*0.0254 # meters
v1_cal = v1_ppi/.0254 # pix/m


v2_ppi = np.sqrt((2278-1872)**2+(1241-1245)**2) #pix per in
v2_range_in = np.array([7-(2560-2278)/v2_ppi, (7-(2560-2278)/v2_ppi) + (2560/v2_ppi)]) #inches
v2_range = v2_range_in*0.0254 # meters
v2_cal = v2_ppi/.0254 # pix/m


v3_ppi = np.sqrt((2359-1945)**2+(429-430)**2) #pix per in
v3_range_in = np.array([18-6-(2560-2359)/v3_ppi, (18-6-(2560-2359)/v3_ppi) + (2560/v3_ppi)]) #inches
v3_range = v3_range_in*0.0254 # meters
v3_cal = v3_ppi/.0254 # pix/m


v4_ppi = np.sqrt((2325-1915)**2+(419-419)**2) #pix per in
v4_range_in = np.array([24-6-(2560-2325)/v4_ppi, (24-6-(2560-2325)/v4_ppi) + (2560/v4_ppi)]) #inches
v4_range = v4_range_in*0.0254 # meters
v4_cal = v4_ppi/.0254 # pix/m

v5_ppi = np.sqrt((1906-1496)**2+(992-999)**2) #pix per in
v5_range_in = np.array([30-6-(2560-2313)/v5_ppi, (30-6-(2560-2313)/v5_ppi) + (2560/v5_ppi)]) #inches
v5_range = v5_range_in*0.0254 # meters
v5_cal = v5_ppi/.0254 # pix/m

v6_ppi = np.sqrt((1122-1532)**2+(606-600)**2) #pix per in
v6_range_in = np.array([36-6-(2560-2354)/v6_ppi, (36-6-(2560-2354)/v6_ppi) + (2560/v6_ppi)]) #inches
v6_range = v6_range_in*0.0254 # meters
v6_cal = v6_ppi/.0254 # pix/m


v7_ppi = np.sqrt((154-562)**2+(912-906)**2) #pix per in
v7_range_in = np.array([44-(154)/v7_ppi, (44-(154)/v7_ppi) + (2560/v7_ppi)]) #inches
v7_range = v7_range_in*0.0254 # meters
v7_cal = v7_ppi/.0254 # pix/m

v8_ppi = np.sqrt((477-874)**2+(5)**2) #pix per in
v8_range_in = np.array([51-(78)/v8_ppi, (51-(78)/v8_ppi) + (2560/v8_ppi)]) #inches
v8_range = v8_range_in*0.0254 # meters
v8_cal = v8_ppi/.0254 # pix/m

z1 = np.linspace(v1_range[0],v1_range[1],2560)
z2 = np.linspace(v2_range[0],v2_range[1],2560)
z3 = np.linspace(v3_range[0],v3_range[1],2560)
z4 = np.linspace(v4_range[0],v4_range[1],2560)
z5 = np.linspace(v5_range[0],v5_range[1],2560)
z6 = np.linspace(v6_range[0],v6_range[1],2560)
z7 = np.linspace(v7_range[0],v7_range[1],2560)
z8 = np.linspace(v8_range[0],v8_range[1],2560)
zs = np.hstack((z1,z2,z3,z4,z5,z6,z7,z8))


#%% curvature and orientation pdfs at different heights
positions = [200,1600]
kernel1 = 10
kernel2 = 100
o1sa = []
o1sb = []
k1sa = []
k1sb = []
for v in v1_list:
    v1 = np.load(v,allow_pickle=True)

    for f in np.arange(0,len(v1),10):
        ks = []
        os = []
        pts = v1[f][:,0:2]
        z = 2560 - pts[:,0][::-1]
        #z = (z/v1_cal +v1_range[0])*v1_cal
        r = pts[:,1][::-1] - (1600-1532)
        k = curvature(z,r,pts[::-1])
        n = v1[f][:,6:8]
        o = orientation(n,np.array([-1,0]))
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            ks.append(k[position-kernel1:position+kernel1])
            os.append(o[position-kernel2:position+kernel2])
        o1sa.extend(os[0])
        o1sb.extend(os[1])
        k1sa.extend(ks[0])
        k1sb.extend(ks[1])
o1s = [o1sa,o1sb]
o1s = np.array(o1s)     
k1s = [k1sa,k1sb]
k1s = np.array(k1s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/k1s.npy',k1s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/o1s.npy',o1s)
del k1s, o1s

o2sa = []
o2sb = []
k2sa = []
k2sb = []
for v in v2_list:
    v1 = np.load(v,allow_pickle=True)

    for f in np.arange(0,len(v1),10):
        ks = []
        os = []
        pts = v1[f][:,0:2]
        z = 2560 - pts[:,0][::-1]
        #z = (z/v1_cal +v1_range[0])*v1_cal
        r = pts[:,1][::-1] - (1600-1450)
        k = curvature(z,r,pts[::-1])
        n = v1[f][:,6:8]
        o = orientation(n,np.array([-1,0]))
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            ks.append(k[position-kernel1:position+kernel1])
            os.append(o[position-kernel2:position+kernel2])
        o2sa.extend(os[0])
        o2sb.extend(os[1])
        k2sa.extend(ks[0])
        k2sb.extend(ks[1])
o2s = [o2sa,o2sb]
o2s = np.array(o2s)     
k2s = [k2sa,k2sb]
k2s = np.array(k2s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/k2s.npy',k2s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/o2s.npy',o2s)
del k2s, o2s

o3sa = []
o3sb = []
k3sa = []
k3sb = []
for v in v3_list:
    v1 = np.load(v,allow_pickle=True)

    for f in np.arange(0,len(v1),10):
        ks = []
        os = []
        pts = v1[f][:,0:2]
        z = 2560 - pts[:,0][::-1]
        #z = (z/v1_cal +v1_range[0])*v1_cal
        r = pts[:,1][::-1] + (3*.0254)*v3_cal - 799
        k = curvature(z,r,pts[::-1])
        n = v1[f][:,6:8]
        o = orientation(n,np.array([-1,0]))
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            ks.append(k[position-kernel1:position+kernel1])
            os.append(o[position-kernel2:position+kernel2])
        o3sa.extend(os[0])
        o3sb.extend(os[1])
        k3sa.extend(ks[0])
        k3sb.extend(ks[1])
o3s = [o3sa,o3sb]
o3s = np.array(o3s)     
k3s = [k3sa,k3sb]
k3s = np.array(k3s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/k3s.npy',k3s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/o3s.npy',o3s)
del k3s, o3s

o4sa = []
o4sb = []
k4sa = []
k4sb = []
for v in v4_list:
    v1 = np.load(v,allow_pickle=True)

    for f in np.arange(0,len(v1),10):
        ks = []
        os = []
        pts = v1[f][:,0:2]
        z = 2560 - pts[:,0][::-1]
        #z = (z/v1_cal +v1_range[0])*v1_cal
        r = pts[:,1][::-1] + (3*.0254)*v4_cal - 783
        k = curvature(z,r,pts[::-1])
        n = v1[f][:,6:8]
        o = orientation(n,np.array([-1,0]))
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            ks.append(k[position-kernel1:position+kernel1])
            os.append(o[position-kernel2:position+kernel2])
        o4sa.extend(os[0])
        o4sb.extend(os[1])
        k4sa.extend(ks[0])
        k4sb.extend(ks[1])
o4s = [o4sa,o4sb]
o4s = np.array(o4s)     
k4s = [k4sa,k4sb]
k4s = np.array(k4s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/k4s.npy',k4s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/o4s.npy',o4s)
del k4s, o4s

o5sa = []
o5sb = []
k5sa = []
k5sb = []
for v in v5_list:
    v1 = np.load(v,allow_pickle=True)

    for f in np.arange(0,len(v1),10):
        ks = []
        os = []
        pts = v1[f][:,0:2]
        z = 2560 - pts[:,0][::-1]
        #z = (z/v1_cal +v1_range[0])*v1_cal
        r = pts[:,1][::-1] + (4.5*.0254)*v5_cal - 1002
        k = curvature(z,r,pts[::-1])
        n = v1[f][:,6:8]
        o = orientation(n,np.array([-1,0]))
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            ks.append(k[position-kernel1:position+kernel1])
            os.append(o[position-kernel2:position+kernel2])
        o5sa.extend(os[0])
        o5sb.extend(os[1])
        k5sa.extend(ks[0])
        k5sb.extend(ks[1])
o5s = [o5sa,o5sb]
o5s = np.array(o5s)     
k5s = [k5sa,k5sb]
k5s = np.array(k5s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/k5s.npy',k5s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/o5s.npy',o5s)
del k5s, o5s


o6sa = []
o6sb = []
k6sa = []
k6sb = []
for v in v6_list:
    v1 = np.load(v,allow_pickle=True)

    for f in np.arange(0,len(v1),10):
        ks = []
        os = []
        pts = v1[f][:,0:2]
        z = 2560 - pts[:,0][::-1]
        #z = (z/v1_cal +v1_range[0])*v1_cal
        r = pts[:,1][::-1] + (4.5*.0254)*v6_cal - 967
        k = curvature(z,r,pts[::-1])
        n = v1[f][:,6:8]
        o = orientation(n,np.array([-1,0]))
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            ks.append(k[position-kernel1:position+kernel1])
            os.append(o[position-kernel2:position+kernel2])
        o6sa.extend(os[0])
        o6sb.extend(os[1])
        k6sa.extend(ks[0])
        k6sb.extend(ks[1])
o6s = [o6sa,o6sb]
o6s = np.array(o6s)     
k6s = [k6sa,k6sb]
k6s = np.array(k6s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/k6s.npy',k6s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/o6s.npy',o6s)

del k6s, o6s

o7sa = []
o7sb = []
k7sa = []
k7sb = []
for v in v7_list:
    v1 = np.load(v,allow_pickle=True)

    for f in np.arange(0,len(v1),10):
        ks = []
        os = []
        pts = v1[f][:,0:2]
        z = 2560 - pts[:,0][::-1]
        #z = (z/v1_cal +v1_range[0])*v1_cal
        r = pts[:,1][::-1] - (1600-1260) + 4.6*0.0254*v7_cal
        k = curvature(z,r,pts[::-1])
        n = v1[f][:,6:8]
        o = orientation(n,np.array([-1,0]))
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            ks.append(k[position-kernel1:position+kernel1])
            os.append(o[position-kernel2:position+kernel2])
        o7sa.extend(os[0])
        o7sb.extend(os[1])
        k7sa.extend(ks[0])
        k7sb.extend(ks[1])
o7s = [o7sa,o7sb]
o7s = np.array(o7s)     
k7s = [k7sa,k7sb]
k7s = np.array(k7s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/k7s.npy',k7s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/o7s.npy',o7s)
del k7s, o7s


o8sa = []
o8sb = []
k8sa = []
k8sb = []
for v in v8_list:
    v1 = np.load(v,allow_pickle=True)

    for f in np.arange(0,len(v1),10):
        ks = []
        os = []
        pts = v1[f][:,0:2]
        z = 2560 - pts[:,0][::-1]
        #z = (z/v1_cal +v1_range[0])*v1_cal
        r = pts[:,1][::-1] - (1600-1274) + 4.6*0.0254*v8_cal
        k = curvature(z,r,pts[::-1])
        n = v1[f][:,6:8]
        o = orientation(n,np.array([-1,0]))
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            ks.append(k[position-kernel1:position+kernel1])
            os.append(o[position-kernel2:position+kernel2])
        o8sa.extend(os[0])
        o8sb.extend(os[1])
        k8sa.extend(ks[0])
        k8sb.extend(ks[1])
o8s = [o8sa,o8sb]
o8s = np.array(o8s)     
k8s = [k8sa,k8sb]
k8s = np.array(k8s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/k8s.npy',k8s)
np.save('/media/apetersen/Backup4/P2_entrainment/topography/o8s.npy',o8s)

del k8s, o8s     

#%%
k1s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/k1s.npy',allow_pickle=True)
k2s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/k2s.npy',allow_pickle=True)
k3s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/k3s.npy',allow_pickle=True)
k4s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/k4s.npy',allow_pickle=True)
k5s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/k5s.npy',allow_pickle=True)
k6s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/k6s.npy',allow_pickle=True)
k7s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/k7s.npy',allow_pickle=True)
k8s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/k8s.npy',allow_pickle=True)

kh1a,kb1a = np.histogram(k1s[0]*v1_cal,bins=50,density=True,range=(-400,400))
kh1b,kb1b = np.histogram(k1s[1]*v1_cal,bins=50,density=True,range=(-400,400))

kh2a,kb2a = np.histogram(k2s[0]*v2_cal,bins=50,density=True,range=(-400,400))
kh2b,kb2b = np.histogram(k2s[1]*v2_cal,bins=50,density=True,range=(-400,400))

kh3a,kb3a = np.histogram(np.array(k3s[0])*v3_cal,bins=50,density=True,range=(-400,400))
kh3b,kb3b = np.histogram(np.array(k3s[1])*v3_cal,bins=50,density=True,range=(-400,400))

kh4a,kb4a = np.histogram(np.array(k4s[0])*v4_cal,bins=50,density=True,range=(-400,400))
kh4b,kb4b = np.histogram(np.array(k4s[1])*v4_cal,bins=50,density=True,range=(-400,400))

kh5a,kb5a = np.histogram(np.array(k5s[0])*v5_cal,bins=50,density=True,range=(-400,400))
kh5b,kb5b = np.histogram(np.array(k5s[1])*v5_cal,bins=50,density=True,range=(-400,400))

kh6a,kb6a = np.histogram(np.array(k6s[0])*v6_cal,bins=50,density=True,range=(-400,400))
kh6b,kb6b = np.histogram(np.array(k6s[1])*v6_cal,bins=50,density=True,range=(-400,400))

kh7a,kb7a = np.histogram(np.array(k7s[0])*v7_cal,bins=50,density=True,range=(-400,400))
kh7b,kb7b = np.histogram(np.array(k7s[1])*v7_cal,bins=50,density=True,range=(-400,400))

kh8a,kb8a = np.histogram(np.array(k8s[0])*v8_cal,bins=50,density=True,range=(-400,400))
kh8b,kb8b = np.histogram(np.array(k8s[1])*v8_cal,bins=50,density=True,range=(-400,400))

z_pos = np.array([z1[2560-200],z1[2560-1600],z2[2560-200],z2[2560-1600],z3[2560-200],z3[2560-1600],
                  z4[2560-200],z4[2560-1600],z5[2560-200],z5[2560-1600],
         z6[2560-200],z6[2560-1600],z7[2560-200],z7[2560-1600],z8[2560-200],z8[2560-1600]])/1.905e-2
kb1a = (kb1a+np.diff(kb1a)[0]/2)[:-1]
kb1b = (kb1b+np.diff(kb1b)[0]/2)[:-1]
kb2a = (kb2a+np.diff(kb2a)[0]/2)[:-1]
kb2b = (kb2b+np.diff(kb2b)[0]/2)[:-1]
kb3a = (kb3a+np.diff(kb3a)[0]/2)[:-1]
kb3b = (kb3b+np.diff(kb3b)[0]/2)[:-1]
kb4a = (kb4a+np.diff(kb4a)[0]/2)[:-1]
kb4b = (kb4b+np.diff(kb4b)[0]/2)[:-1]
kb5a = (kb5a+np.diff(kb5a)[0]/2)[:-1]
kb5b = (kb5b+np.diff(kb5b)[0]/2)[:-1]
kb6a = (kb6a+np.diff(kb6a)[0]/2)[:-1]
kb6b = (kb6b+np.diff(kb6b)[0]/2)[:-1]
kb7a = (kb7a+np.diff(kb7a)[0]/2)[:-1]
kb7b = (kb7b+np.diff(kb7b)[0]/2)[:-1]
kb8a = (kb8a+np.diff(kb8a)[0]/2)[:-1]
kb8b = (kb8b+np.diff(kb8b)[0]/2)[:-1]
#%% plot pdfs colored by height

cs = plt.get_cmap('magma')
cNorm = colors.Normalize(vmin=0,vmax=(z_pos[-1]+10))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
khs = [kh1a,kh1b,kh2a,kh2b,kh3a,kh3b,kh4a,kh4b,kh5a,kh5b,kh6a,kh6b,kh7a,kh7b,kh8a,kh8b]
kbs = [kb1a,kb1b,kb2a,kb2b,kb3a,kb3b,kb4a,kb4b,kb5a,kb5b,kb6a,kb6b,kb7a,kb7b,kb8a,kb8b]

f,ax = plt.subplots(figsize=(12,8))
for i in range(0,16):
    ax.plot(-kbs[i]*z_pos[i],khs[i],color=scalarMap.to_rgba(z_pos[i]),lw=2)
ax.set_xlabel(r'$\kappa z$',fontsize=20,labelpad=15);
ax.set_ylabel(r'$PDF$',fontsize=20,labelpad=15);
cbar=plt.colorbar(scalarMap);
cbar.set_label(r'$z/D_0$',fontsize=20);
plt.tight_layout();


        
#%% orientation pdfs with height
        
o1s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/o1s.npy',allow_pickle=True)
o2s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/o2s.npy',allow_pickle=True)
o3s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/o3s.npy',allow_pickle=True)
o4s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/o4s.npy',allow_pickle=True)
o5s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/o5s.npy',allow_pickle=True)
o6s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/o6s.npy',allow_pickle=True)
o7s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/o7s.npy',allow_pickle=True)
o8s = np.load('/media/apetersen/Backup4/P2_entrainment/topography/o8s.npy',allow_pickle=True)

oh1a,ob1a = np.histogram(np.array(o1s[0]),bins=90,density=True,range=(-180,180))
oh1b,ob1b = np.histogram(np.array(o1s[1]),bins=90,density=True,range=(-180,180))

oh2a,ob2a = np.histogram(np.array(o2s[0]),bins=90,density=True,range=(-180,180))
oh2b,ob2b = np.histogram(np.array(o2s[1]),bins=90,density=True,range=(-180,180))

oh3a,ob3a = np.histogram(np.array(o3s[0]),bins=90,density=True,range=(-180,180))
oh3b,ob3b = np.histogram(np.array(o3s[1]),bins=90,density=True,range=(-180,180))

oh4a,ob4a = np.histogram(np.array(o4s[0]),bins=90,density=True,range=(-180,180))
oh4b,ob4b = np.histogram(np.array(o4s[1]),bins=90,density=True,range=(-180,180))

oh5a,ob5a = np.histogram(np.array(o5s[0]),bins=90,density=True,range=(-180,180))
oh5b,ob5b = np.histogram(np.array(o5s[1]),bins=90,density=True,range=(-180,180))

oh6a,ob6a = np.histogram(np.array(o6s[0]),bins=90,density=True,range=(-180,180))
oh6b,ob6b = np.histogram(np.array(o6s[1]),bins=90,density=True,range=(-180,180))

oh7a,ob7a = np.histogram(np.array(o7s[0]),bins=90,density=True,range=(-180,180))
oh7b,ob7b = np.histogram(np.array(o7s[1]),bins=90,density=True,range=(-180,180))

oh8a,ob8a = np.histogram(np.array(o8s[0]),bins=90,density=True,range=(-180,180))
oh8b,ob8b = np.histogram(np.array(o8s[1]),bins=90,density=True,range=(-180,180))

z_pos = np.array([z1[2560-200],z1[2560-1600],z2[2560-200],z2[2560-1600],z3[2560-200],z3[2560-1600],
                  z4[2560-200],z4[2560-1600],z5[2560-200],z5[2560-1600],
         z6[2560-200],z6[2560-1600],z7[2560-200],z7[2560-1600],z8[2560-200],z8[2560-1600]])/1.905e-2
ob1a = (ob1a+np.diff(ob1a)[0]/2)[:-1]
ob1b = (ob1b+np.diff(ob1b)[0]/2)[:-1]
ob2a = (ob2a+np.diff(ob2a)[0]/2)[:-1]
ob2b = (ob2b+np.diff(ob2b)[0]/2)[:-1]
ob3a = (ob3a+np.diff(ob3a)[0]/2)[:-1]
ob3b = (ob3b+np.diff(ob3b)[0]/2)[:-1]
ob4a = (ob4a+np.diff(ob4a)[0]/2)[:-1]
ob4b = (ob4b+np.diff(ob4b)[0]/2)[:-1]
ob5a = (ob5a+np.diff(ob5a)[0]/2)[:-1]
ob5b = (ob5b+np.diff(ob5b)[0]/2)[:-1]
ob6a = (ob6a+np.diff(ob6a)[0]/2)[:-1]
ob6b = (ob6b+np.diff(ob6b)[0]/2)[:-1]
ob7a = (ob7a+np.diff(ob7a)[0]/2)[:-1]
ob7b = (ob7b+np.diff(ob7b)[0]/2)[:-1]
ob8a = (ob8a+np.diff(ob8a)[0]/2)[:-1]
ob8b = (ob8b+np.diff(ob8b)[0]/2)[:-1]        
#%% plot pdfs colored by height

cs = plt.get_cmap('magma')
cNorm = colors.Normalize(vmin=0,vmax=(z_pos[-1]+5))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
ohs = [oh1a,oh1b,oh2a,oh2b,oh3a,oh3b,oh4a,oh4b,oh5a,oh5b,oh6a,oh6b,oh7a,oh7b,oh8a,oh8b]
obs = [ob1a,ob1b,ob2a,ob2b,ob3a,ob3b,ob4a,ob4b,ob5a,ob5b,ob6a,ob6b,ob7a,ob7b,ob8a,ob8b]

f,ax = plt.subplots(figsize=(12,8))
for i in range(0,16):
    ax.plot(obs[i],ohs[i],color=scalarMap.to_rgba(z_pos[i]),marker='o',linestyle='None')
ax.set_xlabel(r'$\theta$ [degrees]',fontsize=20,labelpad=15);
ax.set_ylabel(r'$PDF$',fontsize=20,labelpad=15);
cbar=plt.colorbar(scalarMap);
cbar.set_label(r'$z/D_0$',fontsize=20);
plt.tight_layout();

#%% calculate and save mean interface position
v_lists = [v1_list,v2_list,v3_list,v4_list,v5_list,v6_list,v7_list,v8_list]
r_adjs = [- (1600-1532),- (1600-1450),
          + (3*.0254)*v3_cal - 799,+ (3*.0254)*v4_cal - 783,+ (4.5*.0254)*v5_cal - 1002,
          + (4.5*.0254)*v6_cal - 967,- (1600-1260) + 4.6*0.0254*v7_cal,-(1600-1274) + 4.6*0.0254*v8_cal]

z_adjs = [v1_range[0],v2_range[0],v3_range[0],v4_range[0],v5_range[0],v6_range[0],v7_range[0],v8_range[0]]

cals = [v1_cal,v2_cal,v3_cal,v4_cal,v5_cal,v6_cal,v7_cal,v8_cal]

rI_means = []
for vvs,vs in enumerate(v_lists):
    if vvs == 1 or vvs == 2:
        pass
    else:
        rIs = []
        print('Running View %d:\n' %(vvs+1))
        for v in vs:          
            v1 = np.load(v,allow_pickle=True)
            v2_b_pos = np.zeros((2560,))
            v2_b_c = np.zeros((2560,))
            
            for f in np.arange(0,len(v1)):
                pts = v1[f][:,0:2][::-1]
                r = (pts[:,1] + r_adjs[3])
                v2s = bs(pts[:,0],pts[:,1],statistic=np.nansum,bins=2560,range=(0,2560))
                v2_b_pos += v2s.statistic
                v2_b_c += bs(pts[:,0],~np.isnan(pts[:,1]),statistic=np.nansum,bins=2560,range=(0,2560)).statistic
            
            rI = v2_b_pos/v2_b_c
            rIs.append(rI)
        rI_means.append(rIs)
        
rI_means = np.array(rI_means)
        


#%% arrays for jpdfs
        


all_ues = []
all_ris = []
all_ks = []
all_ts = []
all_zs = []
for vvs,vs in enumerate(v_lists):
    
    for v in vs:
        v1 = np.load(v,allow_pickle=True)

        for f in np.arange(0,len(v1),40):
            pts = v1[f][:,0:2][::-1]
            uf = v1[f][:, 2:4]
            up = v1[f][:, 4:6]
            n = v1[f][:, 6:8]
            ufmag = -np.sum(uf*n, axis=1)
            upmag = -np.sum(up*n, axis=1)
            ue = ufmag/3 - upmag
            ue = ue[::-1]/cals[vvs]*1000
            z = (2560 - pts[:,0])/cals[vvs] + z_adjs[vvs]
            r = (pts[:,1] + r_adjs[vvs])/cals[vvs]
            k = curvature(z,r,pts)*cals[vvs]
            o = orientation(n[::-1],np.array([-1,0]))
            all_ues.extend(ue)
            all_ris.extend(r)
            all_ks.extend(k)
            all_ts.extend(o)
            all_zs.extend(z)
            
#%%
H,xe,ye = np.histogram2d(ue[~np.isnan(ue)],-ks[~np.isnan(ue)],bins=500,density=True,range=[[-.5,.5],[-300,350]])
f,ax = plt.subplots()
cs = plt.get_cmap('magma')
cNorm = colors.Normalize(vmin=0,vmax=H.max())
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
ax.pcolormesh(ye/.155,xe*1.905e-2,H,cmap='magma');
cbar=plt.colorbar(scalarMap);
ax.axhline(0,color='w');
ax.axvline(0,color='w');
ax.set_xlabel('$\kappa$' '[rad/m]',fontsize=20,labelpad=15);
ax.set_ylabel('$u_e$ [m/s]',fontsize=20,labelpad=15);  
ax.tick_params(axis='both', labelsize=14);
#%% interface position spacetime plot
cs = plt.get_cmap('gray')
cNorm = colors.Normalize(vmin=1.4,vmax=5)
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)

v_lists = [v1_list,v2_list,v3_list,v4_list,v5_list,v6_list,v7_list,v8_list]
r_adjs = [- (1600-1532),- (1600-1450),
          + (3*.0254)*v3_cal - 799,+ (3*.0254)*v4_cal - 783,+ (4.5*.0254)*v5_cal - 1002,
          + (4.5*.0254)*v6_cal - 967,- (1600-1260) + 4.6*0.0254*v7_cal,-(1600-1274) + 4.6*0.0254*v8_cal]

z_adjs = [v1_range[0],v2_range[0],v3_range[0],v4_range[0],v5_range[0],v6_range[0],v7_range[0],v8_range[0]]

cals = [v1_cal,v2_cal,v3_cal,v4_cal,v5_cal,v6_cal,v7_cal,v8_cal]

bins = 100

for vvs,vs in enumerate(v_lists):
    
    for vi,v in enumerate(vs):
        v1 = np.load(v,allow_pickle=True)
        r_map = np.zeros((bins,len(v1)))
        z_bins = np.linspace(z_adjs[vvs],z_adjs[vvs]+2560/cals[vvs],bins+1)
        for f in np.arange(0,len(v1)):
            pts = v1[f][:,0:2]
            
            r = (pts[:,1][~np.isnan(pts[:,0])] + r_adjs[vvs])/cals[vvs]
            z = (2560 - pts[:,0][~np.isnan(pts[:,0])])/cals[vvs] + z_adjs[vvs]
            r_bins = bs(z,r,bins=z_bins).statistic[::-1]
            r_map[:,f] = r_bins
        x = np.arange(0,r_map.shape[1])
        y = np.arange(0,r_map.shape[0])
        r_mapm = np.ma.masked_invalid(r_map)
        xx,yy = np.meshgrid(x,y)
        x1 = xx[~r_mapm.mask]
        y1 = yy[~r_mapm.mask]
        newarr = r_mapm[~r_mapm.mask]
        r_map_filled = griddata((x1,y1),newarr.ravel(),(xx,yy),method='linear')
        extent = np.where(np.isnan(r_map_filled))[1]
        extent.sort()
        if len(extent)>0:
            split = np.where(np.diff(extent)>1000)[0]
            if len(split)>0:
                r_map_filled = r_map_filled[:,extent[split[0]]+1:extent[split[0]+1]-1]
            elif extent[0]<1000:
                r_map_filled = r_map_filled[:,extent[-1]+1:]
            else:
                r_map_filled = r_map_filled[:,0:extent[0]]
        np.save('/media/apetersen/Backup4/P2_entrainment/topography/r_map_V%d_n%d' %(vvs+1,vi+1),r_map_filled)
    
    
#%% interface position correlation -- convection velocity    
    
r_maps = glob.glob('/media/apetersen/Backup4/P2_entrainment/topography/r_map*')
r_maps.sort()

bin_options = [5]#,2,5]
#all_r_acs = [] 
for r in r_maps:
    r_map = np.load(r)
    withnans = np.where(np.isnan(r_map))
    if len(withnans[1])>0:
        r_map = r_map[:,0:np.min(withnans[1])]
    r_acs = []
    for bo in bin_options:
        cut = 100/bo
        r_ac_exp = []
        for c in np.arange(0,bo):
            dname = os.path.dirname(r)+'/convection_vel'
            fname = os.path.basename(r)
            fname = dname+'/' +os.path.splitext(fname)[0]
            r_mean = np.nanmean(r_map[int(c*cut):int(cut)*(c+1),:],axis=1)
            r_f = r_map[int(c*cut):int(cut)*(c+1),:] - r_mean.reshape(len(r_mean),1)
            ac = fftcorrelate2d(r_f, r_f,save=False,trim=1000)
            r_ac_exp.append(ac)
    np.save(fname+'_rmean_ac.npy',np.array(r_ac_exp))



acs_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/topography/convection_vel/*_rmean_ac.npy')
acs_list.sort()
all_V_acs = []
all_zbs = []
for i in np.arange(0,40,5):
    W_convs = []
    zbs = []
    ii = int(i/5)
    z_bins = np.linspace(z_adjs[ii],z_adjs[ii]+2560/cals[ii],bins)
    r_ac1 = np.load(acs_list[i])
    r_ac2 = np.load(acs_list[i+1])
    r_ac3 = np.load(acs_list[i+2])
    r_ac4 = np.load(acs_list[i+3])
    r_ac5 = np.load(acs_list[i+4])
    for r in range(0,len(r_ac1)):
        zb = z_bins[r*20:(r*20)+20]
        zbs.append(np.mean(zb))
        r1 = r_ac1[r]
        r2 = r_ac2[r]
        r3 = r_ac3[r]
        r4 = r_ac4[r]
        r5 = r_ac5[r]
        maxes1 = []
        maxes2 = []
        maxes3 = []
        maxes4 = []
        maxes5 = []
        for zi in range(0,20):
            maxes1.extend(np.where((r1*(r1>(.3*np.max(r1))))[zi,:]==np.nanmax(r1[zi,:]))[0])
            maxes2.extend(np.where((r2*(r2>(.3*np.max(r2))))[zi,:]==np.nanmax(r2[zi,:]))[0])
            maxes3.extend(np.where((r3*(r3>(.3*np.max(r3))))[zi,:]==np.nanmax(r3[zi,:]))[0])
            maxes4.extend(np.where((r4*(r4>(.3*np.max(r4))))[zi,:]==np.nanmax(r4[zi,:]))[0])
            maxes5.extend(np.where((r5*(r5>(.3*np.max(r5))))[zi,:]==np.nanmax(r5[zi,:]))[0])
        
        fit1 = np.polyfit(np.array(maxes1)/1000,zb[-len(maxes1):],1)
        fit2 = np.polyfit(np.array(maxes2)/1000,zb[-len(maxes2):],1)
        fit3 = np.polyfit(np.array(maxes3)/1000,zb[-len(maxes3):],1)
        fit4 = np.polyfit(np.array(maxes4)/1000,zb[-len(maxes4):],1)
        fit5 = np.polyfit(np.array(maxes5)/1000,zb[-len(maxes5):],1)    
        W_convs.append(np.mean(np.array([fit1[0],fit2[0],fit3[0],fit4[0],fit5[0]])))
        
    all_zbs.extend(zbs[::-1])
    all_V_acs.extend(W_convs)

#%%

upper_cal = 1/3475.93 #m/pix
lower_cal = 1/3461.77 #m/pix
upper_angle = .00809 # rad
R = np.array([[np.cos(upper_angle),-np.sin(upper_angle)],[np.sin(upper_angle), np.cos(upper_angle)]])
deltat_w = 1/(600) #s
D0 = 1.905e-2

outlet = 130
overlap = 0.6731 - 0.6270625 #m
pivdn32_upper = pivio.pivio('/media/apetersen/Backup2/Plume_dn32/whole_plume/upper/dn32_upper.0048.def.msk.ave.piv')
pivdn32_upper2 = pivio.pivio('/media/apetersen/Backup2/Plume_dn32/whole_plume/upper/dn32_upper2.0048.def.msk.ave.piv')
pivdn32_lower = pivio.pivio('/media/apetersen/Backup2/Plume_dn32/whole_plume/lower/dn32_lower.0048.def.msk.ave.piv')
pivdn32_lower2 = pivio.pivio('/media/apetersen/Backup2/Plume_dn32/whole_plume/lower/dn32_lower2.0048.def.msk.ave.piv')

dn32_upper_piv1 = pivdn32_upper.read_frame2d(0)
dn32_upper_piv2 = pivdn32_upper2.read_frame2d(0)
zpiv1 = (np.arange(0,dn32_upper_piv1[0].shape[1])*pivdn32_upper.dx-outlet)*upper_cal/D0
dn32_lower_piv1 = pivdn32_lower.read_frame2d(0)
dn32_lower_piv2 = pivdn32_lower2.read_frame2d(0)
zpiv2 = ((np.linspace(2560,2*2560,dn32_lower_piv1[1].shape[1])-outlet)*lower_cal-overlap)/D0


tau_p = 7.4e-3
w0 = tau_p*9.81

dn32_centerline1 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W1 = ((dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]) +(dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]))/2
for p in range(0,dn32_upper_piv1[1].shape[1]):
    w_prof = W1[:,p]
    dn32_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
Wc_dn32a = windowed_average(np.mean(W1[20:22,np.where(zpiv1>3)[0].astype('int')],axis=0)[:-1],10)*upper_cal/deltat_w
dn32_centerline2 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W2 = (dn32_lower_piv1[1] + dn32_lower_piv2[1])/2
for p in range(0,dn32_lower_piv1[1].shape[1]):
    w_prof = W2[:,p]
    dn32_centerline2[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
Wc_dn32b = windowed_average(W2[[r for r in dn32_centerline2],[c for c in range(0,pivdn32_lower2.nx)]][1:-1],10)*lower_cal/deltat_w
Wc_dn32 = np.hstack((Wc_dn32a,Wc_dn32b[4:]))
zpiv = np.hstack((zpiv1[zpiv1>3][:-1],zpiv2[4:][1:-1]))
#%%
nearest = []
for zb in all_zbs:
    d = abs(zb - zpiv*D0)
    nearest.extend(zpiv[np.where(d==np.min(d))[0]])    
    
w_profs = []
wc_ac_comp = []
for n in nearest:
    if n in zpiv1:
        idx = np.where(n==zpiv1)[0][0]
        wprof = W1[:,idx]*upper_cal/deltat_w
        w_profs.append(wprof)
        wc_ac_comp.append(Wc_dn32[idx])
    elif n in zpiv2:
        idx = np.where(n==zpiv2)[0][0]
        wprof = W2[:,idx]*lower_cal/deltat_w
        w_profs.append(wprof)  
        wc_ac_comp.append(Wc_dn32[idx])
    
w_profs = np.array(w_profs)
wc_ac_comp = np.array(wc_ac_comp)
all_V_acs = np.array(all_V_acs)
#%% fractal dimension

def box_count_dim(c,dims):
    width = np.max(dims)
    p = np.log(width)/np.log(2)
    p = int(np.ceil(p))
    width = int(2**p)

        
    n = np.zeros((p+1,))
    box_tot = np.zeros((p+1,))
    
    for g in range(p,-1,-1):
        siz = 2**(p-g)

        gridx = np.linspace(0,width,siz+1)
        gridy = np.linspace(0,width,siz+1)
        H = np.histogram2d(c[:,0],c[:,1],bins=[gridx,gridy])
        
        box_tot[g] = H[0].size
        n[g] = np.sum(H[0]>0)
        
    box_sizes = 2**(np.arange(0,p+1))
    n = n
    return(n,box_sizes,box_tot)
    
#%% 
n_all = np.zeros((13,v1.shape[0]))
for f in range(0,v1.shape[0]):
    pts = v1[f][:,0:2][::-1]
    n,bs,bt = box_count_dim(pts,(1600,2560))
    n_all[:,f] = n
    

    
#%%
v = v3_list[1]
v1 = np.load(v,allow_pickle=True)



filter_sizes = np.array([2,4,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,320,380,
                         440,500,600,700,800,900,1000])+1

LsLx_ratios = np.zeros(filter_sizes.shape[0]+1,)  
Ls_all = np.zeros(filter_sizes.shape[0]+1,)
for frame in np.arange(0,len(v1)):
    pts = v1[frame][:,0:2][::-1]
    frame_Lratios = np.zeros(filter_sizes.shape[0]+1,)
    frame_Lratios[0] = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))/2560
    Ls_frame = np.zeros(filter_sizes.shape[0]+1,)
    Ls_frame[0] = np.sum(np.linalg.norm(np.diff(pts,axis=0),axis=1))
    for ff,f in enumerate(filter_sizes):
        pts_f = windowed_average(pts[:,0],f,mode='valid')
        tx = pts[int(f/2):-int(f/2),0]
        tx = tx.reshape(len(tx),1)
        ty = windowed_average(pts[:,1],f,mode='valid')
        ty = ty.reshape(len(ty),1)
        pts_temp = np.concatenate((tx,ty),axis=1)
        Ls = np.sum(np.linalg.norm(np.diff(pts_temp,axis=0),axis=1))
        Lx = 2560 - f
        LsLx = Ls/Lx
        frame_Lratios[ff+1] = LsLx
        Ls_frame[ff+1]=Ls
    LsLx_ratios+=frame_Lratios
    Ls_all+=Ls_frame
    
Ls_all = Ls_all/len(v1)
LsLx_ratios = LsLx_ratios/(len(v1))
filter_sizes = np.array([0,2,4,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,240,280,320,380,
                         440,500,600,700,800,900,1000])+1
    
#%% autocorrelations -- first calculate means for each view


v_lists = [v1_list,v2_list,v3_list,v4_list,v5_list,v6_list,v7_list,v8_list]
r_adjs = [- (1600-1532),- (1600-1450),
          + (3*.0254)*v3_cal - 799,+ (3*.0254)*v4_cal - 783,+ (4.5*.0254)*v5_cal - 1002,
          + (4.5*.0254)*v6_cal - 967,- (1600-1260) + 4.6*0.0254*v7_cal,-(1600-1274) + 4.6*0.0254*v8_cal]

z_adjs = [v1_range[0],v2_range[0],v3_range[0],v4_range[0],v5_range[0],v6_range[0],v7_range[0],v8_range[0]]

cals = [v1_cal,v2_cal,v3_cal,v4_cal,v5_cal,v6_cal,v7_cal,v8_cal]

bins = 100

all_ue_means = []

for vvs,vs in enumerate(v_lists):
    
    for vi,v in enumerate(vs):
        v1 = np.load(v,allow_pickle=True)
        ue_mean = []
        for f in np.arange(0,len(v1)):
            uf = v1[f][:, 2:4]
            up = v1[f][:, 4:6]
            n = v1[f][:, 6:8]
            ufmag = -np.sum(uf*n, axis=1)
            upmag = -np.sum(up*n, axis=1)
            ue = ufmag/3 - upmag
            ue = ue[::-1]
            ue_mean.extend(ue)
            
        ue_mean = np.nanmean(ue_mean)
        
        all_ue_means.append(ue_mean)
        
#%% calculate fluctuating components & do autocorrelation
from scipy.interpolate import interp1d

def autocorr(x):

    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def autocorrelation2 (x) :

    xp = x
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:]/np.sum(xp**2)
def autocorrelation (x) :

    xp = x - x.mean()
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:]/np.sum(xp**2)
def acf(x):

    length = len(x)
    return(np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)]))
       
import warnings
warnings.filterwarnings('ignore')   
for vvs,vs in enumerate(v_lists):
    
    for vi,v in enumerate(vs):

        v1 = np.load(v,allow_pickle=True)
        ac_bins = np.logspace(0,3.5,100)

        all_acs = []
        all_ac1 = np.zeros(len(ac_bins)-1,)
        all_ac2 = np.zeros(len(ac_bins)-1,)
        all_ac3 = np.zeros(len(ac_bins)-1,)
        all_ac1_c = np.zeros(len(ac_bins)-1,)
        all_ac2_c = np.zeros(len(ac_bins)-1,)
        all_ac3_c = np.zeros(len(ac_bins)-1,)
        s_lens = []
        for f in np.arange(0,len(v1)):
            pts = v1[f][:,0:2][::-1]
            s_lens.append(len(pts))
            """if len(pts)<3200:
                pass
            else:
                pts = pts[0:3200,:]"""
            s = np.linalg.norm(np.diff(pts,axis=0),axis=1)
            uf = v1[f][:, 2:4]
            up = v1[f][:, 4:6]
            n = v1[f][:, 6:8]
            
            z = 2560 - pts[:,0]
            r = pts[:,1] + r_adjs[3]
            k = curvature(z,r,pts)
            o = orientation(n,np.array([-1,0]))
            
            ufmag = -np.sum(uf*n, axis=1)
            upmag = -np.sum(up*n, axis=1)
            ue = ufmag/3 - upmag
            ue = ue[::-1]
            ue_prime = ue - ue_mean
            good_ue = np.where(~np.isnan(ue_prime))[0]
            ue_prime_nonans = ue_prime[~np.isnan(ue_prime)]
            s = np.insert(np.cumsum(s[np.min(good_ue):np.max(good_ue)-1]),0,0)
            interp_f = interp1d(good_ue,ue_prime_nonans)
            ue_prime_interp = interp_f(np.arange(np.min(good_ue),np.max(good_ue)))
            ac1 = autocorrelation2(ue_prime_interp);
            ccuk = np.correlate(ue_prime_interp,k[np.min(good_ue):np.max(good_ue)],mode='valid')
            """
            ac1 = autocorr(ue_prime_interp);"""
            #ac2 = autocorrelation(ue_prime_interp);
            
            #ac3 = acf(ue_prime_interp);
            
            all_ac1_c+=bs(s[0:int(len(s)/2)],~np.isnan(ac1[0:int(len(s)/2)]),statistic=np.nansum,bins=ac_bins).statistic
            all_ac1+=bs(s[0:int(len(s)/2)],ac1[0:int(len(s)/2)],statistic=np.nansum,bins=ac_bins).statistic
            all_acs.append(ac1)
            """all_ac2_c+=bs(s[0:int(len(s)/2)],~np.isnan(ac2[0:int(len(s)/2)]),statistic=np.sum,bins=ac_bins).statistic
            all_ac2+=bs(s[0:int(len(s)/2)],ac2[0:int(len(s)/2)],statistic=np.sum,bins=ac_bins).statistic
            
            all_ac3_c+=bs(s[0:int(len(s)/2)],~np.isnan(ac3[0:int(len(s)/2)]),statistic=np.sum,bins=ac_bins).statistic
            all_ac3+=bs(s[0:int(len(s)/2)],ac3[0:int(len(s)/2)],statistic=np.sum,bins=ac_bins).statistic
            """

ac_bins = (ac_bins+np.diff(ac_bins)[0]/2)[:-1] 

#%% autocorrelation of r_I

v1 = np.load(v,allow_pickle=True)

v2_b_pos = np.zeros((2560,))
v2_b_c = np.zeros((2560,))

for f in np.arange(0,len(v1)):
    pts = v1[f][:,0:2][::-1]
    r = (pts[:,1] + r_adjs[3])
    v2s = bs(pts[:,0],pts[:,1],statistic=np.nansum,bins=2560,range=(0,2560))
    v2_b_pos += v2s.statistic
    v2_b_c += bs(pts[:,0],~np.isnan(pts[:,1]),statistic=np.nansum,bins=2560,range=(0,2560)).statistic

rI = v2_b_pos/v2_b_c


ac_bins = np.logspace(0,3.5,100)

all_ac1 = np.zeros(len(ac_bins)-1,)
all_ac1_c = np.zeros(len(ac_bins)-1,)
billow_A = []
billow_s = []
billow_L = []
for f in np.arange(0,len(v1),5):
    pts = v1[f][2:-2,0:2][::-1]
    bin_n = bs(pts[:,0],pts[:,1],statistic=np.nansum,bins=2560,range=(0,2560)).binnumber - 3
    s = np.linalg.norm(np.diff(pts,axis=0),axis=1)

    s = np.insert(np.cumsum(s),0,0)
    rI_prime = pts[:,1]-rI_filt[bin_n]
    cross_idx = np.where(np.diff(np.sign(rI_prime) >= 0))[0]
    billow_L.extend(np.diff(s[cross_idx]))

    for b in range(0,len(cross_idx)-1):
        idx = np.where(abs(rI_prime[cross_idx[b]:cross_idx[b+1]])==abs(rI_prime[cross_idx[b]:cross_idx[b+1]]).max())[0]
        billow_A.extend(rI_prime[idx])
        billow_s.append(s[cross_idx[1]]-s[cross_idx[0]])
    ac1 = autocorrelation2(rI_prime);
    """
    ac1 = autocorr(ue_prime_interp);
    ac2 = autocorrelation(ue_prime_interp);
    
    ac3 = acf(ue_prime_interp);
    """
    all_ac1_c+=bs(s[0:int(len(s)/2)],~np.isnan(ac1[0:int(len(s)/2)]),statistic=np.sum,bins=ac_bins).statistic
    all_ac1+=bs(s[0:int(len(s)/2)],ac1[0:int(len(s)/2)],statistic=np.sum,bins=ac_bins).statistic
    """
    all_ac2_c+=bs(s[0:int(len(s)/2)],~np.isnan(ac2[0:int(len(s)/2)]),statistic=np.sum,bins=ac_bins).statistic
    all_ac2+=bs(s[0:int(len(s)/2)],ac2[0:int(len(s)/2)],statistic=np.sum,bins=ac_bins).statistic
    
    all_ac3_c+=bs(s[0:int(len(s)/2)],~np.isnan(ac3[0:int(len(s)/2)]),statistic=np.sum,bins=ac_bins).statistic
    all_ac3+=bs(s[0:int(len(s)/2)],ac3[0:int(len(s)/2)],statistic=np.sum,bins=ac_bins).statistic"""


ac_bins = (ac_bins+np.diff(ac_bins)[0]/2)[:-1] 