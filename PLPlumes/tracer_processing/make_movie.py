#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:42:35 2021

@author: apetersen
"""
import numpy as np
from scipy.stats import binned_statistic as bs
import h5py
from PLPlumes.pio import pivio, imgio
from PLPlumes.plume_processing.plume_functions import windowed_average as windowed_average

import time 
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from math import sin,cos

#%%
v1_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View1/*u_e.npy')
v2_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View2/*u_e.npy')
v3_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View3/*u_e.npy')
v4_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View4/*u_e.npy')
v5_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View5/*u_e.npy')
v6_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View6/*u_e.npy')
v7_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View7/*u_e.npy')
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
#%%

v_lists = [v1_list,v2_list,v3_list,v4_list,v5_list,v6_list,v7_list,v8_list]
r_adjs = [- (1600-1532),- (1600-1450),
          + (3*.0254)*v3_cal - 799,+ (3*.0254)*v4_cal - 783,+ (4.5*.0254)*v5_cal - 1002,
          + (4.5*.0254)*v6_cal - 967,- (1600-1260) + 4.6*0.0254*v7_cal,-(1600-1274) + 4.6*0.0254*v8_cal]

z_adjs = [v1_range[0],v2_range[0],v3_range[0],v4_range[0],v5_range[0],v6_range[0],v7_range[0],v8_range[0]]

cals = [v1_cal,v2_cal,v3_cal,v4_cal,v5_cal,v6_cal,v7_cal,v8_cal]
vvs = 3

v = v4_list[0]
v1 = np.load(v,allow_pickle=True)

img = imgio.imgio('/media/apetersen/Backup4/P2_entrainment/View4/raw1.bsub.img')
piv = pivio.pivio('/media/apetersen/Backup4/P2_entrainment/View4/raw1.bsub.tracers.0032.def.msk.piv')
theta=np.deg2rad(270)
rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

for f in np.arange(40,12400,1143):
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
    
    
    ue_vec = n*np.abs(np.vstack((ue[::-1],ue[::-1])).reshape(len(n),2))
    fig,ax = plt.subplots(figsize=(6,6*(2560/1600)))
    x = (1600-np.arange(0,1600)+r_adjs[vvs])/cals[vvs]/1.905e-2
    y = ((2560-np.arange(0,2560))/cals[vvs]+z_adjs[vvs])/1.905e-2
    X = (1600-np.linspace(0,1600,99)+r_adjs[vvs])/cals[vvs]/1.905e-2
    Y = ((2560-np.linspace(0,2560,159))/cals[vvs]+z_adjs[vvs])/1.905e-2
    ax.pcolormesh(x,y,
                  (np.rot90(img.read_frame2d(f),3)),cmap='gray',shading='auto')
    ax.plot(r/1.905e-2,z/1.905e-2,color='lime',linestyle='-',lw=2)
    """ax.quiver(r[::6]/1.905e-2,z[::6]/1.905e-2,-np.dot(rot,n.transpose())[0,:][::-1][::6],
                             -np.dot(rot,n.transpose())[1,:][::-1][::6],scale=20,color='r',
                             headwidth=3,width=.005);"""
    """ax.quiver(r[::6]/1.905e-2,z[::6]/1.905e-2,-np.dot(rot,ue_vec.transpose())[0,:][::-1][::6],
                             -np.dot(rot,ue_vec.transpose())[1,:][::-1][::6],scale=1,color='y',
                             headwidth=3,width=.005);"""
    piv1 = -np.rot90(piv.read_frame2d(f)[2]*piv.read_frame2d(f)[0],3)
    piv2 = np.rot90(piv.read_frame2d(f)[1]*piv.read_frame2d(f)[0],3)
    piv1[piv1==0]=np.nan
    piv2[piv2==0]=np.nan
    ax.quiver((1600-np.linspace(0,1600,99)+r_adjs[vvs])/cals[vvs]/1.905e-2,
              ((2560-np.linspace(0,2560,159))/cals[vvs]+z_adjs[vvs])/1.905e-2,
              piv1,
              piv2,
              scale=200,color='mediumblue',headwidth=3,width=.005);
    ax.yaxis.tick_right()
    plt.gca().invert_yaxis();plt.gca().invert_xaxis();
    ax.set_ylabel(r'$z/D_0$',fontsize=20,labelpad=15);
    ax.set_xlabel(r'$r/D_0$',fontsize=20,labelpad=15);
    ax.axis('equal');
    pts2 = v1[f+3][:,0:2][::-1]
    z2 = (2560 - pts2[:,0])/cals[vvs] + z_adjs[vvs]
    r2 = (pts2[:,1] + r_adjs[vvs])/cals[vvs]
    ax.plot(r2/1.905e-2,z2/1.905e-2,color='violet',ls='-',lw=2)
    #fig.savefig('/media/apetersen/Backup4/P2_entrainment/interface_movie/frame_%06d.png' %f,dpi=300,format='png',bbox_inches='tight')
    plt.close();
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    