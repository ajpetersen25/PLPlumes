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
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import os
#from matplotlib.ticker import *
#import matplotlib.gridspec as gridspec
#import matplotlib.patches as patches
#from matplotlib.lines import Line2D
#from matplotlib.font_manager import FontProperties
#from mpl_toolkits.axes_grid1 import make_axes_locatable
rc('text', usetex=True)
from scipy.stats import binned_statistic
from PLPlumes.plume_processing.plume_functions import windowed_average, plume_outline
from PLPlumes import cmap_sunset
from scipy.integrate import simps
#%%
v1_cal = 6.64e-5
v2_cal = 6.76e-5
v3_cal = 7.689e-5
v4_cal = 7.75e-5
v5_cal = 7.52e-5
v6_cal = 7.587e-5
v7_cal = 7.86e-5
v8_cal = 7.96e-5
v1_range = (1.25/39.37,7.9375/39.37)
v2_range = (.201,.373)
v3_range = (.357,.553)
v4_range = (.549,.746)
v5_range = (29.25/39.37,34/39.37)
v6_range = (34.3125/39.37,39.125/39.37)
v7_range = (39.25/39.37,44.25/39.37)
v8_range = (44.3125/39.37, 49.3125/39.37)
deltat = 1/(300) #s

upper_cal = 1/3475.93 #m/pix
lower_cal = 1/3461.77 #m/pix
upper_angle = .00809 # rad
R = np.array([[np.cos(upper_angle),-np.sin(upper_angle)],[np.sin(upper_angle), np.cos(upper_angle)]])
deltat_w = 1/(600) #s
D0 = 1.905e-2

outlet = 130
overlap = 0.6731 - 0.6270625 #m
#%% load in tracer imgs & piv 
#%% Plume bidn32 files
upper_cal = 1/3475.93 #m/pix
lower_cal = 1/3461.77 #m/pix
upper_angle = .00809 # rad
R = np.array([[np.cos(upper_angle),-np.sin(upper_angle)],[np.sin(upper_angle), np.cos(upper_angle)]])
deltat_w = 1/(600) #s
D0 = 1.905e-2

outlet = 130
overlap = 0.6731 - 0.6270625 #m
pivbidn32_upper = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/upper/bidn32_upper.0048.def.msk.ave.piv')
pivbidn32_upper2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/upper/bidn32_upper2.0048.def.msk.ave.piv')
pivbidn32_lower = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/lower/bidn32_lower.0048.def.msk.ave.piv')
pivbidn32_lower2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/lower/bidn32_lower2.0048.def.msk.ave.piv')

bidn32_upper_piv1 = pivbidn32_upper.read_frame2d(0)
bidn32_upper_piv2 = pivbidn32_upper2.read_frame2d(0)
zpiv1 = (np.arange(0,bidn32_upper_piv1[0].shape[1])*pivbidn32_upper.dx-outlet)*upper_cal/D0
bidn32_lower_piv1 = pivbidn32_lower.read_frame2d(0)
bidn32_lower_piv2 = pivbidn32_lower2.read_frame2d(0)
zpiv2 = ((np.linspace(2560,2*2560,bidn32_lower_piv1[1].shape[1])-outlet)*lower_cal-overlap)/D0


tau_p = 7.4e-3
w0 = tau_p*9.81

bidn32_centerline1 = np.zeros(np.arange(0,bidn32_upper_piv1[1].shape[1]).shape).astype('int')
W1 = ((bidn32_upper_piv1[1]*R[0,0] - bidn32_upper_piv1[2]*R[0,1]) +(bidn32_upper_piv2[1]*R[0,0] - bidn32_upper_piv2[2]*R[0,1]))/2
for p in range(0,bidn32_upper_piv1[1].shape[1]):
    w_prof = W1[:,p]
    bidn32_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
Wc_bidn32a = windowed_average(np.mean(W1[20:22,np.where(zpiv1>3)[0].astype('int')],axis=0)[:-1],10)*upper_cal/deltat_w

bidn32_centerline2 = np.zeros(np.arange(0,bidn32_upper_piv1[1].shape[1]).shape).astype('int')
W2 = (bidn32_lower_piv1[1]+ bidn32_lower_piv2[1])/2
for p in range(0,bidn32_lower_piv1[1].shape[1]):
    w_prof = W2[:,p]
    bidn32_centerline2[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
Wc_bidn32b = windowed_average(W2[[r for r in bidn32_centerline2],[c for c in range(0,pivbidn32_lower2.nx)]][1:-1],10)*lower_cal/deltat_w

Wc_bidn32 = np.hstack((Wc_bidn32a,Wc_bidn32b[4:]))
zpiv = np.hstack((zpiv1[zpiv1>3][:-1],zpiv2[4:][1:-1]))
#%% Plume dn32 files
upper_cal = 1/3475.93 #m/pix
lower_cal = 1/3461.77 #m/pix
upper_angle = .00809 # rad
R = np.array([[np.cos(upper_angle),-np.sin(upper_angle)],[np.sin(upper_angle), np.cos(upper_angle)]])
deltat_w = 1/(600) #s
D0 = 1.905e-2

outlet = 130
overlap = 0.6731 - 0.6270625 #m
pivdn32_upper = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper.0048.def.msk.ave.piv')
pivdn32_upper2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper2.0048.def.msk.ave.piv')
pivdn32_lower = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower.0048.def.msk.ave.piv')
pivdn32_lower2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower2.0048.def.msk.ave.piv')

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
upper_cal = 1/3475.93 #m/pix
lower_cal = 1/3461.77 #m/pix
upper_angle = .00809 # rad
R = np.array([[np.cos(upper_angle),-np.sin(upper_angle)],[np.sin(upper_angle), np.cos(upper_angle)]])
deltat_w = 1/(600) #s
D0 = 1.905e-2

outlet = 130
overlap = 0.6731 - 0.6270625 #m
pivdn45_upper = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper.0048.def.msk.ave.piv')
pivdn45_upper2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper2.0048.def.msk.ave.piv')
pivdn45_lower = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower.0048.def.msk.ave.piv')
pivdn45_lower2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower2.0048.def.msk.ave.piv')

dn45_upper_piv1 = pivdn45_upper.read_frame2d(0)
dn45_upper_piv2 = pivdn45_upper2.read_frame2d(0)
zpiv1a = (np.arange(0,dn45_upper_piv1[0].shape[1])*pivdn45_upper.dx-outlet)*upper_cal/D0
dn45_lower_piv1 = pivdn45_lower.read_frame2d(0)
dn45_lower_piv2 = pivdn45_lower2.read_frame2d(0)
zpiv2 = ((np.linspace(2560,2*2560,dn45_lower_piv1[1].shape[1])-outlet)*lower_cal-overlap)/D0


tau_p = 7.4e-3
w0 = tau_p*9.81

dn45_centerline1 = np.zeros(np.arange(0,dn45_upper_piv1[1].shape[1]).shape).astype('int')
W1 = ((dn45_upper_piv2[1]*R[0,0] - dn45_upper_piv2[2]*R[0,1]))# +(dn45_upper_piv1[1]*R[0,0] - dn45_upper_piv1[2]*R[0,1]))/2
for p in range(0,dn45_upper_piv1[1].shape[1]):
    w_prof = W1[:,p]
    dn45_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
Wc_dn45a = windowed_average(W1[[r for r in dn45_centerline1[np.where(zpiv1a>6)[0][0]:]],
                            [c for c in range(np.where(zpiv1a>6)[0][0],pivdn45_upper2.nx)]][1:-1],10)*upper_cal/deltat_w

dn45_centerline2 = np.zeros(np.arange(0,dn45_lower_piv1[1].shape[1]).shape).astype('int')
W = (dn45_lower_piv1[1])# + dn45_lower_piv2[1])/2
for p in range(0,dn45_lower_piv1[1].shape[1]):
    w_prof = W[:,p]
    dn45_centerline2[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
    
wc1 = W[[r for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
wc2 = W[[r+1 for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
wc3 = W[[r-1 for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
wc4 = W[[r+2 for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
wc5 = W[[r-2 for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
Wc_dn45b = np.mean([wc1,wc2,wc3],axis=0)*lower_cal/deltat_w

Wc_dn45 = np.hstack((Wc_dn45a,Wc_dn45b[4:]))
zpiv3 = np.hstack((zpiv1a[zpiv1a>6][:-1],zpiv2[4:][1:-1]))
#%% 
def plume_mask(img,piv,threshold,window_threshold,frame,orientation='horz'):
    """
    img_outline = frame>threshold
    contours, hierarchy =   cv2.findContours(img_outline.copy().astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    c_size = ([c.size for c in contours])
    lc = np.where(c_size == np.max(c_size))[0][0]
    plume_contour = cv2.drawContours(image,contours[lc],-1, 255, 1)

    return plume_contour"""

    mask = piv.read_frame2d(frame)[0]
    mask[mask!=1]=0
    step=(piv.dx,piv.dy)
    slices = []
    x = np.arange(piv.dx,img.ix-piv.dx,piv.dx)
    y = np.arange(piv.dx,img.iy-piv.dx,piv.dx)
    slices = []
    for i in x:
       for j in y:
           slices.append((slice(int(j),int(j+piv.dx)),slice(int(i),int(i+piv.dx))))
    for s in slices:
        window = img.read_frame2d(frame)[s]
        if np.sum(window>threshold)/(step[0]*step[1]) > window_threshold and mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] ==1:
            mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] = 1
        else:
            mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)]  = 0
    return(mask)
#%% P1
v1 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/View1/processed/quiescent/entrainment_v1.mean.txt')*v1_cal/deltat
v2 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/View2/processed/quiescent/entrainment_v2.mean.txt')*v2_cal/deltat
v3 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/View3/processed/quiescent/entrainment_v3.mean.txt')*v3_cal/deltat
#v4 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/View4/processed/quiescent/entrainment_v4_mean.txt')*v4_cal/deltat
v5 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/View5/processed/quiescent/entrainment_v5.mean.txt')*v5_cal/deltat
v6 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/View6/processed/quiescent/entrainment_v6.mean.txt')*v6_cal/deltat
v7 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/View7/processed/quiescent/test_v7.mean.txt')*v7_cal/deltat
v8 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/View8/processed/quiescent/entrainment_v8.mean.txt')*v8_cal/deltat




v4 = np.full_like(v1,np.nan)
z1 = np.linspace(v1_range[0],v1_range[1],2560)
z2 = np.linspace(v2_range[0],v2_range[1],2560)
z3 = np.linspace(v3_range[0],v3_range[1],2560)
z4 = np.linspace(v4_range[0],v4_range[1],2560)
z5 = np.linspace(v5_range[0],v5_range[1],1600)
z6 = np.linspace(v6_range[0],v6_range[1],1600)
z7 = np.linspace(v7_range[0],v7_range[1],1600)
z8 = np.linspace(v8_range[0],v8_range[1],1600)

cs = plt.get_cmap('inferno')
zs = np.hstack((z1,z2,z3,z4,z5,z6,z7,z8))
entrainment_bidn32 = np.hstack((v1,v2,v3,v4,v5,v6,v7,v8))
zs = zs[~np.isnan(entrainment_bidn32)]
entrainment_bidn32 = entrainment_bidn32[~np.isnan(entrainment_bidn32)]
entrainment_bidn32_mean = windowed_average(entrainment_bidn32,100)
f,ax = plt.subplots(1,3,figsize=(15,7),sharey=True)
ax[0].plot(entrainment_bidn32/.695,zs/D0,'.',color='0.5')
#ax[0].plot(entrainment_bidn32_mean/.695,zs/D0,'-',color=cs(0),linewidth=2)
ax[0].set_ylim(65,0);

ax[0].xaxis.set_label_position('top')
#xlab = ax[0].set_xlabel('$u_e$ [$m/s$]',fontsize=20,labelpad=15)
ylab = ax[0].set_ylabel('$z/D_0$',fontsize=20,labelpad=15)
ax[0].tick_params(axis='both',which='major',labelsize=18);
ax[0].xaxis.tick_top()
#f.savefig('/home/ajp25/Desktop/u_e.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')

bins = np.hstack((zpiv[0]-np.diff(zpiv[0:2]), zpiv+np.hstack((np.diff(zpiv), np.diff(zpiv)[-1]))))
entrainment_bidn32_binned = binned_statistic(zs[~np.isnan(entrainment_bidn32)]/D0,entrainment_bidn32[~np.isnan(entrainment_bidn32)],statistic=np.nanmean,bins=bins)

alpha = abs(entrainment_bidn32_binned.statistic)/((Wc_bidn32-w0)/2)
f2,ax2 = plt.subplots(figsize=(5,7))
ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax2.plot(alpha,zpiv,'o',color=cs(0))
ax2.set_ylim(65,0);
ax2.xaxis.set_label_position('top')
xlab = ax2.set_xlabel(r'$\alpha = u_e/(W_c - \tau_p g)$',fontsize=20,labelpad=15)
ylab = ax2.set_ylabel('$z/D_0$',fontsize=20,labelpad=15)
ax2.tick_params(axis='both',which='major',labelsize=18);
ax2.xaxis.tick_top()
#f.savefig('/home/ajp25/Desktop/alpha.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='xaxis')

#%% P2
v1 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/entrainment_v1.mean.txt')*v1_cal/deltat
v2 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/bad_entrainment/entrainment_v2.mean.txt')[1000:]*v2_cal/deltat
v3 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View3/processed/quiescent/entrainment_v3.mean.txt')*v3_cal/deltat
v4 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/entrainment_v4_mean.txt')*v4_cal/deltat
v5 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View5/processed/quiescent/entrainment_v5.mean.txt')*v5_cal/deltat
v6 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View6/processed/quiescent/entrainment_v6.mean.txt')*v6_cal/deltat
v7 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View7/processed/quiescent/entrainment_v7.mean.txt')*v7_cal/deltat
v8 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View8/processed/quiescent/entrainment_v8.mean.txt')*v8_cal/deltat

z1 = np.linspace(v1_range[0],v1_range[1],2560)
z2 = np.linspace(2560+1000,2560*2,2560-1000)*v2_cal
z3 = np.linspace(v3_range[0],v3_range[1],2560)
z4 = np.linspace(v4_range[0],v4_range[1],2560)
z5 = np.linspace(v5_range[0],v5_range[1],1600)
z6 = np.linspace(v6_range[0],v6_range[1],1600)
z7 = np.linspace(v7_range[0],v7_range[1],1600)
z8 = np.linspace(v8_range[0],v8_range[1],1600)

cs = plt.get_cmap('inferno')
zs = np.hstack((z1,z2,z3,z4,z5,z6,z7,z8))
entrainment_dn32 = np.hstack((v1,v2,v3,v4,v5,v6,v7,v8))
zs = zs[~np.isnan(entrainment_dn32)]
entrainment_dn32 = entrainment_dn32[~np.isnan(entrainment_dn32)]
entrainment_dn32_mean = windowed_average(entrainment_dn32,200)
zs = zs[~np.isnan(entrainment_dn32)]
ax[1].plot(entrainment_dn32/.155,zs/D0,'.',color='0.5')
#ax[1].plot(entrainment_dn32_mean/.155,zs/D0,'-',color=cs(0.392),linewidth=2)
ax[1].set_ylim(65,0);
ax[1].xaxis.set_label_position('top')
xlab = ax[1].set_xlabel(r'$u_e/W_0$',fontsize=30,labelpad=15)
#ylab = ax.set_ylabel('$z/D_0$',fontsize=20,labelpad=15)
ax[1].tick_params(axis='both',which='major',labelsize=18);
ax[1].xaxis.tick_top()
#f.savefig('/home/ajp25/Desktop/u_e.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')

bins = np.hstack((zpiv[0]-np.diff(zpiv[0:2]), zpiv+np.hstack((np.diff(zpiv), np.diff(zpiv)[-1]))))
entrainment_dn32_binned = binned_statistic(zs[~np.isnan(entrainment_dn32)]/D0,entrainment_dn32[~np.isnan(entrainment_dn32)],statistic=np.nanmean,bins=bins)

alpha = (-entrainment_dn32_binned.statistic)/((np.hstack((w1_p1,w2_p1[6:]))-w0)/2)#((Wc_dn32-w0)/2)
ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax2.plot(alpha,zpiv,'o',color=cs(0.392))
ax2.set_ylim(65,0);
ax2.xaxis.set_label_position('top')
xlab = ax2.set_xlabel(r'$\alpha = u_e/(W_c/2 - \tau_p g)$',fontsize=20,labelpad=15)
ylab = ax2.set_ylabel('$z/D_0$',fontsize=20,labelpad=15)
ax2.tick_params(axis='both',which='major',labelsize=18);
ax2.xaxis.tick_top()
#f.savefig('/home/ajp25/Desktop/alpha.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='xaxis')
#%% P3
v1 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/View1/processed/quiescent/entrainment_v1.mean.txt')*v1_cal/deltat
v2 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/View2/processed/quiescent/entrainment_v2.mean.txt')*v2_cal/deltat
v3 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/View3/processed/quiescent/entrainment_v3.mean.txt')*v3_cal/deltat/1.2
v4 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/View4/processed/quiescent/test_v4.mean.txt')*v4_cal/deltat/1.3
v5 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/View5/processed/quiescent/entrainment_v5.mean.txt')*v5_cal/deltat
v6 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/View6/processed/quiescent/entrainment_v6.mean.txt')*v6_cal/deltat
v7 = np.loadtxt('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/View7/processed/quiescent/entrainment_v7.mean.txt')*v7_cal/deltat
v8 = np.full_like(v7,np.nan)

z1 = np.linspace(v1_range[0],v1_range[1],2560)
z2 = np.linspace(v2_range[0],v2_range[1],2560)
z3 = np.linspace(v3_range[0],v3_range[1],2560)
z4 = np.linspace(v4_range[0],v4_range[1],2560)
z5 = np.linspace(v5_range[0],v5_range[1],1600)
z6 = np.linspace(v6_range[0],v6_range[1],1600)
z7 = np.linspace(v7_range[0],v7_range[1],1600)
z8 = np.linspace(v8_range[0],v8_range[1],1600)

cs = plt.get_cmap('inferno')
zs = np.hstack((z1,z2,z3,z4,z5,z6,z7,z8))

entrainment_dn45 = np.hstack((v1,v2,v3,v4,v5,v6,v7,v8))
zs = zs[~np.isnan(entrainment_dn45)]
entrainment_dn45 = entrainment_dn45[~np.isnan(entrainment_dn45)]
entrainment_dn45_mean = windowed_average(entrainment_dn45,100)
ax[2].plot(entrainment_dn45/.22,zs/D0,'.',color='0.5')
#ax[2].plot(entrainment_dn45_mean/.22,zs/D0,'-',color=cs(0.86),linewidth=2)
ax[2].set_ylim(65,0);
ax[0].set_xlim(-.07,0);
ax[1].set_xlim(-.4,-.1);
ax[2].set_xlim(-.3,-.05);

ax[2].xaxis.set_label_position('top')
xlab = ax[1].set_xlabel(r'$u_e/W_0$',fontsize=30,labelpad=15)
ylab = ax[0].set_ylabel('$z/D_0$',fontsize=30,labelpad=15)

ax[2].tick_params(axis='both',which='major',labelsize=18);
ax[2].xaxis.tick_top()
#t1 = ax[0].text(-.38,70,r'$P_1$',fontsize=30)
#t2 = ax[1].text(-.6,70,r'$P_2$',fontsize=30)
#t3 = ax[2].text(-.575,70,r'$P_3$',fontsize=30)

f.savefig('/home/ajp25/Desktop/u_e.png',dpi=1200,format='png',transparent=True,bbox_extra_artists=[xlab,ylab], bbox_inches='tight')

bins = np.hstack((zpiv3[0]-np.diff(zpiv3[0:2]), zpiv3+np.hstack((np.diff(zpiv3), np.diff(zpiv3)[-1]))))
entrainment_dn45_binned = binned_statistic(zs[~np.isnan(entrainment_dn45)]/D0,entrainment_dn45[~np.isnan(entrainment_dn45)],statistic=np.nanmean,bins=bins)

alpha = (-entrainment_dn45_binned.statistic[:-1])/((Wc_dn45-w0)/2)
ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
ax2.plot(alpha,zpiv3[:-1],'o',color=cs(0.86))
ax2.set_ylim(65,0);
ax2.xaxis.set_label_position('top')
xlab = ax2.set_xlabel(r'$\alpha = u_e/w_{f,avg}$',fontsize=20,labelpad=15)
ylab = ax2.set_ylabel('$z/D_0$',fontsize=20,labelpad=15)
ax2.tick_params(axis='both',which='major',labelsize=18);
ax2.xaxis.tick_top()
leg = ax2.legend([r'$P1$', '$P2$','$P3$'],fontsize=20,framealpha=0.5,fancybox=True,ncol=3,loc='upper center',bbox_to_anchor=(0.5,-0.01))
#f2.savefig('/home/ajp25/Desktop/alphas.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab,leg], bbox_inches='tight')
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='xaxis')
#%% entrainment plots
def plume_outline(frame,dilation_size,dilation_iterations,threshold,med_filt_size):
    """
    img_outline = frame>threshold
    contours, hierarchy =   cv2.findContours(img_outline.copy().astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    c_size = ([c.size for c in contours])
    lc = np.where(c_size == np.max(c_size))[0][0]
    plume_contour = cv2.drawContours(image,contours[lc],-1, 255, 1)

    return plume_contour"""
    frame_d = frame.copy()
    i=0
    kernel = square(dilation_size)
    while i<dilation_iterations:
        frame_d = dilation(frame_d,kernel)
        i+=1
    frame_db = frame_d > threshold
    frame_mask = median_filter(frame_db,med_filt_size)
    return(frame_mask)

v1 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/tracers_dn32_v1.t.bsub.0032.def.piv')
v2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.t.bsub.0032.def.piv')
v3 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View3/processed/quiescent/tracers_dn32_v3.tracers.bsub.0032.def.piv')
v4 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/tracers_dn32_v4.tracers.bsub.0032.def.piv')
v5 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View5/processed/quiescent/tracers_dn32_v5.tracers.bsub.0032.def.piv')
v6 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View6/processed/quiescent/tracers_dn32_v6.tracers.bsub.0032.def.piv')
v7 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View7/processed/quiescent/tracers_dn32_v7.tracers.bsub.0032.def.piv')
v8 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View8/processed/quiescent/tracers_dn32_v8.tracers.bsub.0032.def.piv')

v1img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/tracers_dn32_v1.img')
v2img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.img')
v3img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View3/processed/quiescent/tracers_dn32_v3.img')
v4img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/tracers_dn32_v4.img')
v5img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View5/processed/quiescent/tracers_dn32_v5.img')
v6img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View6/processed/quiescent/tracers_dn32_v6.img')
v7img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View7/processed/quiescent/tracers_dn32_v7.img')
v8img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View8/processed/quiescent/tracers_dn32_v8.img')

z1 = np.linspace(v1_range[0],v1_range[1],2560)
z2 = np.linspace(2560+1000,2560*2,2560-1000)*v2_cal
z3 = np.linspace(v3_range[0],v3_range[1],2560)
z4 = np.linspace(v4_range[0],v4_range[1],2560)
z5 = np.linspace(v5_range[0],v5_range[1],1600)
z6 = np.linspace(v6_range[0],v6_range[1],1600)
z7 = np.linspace(v7_range[0],v7_range[1],1600)
z8 = np.linspace(v8_range[0],v8_range[1],1600)


#%%
import time
import numpy.ma as nma
piv = v1
threshold = 1
window_threshold = 0.3
img = v1img
step = (piv.dx,piv.dy)
x = np.arange(piv.dx,img.ix-piv.dx,piv.dx)
y = np.arange(piv.dy,img.iy-piv.dy,piv.dy)
slices = []
for i in x:
   for j in y:
       slices.append((slice(int(j),int(j+piv.dy)),slice(int(i),int(i+piv.dx))))
           
tic = time.time()
masks = []
u = []
v = []
for f in range(0,piv.nt):
    piv_frame = piv.read_frame2d(f)
    img_frame = img.read_frame2d(f)
    frame_mask = plume_outline(img_frame,40,1,3200,100)
    mask = piv.read_frame2d(f)[0]
    mask[mask==4]=1
    mask[mask!=1]=0
    # loop through all interrogation windows, looking at which meet threshold criteria
    # elements in the frame mask array which meet this criteria are set to 0
    for s in slices:
            window = img_frame[s]
            if np.sum(window==threshold)/(step[0]*step[1]) < window_threshold and mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] ==1:
                mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)] = 1
            else:
                mask[int((s[0].start+piv.dy/2)/(piv.dy)-1),int((s[1].start+piv.dx/2)/(piv.dx)-1)]  = 0
    u.append(piv_frame[1])
    v.append(piv_frame[2])
    masks.append(mask)
    
masks = np.array(masks).astype('bool')
masked_u = nma.masked_array(u,mask=~masks)
masked_v = nma.masked_array(v,mask=~masks)          
u_m_avg = nma.mean(masked_u,axis=0)
v_m_avg = nma.mean(masked_v,axis=0)
print('Time elapsed: %0.3f seconds' %(time.time()-tic))
#%% Entrainment calculated by mean tracer velocity field

v1 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/tracers_dn32_v1.tracers.bsub.0032.def.msk.piv')
v2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.tracers.bsub.0032.def.msk.piv')
v3 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View3/processed/quiescent/tracers_dn32_v3.tracers.bsub.0032.def.msk.piv')
v4 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/tracers_dn32_v4.tracers.bsub.0032.def.msk.piv')
v5 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View5/processed/quiescent/tracers_dn32_v5.tracers.bsub.0032.def.msk.piv')
v6 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View6/processed/quiescent/tracers_dn32_v6.tracers.bsub.0032.def.msk.piv')
v7 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View7/processed/quiescent/tracers_dn32_v7.tracers.bsub.0032.def.msk.piv')
v8 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View8/processed/quiescent/tracers_dn32_v8.tracers.bsub.0032.def.msk.piv')

v1_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/tracers_dn32_v1.tracers.bsub.0032.def.msk.ave.piv')
v2_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.tracers.bsub.0032.def.msk.ave.piv')
v3_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View3/processed/quiescent/tracers_dn32_v3.tracers.bsub.0032.def.msk.ave.piv')
v4_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/tracers_dn32_v4.tracers.bsub.0032.def.msk.ave.piv')
v5_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View5/processed/quiescent/tracers_dn32_v5.tracers.bsub.0032.def.msk.ave.piv')
v6_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View6/processed/quiescent/tracers_dn32_v6.tracers.bsub.0032.def.msk.ave.piv')
v7_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View7/processed/quiescent/tracers_dn32_v7.tracers.bsub.0032.def.msk.ave.piv')
v8_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View8/processed/quiescent/tracers_dn32_v8.tracers.bsub.0032.def.msk.ave.piv')

v1_ave.cal = v1_cal
v2_ave.cal = v2_cal
v3_ave.cal = v3_cal
v4_ave.cal = v4_cal
v5_ave.cal = v5_cal
v6_ave.cal = v6_cal
v7_ave.cal = v7_cal
v8_ave.cal = v8_cal

v1.cal = v1_cal
v2.cal = v2_cal
v3.cal = v3_cal
v4.cal = v4_cal
v5.cal = v5_cal
v6.cal = v6_cal
v7.cal = v7_cal
v8.cal = v8_cal

z1_piv = np.linspace(v1_range[0],v1_range[1],v1.nx)
z2_piv = np.linspace(v2_range[0],v2_range[1],v2.nx)
z3_piv = np.linspace(v3_range[0],v3_range[1],v3.nx)
z4_piv = np.linspace(v4_range[0],v4_range[1],v4.nx)
z5_piv = np.linspace(v5_range[0],v5_range[1],v5.ny)
z6_piv = np.linspace(v6_range[0],v6_range[1],v6.ny)
z7_piv = np.linspace(v7_range[0],v7_range[1],v7.ny)
z8_piv = np.linspace(v8_range[0],v8_range[1],v8.ny)
#%% percentage of vectors used in average maps:
v1_num = np.zeros(v1.read_frame2d(0)[0].shape)
v2_num = np.zeros(v2.read_frame2d(0)[0].shape)
v3_num = np.zeros(v3.read_frame2d(0)[0].shape)
v4_num = np.zeros(v4.read_frame2d(0)[0].shape)
v5_num = np.zeros(v5.read_frame2d(0)[0].shape)
v6_num = np.zeros(v6.read_frame2d(0)[0].shape)
v7_num = np.zeros(v7.read_frame2d(0)[0].shape)
v8_num = np.zeros(v8.read_frame2d(0)[0].shape)

for f in range(0,v1.nt):
    v1_num += v1.read_frame2d(f)[0]
    v2_num += v2.read_frame2d(f)[0]
    v3_num += v3.read_frame2d(f)[0]
    v4_num += v4.read_frame2d(f)[0]
    v5_num += v5.read_frame2d(f)[0]
    v6_num += v6.read_frame2d(f)[0]


for f in range(0,v8.nt):
    v7_num += v7.read_frame2d(f)[0]
    v8_num += v8.read_frame2d(f)[0]

np.savetxt(os.path.dirname(v1.file_name)+'/v1_frac.txt',v1_num/v1.nt)
np.savetxt(os.path.dirname(v2.file_name)+'/v2_frac.txt',v2_num/v2.nt)
np.savetxt(os.path.dirname(v3.file_name)+'/v3_frac.txt',v3_num/v3.nt)
np.savetxt(os.path.dirname(v4.file_name)+'/v4_frac.txt',v4_num/v4.nt)
np.savetxt(os.path.dirname(v5.file_name)+'/v5_frac.txt',v5_num/v5.nt)
np.savetxt(os.path.dirname(v6.file_name)+'/v6_frac.txt',v6_num/v6.nt)
np.savetxt(os.path.dirname(v7.file_name)+'/v7_frac.txt',v7_num/v7.nt)
np.savetxt(os.path.dirname(v8.file_name)+'/v8_frac.txt',v8_num/v8.nt)

#%% load in
v1_frac = np.loadtxt(os.path.dirname(v1.file_name)+'/v1_frac.txt')
v2_frac = np.loadtxt(os.path.dirname(v2.file_name)+'/v2_frac.txt')
v3_frac = np.loadtxt(os.path.dirname(v3.file_name)+'/v3_frac.txt')
v4_frac = np.loadtxt(os.path.dirname(v4.file_name)+'/v4_frac.txt')
v5_frac = np.loadtxt(os.path.dirname(v5.file_name)+'/v5_frac.txt')
v6_frac = np.loadtxt(os.path.dirname(v6.file_name)+'/v6_frac.txt')
v7_frac = np.loadtxt(os.path.dirname(v7.file_name)+'/v7_frac.txt')
v8_frac = np.loadtxt(os.path.dirname(v8.file_name)+'/v8_frac.txt')


#%% radial zero-position
v1_0 = 70
v2_0 = 76
v3_0 = 87 #85
v4_0 = 88 #

v5_0 = 36
v6_0 = 36
v7_0 = 33
v8_0 = 33

#%% global mass flux
v1_ave.mask = (v1_frac > 0.6)
v2_ave.mask = (v2_frac > 0.6)
v3_ave.mask = (v3_frac > 0.6)
v4_ave.mask = (v4_frac > 0.6)
v5_ave.mask = (v5_frac > 0.6)
v6_ave.mask = (v6_frac > 0.6)
v7_ave.mask = (v7_frac > 0.6)
v8_ave.mask = (v8_frac > 0.6)
v1_ave.center = v1_0
v2_ave.center = v2_0
v3_ave.center = v3_0
v4_ave.center = v4_0
v5_ave.center = v5_0
v6_ave.center = v6_0
v7_ave.center = v7_0
v8_ave.center = v8_0
first = [v1_ave,v2_ave,v3_ave,v4_ave]
last = [v5_ave,v6_ave,v7_ave,v8_ave]
mfdot = []
for v in first:
    r = -(np.arange(0,v.ny) - v.center)*16*v.cal
    ue = v.read_frame2d(0)[2]*v.cal/deltat *v.mask
    ue[ue==0]=np.nan
    for col in range(0,v.nx-1):
        mfdot.append(2*np.pi*1.225*simps(ue[:,col][~np.isnan(ue[:,col])]*r[~np.isnan(ue[:,col])]))
for v in last:
    r = -(np.arange(0,v.nx) - v.center)*16*v.cal
    ue = v.read_frame2d(0)[1]*v.cal/deltat *v.mask
    for row in range(0,v.ny-1):
        mfdot.append(2*np.pi*1.225*simps(ue[row,:][~np.isnan(ue[row,:])]*r[~np.isnan(ue[row,:])]))
        
#%% TNTI radial position PDFs
v1.center = v1_0
v2.center = v2_0
v3.center = v3_0
v4.center = v4_0
v5.center = v5_0
v6.center = v6_0
v7.center = v7_0
v8.center = v8_0
first = [v1,v2,v3,v4]
last = [v5,v6,v7,v8]
for v in first: 
    v.bs = []
    for f in range(0,v.nt):
        s = v.read_frame2d(f)[0][0:-2,:]
        pix = np.where(np.diff(s,axis=0)==-1)[0]
        b = (v.center - pix)*16*v.cal
        v.bs.extend(b)
#%% radial positions to take horizontal velocity
v1_row = [45,38,30]
v2_row = [30,25,18]
v3_row = [25,20,10]
v4_row = [15,10,5]

v5_col = [125,135,145]
v6_col = [135,142,150]
v7_col = [140,146,155]
v8_col = [145,150,155]
#%% 
z1 = np.linspace(v1_range[0],v1_range[1],v1.nx)
z2 = np.linspace(v2_range[0],v2_range[1],v2.nx)
z3 = np.linspace(v3_range[0],v3_range[1],v3.nx)
z4 = np.linspace(v4_range[0],v4_range[1],v4.nx)
z5 = np.linspace(v5_range[0],v5_range[1],v5.ny)
z6 = np.linspace(v6_range[0],v6_range[1],v6.ny)
z7 = np.linspace(v7_range[0],v7_range[1],v7.ny)
z8 = np.linspace(v8_range[0],v8_range[1],v8.ny)