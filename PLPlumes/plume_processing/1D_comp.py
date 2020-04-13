#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:23:16 2020

@author: ajp25
"""


import numpy as np
from PLPlumes.pio import imgio,pivio

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.lines as mlines
from matplotlib import rc
#from matplotlib.ticker import *
#import matplotlib.gridspec as gridspec
#import matplotlib.patches as patches
#from matplotlib.lines import Line2D
#from matplotlib.font_manager import FontProperties
#from mpl_toolkits.axes_grid1 import make_axes_locatable
rc('text', usetex=True)

from PLPlumes.plume_processing.plume_functions import windowed_average
from PLPlumes import cmap_sunset

from PLPlumes.plume_processing.plume_functions import rho_to_phi, phi_to_rho
from PLPlumes.modeling import pure_plume, lazy_plume
#%%
upper_cal = 1/3475.93 #m/pix
lower_cal = 1/3461.77 #m/pix
upper_angle = .00809 # rad
R = np.array([[np.cos(upper_angle),-np.sin(upper_angle)],[np.sin(upper_angle), np.cos(upper_angle)]])
deltat = 1/(600) #s
D0 = 1.905e-2

outlet = 130
overlap = 0.6731 - 0.6270625 #m
#%% Plume dn32 files
dn32_upper = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper.rho_b.avg.img')
dn32_upper2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper2.rho_b.avg.img')
dn32_lower = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower.rho_b.avg.img')
dn32_lower2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower2.rho_b.avg.img')

pivdn32_upper = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper.0048.def.msk.ave.piv')
pivdn32_upper2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper2.0048.def.msk.ave.piv')
pivdn32_lower = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower.0048.def.msk.ave.piv')
pivdn32_lower2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower2.0048.def.msk.ave.piv')


#%% Plume bi_dn32 files
bidn32_upper = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/upper/bidn32_upper.rho_b.avg.img')
bidn32_upper2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/upper/bidn32_upper2.rho_b.avg.img')
bidn32_lower = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/lower/bidn32_lower.rho_b.avg.img')
bidn32_lower2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/lower/bidn32_lower2.rho_b.avg.img')

pivbidn32_upper = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/upper/bidn32_upper.0048.def.msk.ave.piv')
pivbidn32_upper2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/upper/bidn32_upper2.0048.def.msk.ave.piv')
pivbidn32_lower = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/lower/bidn32_lower.0048.def.msk.ave.piv')
pivbidn32_lower2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/lower/bidn32_lower2.0048.def.msk.ave.piv')

#%% Plume dn45 files
dn45_upper = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper.rho_b.avg.img')
dn45_upper2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper2.rho_b.avg.img')
dn45_lower = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower.rho_b.avg.img')
dn45_lower2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower2.rho_b.avg.img')

pivdn45_upper = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper.0048.def.msk.ave.piv')
pivdn45_upper2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper2.0048.def.msk.ave.piv')
pivdn45_lower = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower.0048.def.msk.ave.piv')
pivdn45_lower2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower2.0048.def.msk.ave.piv')

#%% Concentration Centerline Setup
# z arrays
img1 = dn32_upper.read_frame2d(0)
img2 = dn32_upper2.read_frame2d(0)
dn32_upper_img = (img1+img2)/2
z1 = (np.arange(0,2560)-outlet)*upper_cal/D0

img1 = dn32_lower.read_frame2d(0)
img2 = dn32_lower2.read_frame2d(0)
dn32_lower_img = (img1+img2)/2
z2 = ((np.arange(2560,2560*2)-outlet)*lower_cal-overlap)/D0

img1 = bidn32_upper.read_frame2d(0)
img2 = bidn32_upper2.read_frame2d(0)
bidn32_upper_img = (img1+img2)/2

img1 = bidn32_lower.read_frame2d(0)
img2 = bidn32_lower2.read_frame2d(0)
bidn32_lower_img = (img1+img2)/2

img1 = dn45_upper.read_frame2d(0)
img2 = dn45_upper2.read_frame2d(0)
dn45_upper_img = (img1+img2)/2

img1 = dn45_lower.read_frame2d(0)*1.65
img2 = dn45_lower2.read_frame2d(0)*1.65
dn45_lower_img = (img1+img2)/2

#%% modeling
from PLPlumes.modeling import lai_model,liu_model,lazy_model
from PLPlumes.plume_processing.plume_functions import phi_to_rho,rho_to_phi
r0 = 1.905e-2/2
tau_p = 7.4e-3
w0 = tau_p*9.81
z = np.arange(0,20,5e-6)
rho_p = 2500
rho_f = 1.225
dp = 30e-6
mu_f = 1.825e-5

# Lai z,bf_0,bp_0,wf_0,ws_0,phi_v0,dp,rho_p,rho_f,mu_f=1.825e-5
mfr = 1.42e-3 #g/s
rhob_0 = 26
wp_0 = mfr/(rhob_0*np.pi*r0**2)
wf_0 = wp_0-w0
bf_0 = r0
bp_0 = r0
ws_0 = w0
phi_v0 = rho_to_phi(rhob_0,rho_p,rho_f)
lai_bidn32 = lai_model(z,bf_0,bp_0,wf_0,ws_0,phi_v0,dp,rho_p,rho_f,mu_f)

mfr = 3e-3 #g/s
rhob_0 = 55
wp_0 = mfr/(rhob_0*np.pi*r0**2)
wf_0 = wp_0-w0
bf_0 = r0
bp_0 = r0
ws_0 = w0 
phi_v0 = rho_to_phi(rhob_0,rho_p,rho_f)
lai_dn32 = lai_model(z,bf_0,bp_0,wf_0,ws_0,phi_v0,dp,rho_p,rho_f,mu_f)

mfr = 8e-3 #g/s
rhob_0 = 100
wp_0 = mfr/(rhob_0*np.pi*r0**2)
wf_0 = wp_0-w0
bf_0 = r0
bp_0 = r0
ws_0 = w0
phi_v0 = rho_to_phi(rhob_0,rho_p,rho_f)
lai_dn45 = lai_model(z,bf_0,bp_0,wf_0,ws_0,phi_v0,dp,rho_p,rho_f,mu_f)

# Liu w_p0,w_a0,beta_0,rho_p,rho_a,ra_0,rp_0,dp,mu_a,alpha,mfr,z
mfr = 1.42e-3 #g/s
rhob_0 = 26
wp_0 = mfr/(rhob_0*np.pi*r0**2)/2
wf_0 = wp_0-w0/2
bf_0 = r0
alpha = 0.03
liu_bidn32 = liu_model(wp_0,wf_0,rho_to_phi(rhob_0,rho_p,rho_f),rho_p,rho_f,r0,r0,dp,mu_f,alpha,mfr,z)

mfr = 3e-3 #g/s
rhob_0 = 55
wp_0 = mfr/(rhob_0*np.pi*r0**2)/2
wf_0 = wp_0-w0/2
bf_0 = r0
alpha = 0.03
liu_dn32 = liu_model(wp_0,wf_0,rho_to_phi(rhob_0,rho_p,rho_f),rho_p,rho_f,r0,r0,dp,mu_f,alpha,mfr,z)

mfr = 8e-3 #g/s
rhob_0 = 100
wp_0 = mfr/(rhob_0*np.pi*r0**2)/2
wf_0 = wp_0-w0/2
bf_0 = r0
alpha = 0.03
liu_dn45 = liu_model(wp_0,wf_0,rho_to_phi(rhob_0,rho_p,rho_f),rho_p,rho_f,r0,r0,dp,mu_f,alpha,mfr,z)

# lazy single-phase z,r0,rho_a,rho_p,beta_0,wp0,alpha=0.03
mfr = 1.42e-3 #g/s
rhob_0 = 26
wp_0 = mfr/(rhob_0*np.pi*r0**2)
alpha = 0.03
lazy_bidn32 = lazy_model(z,r0,rho_f,rho_p,rho_to_phi(rhob_0,rho_p,rho_f),wp_0,alpha)

mfr = 3e-3 #g/s
rhob_0 = 55
wp_0 = mfr/(rhob_0*np.pi*r0**2)
alpha = 0.03
lazy_dn32 = lazy_model(z,r0,rho_f,rho_p,rho_to_phi(rhob_0,rho_p,rho_f),wp_0,alpha)

mfr = 8e-3 #g/s
rhob_0 = 100
wp_0 = mfr/(rhob_0*np.pi*r0**2)
alpha = 0.03
lazy_dn45 = lazy_model(z,r0,rho_f,rho_p,rho_to_phi(rhob_0,rho_p,rho_f),wp_0,alpha)
#%% Centerline Concentration
cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]/D0))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax  = plt.subplots(figsize=(6,10))
fc,axc = plt.subplots(1,3,figsize=(18,10))
dn32_centerline1 = []
for p in range(500,2560):
    c_prof = dn32_upper_img[:,p]
    dn32_centerline1.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)/1.225,z1[500:],'-',linewidth=4,color=cs(100))
axc.plot(windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)/1.225,z1[500:],'-',linewidth=4,color=cs(100))

dn32_centerline2 = []
for p in range(0,2560):
    c_prof = dn32_lower_img[:,p]
    dn32_centerline2.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(dn32_lower_img[dn32_centerline2,np.arange(0,2560)],100)/1.225,z2[:],'-',linewidth=4,color=cs(100))
axc.plot(windowed_average(dn32_lower_img[dn32_centerline2,np.arange(0,2560)],100)/1.225,z2[:],'-',linewidth=4,color=cs(100))


z = np.linspace(z1[500]*D0,2,10000)
beta0 = rho_to_phi(windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)[0],2500,1.225)
lazy1 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.5)
#ax.plot(lazy1[3]/1.225,z/D0,'--',color=cs(100),linewidth=2)

bidn32_centerline1 = []
for p in range(330,2560):
    c_prof = bidn32_upper_img[:,p]
    bidn32_centerline1.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(bidn32_upper_img[bidn32_centerline1,np.arange(330,2560)],100)/1.225,z1[330:],'-',linewidth=4,color=cs(0))

bidn32_centerline2 = []
for p in range(0,2560):
    c_prof = dn32_lower_img[:,p]
    bidn32_centerline2.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(bidn32_lower_img[bidn32_centerline2,np.arange(0,2560)],100)/1.225,z2[:],'-',linewidth=4,color=cs(0))

z = np.linspace(z1[330]*D0,2,10000)
beta0 = rho_to_phi(windowed_average(bidn32_upper_img[bidn32_centerline1,np.arange(330,2560)],100)[0],2500,1.225)
lazy2 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.7)
#ax.plot(lazy2[3]/1.225,z/D0,'--',color=cs(0),linewidth=2)

dn45_centerline1 = []
for p in range(1330,2560):
    c_prof = dn45_upper_img[:,p]
    dn45_centerline1.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(dn45_upper_img[dn45_centerline1,np.arange(1330,2560)],100)/1.225,z1[1330:],'-',linewidth=4,color=cs(500))

dn45_centerline2 = []
for p in range(0,2560):
    c_prof = dn45_lower_img[:,p]
    dn45_centerline2.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(dn45_lower_img[dn45_centerline2,np.arange(0,2560)],100)/1.225,z2[:],'-',linewidth=4,color=cs(500))

z = np.linspace(z1[1330]*D0,2,10000)
beta0 = rho_to_phi(windowed_average(dn45_upper_img[dn45_centerline1,np.arange(1330,2560)],100)[0],2500,1.225)
lazy3 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.64)
#ax.plot(lazy3[3]/1.225,z/D0,'--',color=cs(500),linewidth=2)

ylab = ax.set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
xlab = ax.set_xlabel(r'$\rho_b/\rho_f$',fontsize=28,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=18);

ax.set_ylim(75,0);
ax.set_xlim(0,60);
ax.xaxis.tick_top()
yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
ax.xaxis.set_label_position('top')
#mline = mlines.Line2D([],[],color='k',marker='None',linestyle='--',label='Lazy Plume Model')
#dline = mlines.Line2D([],[],color='k',marker='None',linestyle='-',label='Experimental Data')

#ax.legend(handles=[dline,mline],fontsize=25,framealpha=0.5)

p1line = mlines.Line2D([],[],color=cs(0),marker='None',linestyle='-',label='$P_1$',linewidth=4)
p2line = mlines.Line2D([],[],color=cs(100),marker='None',linestyle='-',label='$P_2$',linewidth=4)
p3line = mlines.Line2D([],[],color=cs(500),marker='None',linestyle='-',label='$P_3$',linewidth=4)

ax.legend(handles=[p1line,p2line,p3line],fontsize=25,framealpha=0.5,loc=4)
f.savefig('/home/ajp25/Desktop/rhobs_c.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')


#%% Velocity Centerline
# z arrays
dn32_upper_piv1 = pivdn32_upper.read_frame2d(0)
dn32_upper_piv2 = pivdn32_upper2.read_frame2d(0)
zpiv1 = (np.arange(0,dn32_upper_piv1[0].shape[1])*pivdn32_upper.dx-outlet)*upper_cal/D0
dn32_lower_piv1 = pivdn32_lower.read_frame2d(0)
dn32_lower_piv2 = pivdn32_lower2.read_frame2d(0)
zpiv2 = ((np.linspace(2560,2*2560,dn32_lower_piv1[1].shape[1])-outlet)*lower_cal-overlap)/D0


bidn32_upper_piv1 = pivbidn32_upper.read_frame2d(0)
bidn32_upper_piv2 = pivbidn32_upper2.read_frame2d(0)
bidn32_lower_piv1 = pivbidn32_lower.read_frame2d(0)
bidn32_lower_piv2 = pivbidn32_lower2.read_frame2d(0)


pivdn45_upper = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper.0064.def.msk.ave.piv')
pivdn45_upper2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper2.0064.def.msk.ave.piv')
pivdn45_lower = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower.0048.def.msk.ave.piv')
pivdn45_lower2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower2.0048.def.msk.ave.piv')

dn45_upper_piv1 = pivdn45_upper.read_frame2d(0)
dn45_upper_piv2 = pivdn45_upper2.read_frame2d(0)
dn45_lower_piv1 = pivdn45_lower.read_frame2d(0)
dn45_lower_piv2 = pivdn45_lower2.read_frame2d(0)
zpiv1a = (np.arange(0,dn45_upper_piv1[0].shape[1])*pivdn45_upper.dx-outlet)*upper_cal/D0
zpiv2a = ((np.linspace(2560,2*2560,dn45_lower_piv2[1].shape[1])-outlet)*lower_cal-overlap)/D0


#%% Centerline Vertical Velocity
tau_p = 7.4e-3
w0 = tau_p*9.81
f,ax  = plt.subplots(figsize=(6,10))
cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]/D0))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
dn32_centerline1 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W = ((dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]) +(dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]))/2
for p in range(0,dn32_upper_piv1[1].shape[1]):
    w_prof = W[:,p]
    dn32_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
ax.plot(windowed_average(np.mean(W[20:22,np.where(zpiv1>3)[0].astype('int')],axis=0)[:-1],10)*upper_cal/deltat/w0,zpiv1[zpiv1>3][:-1],'-',linewidth=4,color=cs(100))

dn32_centerline2 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W = (dn32_lower_piv1[1] + dn32_lower_piv2[1])/2
for p in range(0,dn32_lower_piv1[1].shape[1]):
    w_prof = W[:,p]
    dn32_centerline2[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
ax.plot(windowed_average(W[[r for r in dn32_centerline2],[c for c in range(0,pivdn32_lower2.nx)]][1:-1],10)*lower_cal/deltat/w0,zpiv2[1:-1],'-',linewidth=4,color=cs(100))

#ax.plot(z/D0,lazy1[1]/w0,'--',color=cs(100),linewidth=2)


bidn32_centerline1 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W = ((bidn32_upper_piv1[1]*R[0,0] - bidn32_upper_piv1[2]*R[0,1]) +(bidn32_upper_piv2[1]*R[0,0] - bidn32_upper_piv2[2]*R[0,1]))/2
for p in range(0,dn32_upper_piv1[1].shape[1]):
    w_prof = W[:,p]
    bidn32_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
ax.plot(windowed_average(np.mean(W[20:22,np.where(zpiv1>3)[0].astype('int')],axis=0)[:-1],10)*upper_cal/deltat/w0,zpiv1[zpiv1>3][:-1],'-',linewidth=4,color=cs(0))

bidn32_centerline2 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W = (bidn32_lower_piv1[1]+ bidn32_lower_piv2[1])/2
for p in range(0,dn32_lower_piv1[1].shape[1]):
    w_prof = W[:,p]
    bidn32_centerline2[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
ax.plot(windowed_average(W[[r for r in bidn32_centerline2],[c for c in range(0,pivbidn32_lower2.nx)]][1:-1],10)*lower_cal/deltat/w0,zpiv2[1:-1],'-',linewidth=4,color=cs(0))


#ax.plot(z/D0,lazy2[1]/w0,'--',color=cs(0),linewidth=2)

dn45_centerline1 = np.zeros(np.arange(0,dn45_upper_piv1[1].shape[1]).shape).astype('int')
W = ((dn45_upper_piv2[1]*R[0,0] - dn45_upper_piv2[2]*R[0,1]))# +(dn45_upper_piv1[1]*R[0,0] - dn45_upper_piv1[2]*R[0,1]))/2
for p in range(0,dn45_upper_piv1[1].shape[1]):
    w_prof = W[:,p]
    dn45_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
ax.plot(windowed_average(W[[r for r in dn45_centerline1[np.where(zpiv1a>6)[0][0]:]],[c for c in range(np.where(zpiv1a>6)[0][0],pivdn45_upper2.nx)]][1:-1],10)*upper_cal/deltat/w0,zpiv1a[zpiv1a>6][1:-1],'-',linewidth=4,color=cs(500))

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
wc = np.mean([wc1,wc2,wc3],axis=0)
ax.plot(windowed_average(wc,25)*lower_cal/deltat/w0,zpiv2[1:-1],'-',linewidth=4,color=cs(500))
#ax.plot(windowed_average(np.mean(W[15:20,:],axis=0)[1:-1],5)*lower_cal/deltat/w0,zpiv2[1:-1],'-',linewidth=4,color=cs(500))


#ax.plot(z/D0,lazy3[1]/w0,'--',color=cs(500),linewidth=2)

xlab = ax.set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
ylab = ax.set_xlabel(r'$W_c / \tau_p g$',fontsize=28,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=18);
#ax.set_xlim(0,65)
ax.set_ylim(65,0);
ax.xaxis.tick_top()
yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
ax.xaxis.set_label_position('top')
#mline = mlines.Line2D([],[],color='k',marker='None',linestyle='--',label='Lazy Plume Model')
#dline = mlines.Line2D([],[],color='k',marker='None',linestyle='-',label='Experimental Data')

#ax.legend(handles=[dline,mline],fontsize=20)

p1line = mlines.Line2D([],[],color=cs(0),marker='None',linestyle='-',label='$P_1$',linewidth=4)
p2line = mlines.Line2D([],[],color=cs(100),marker='None',linestyle='-',label='$P_2$',linewidth=4)
p3line = mlines.Line2D([],[],color=cs(500),marker='None',linestyle='-',label='$P_3$',linewidth=4)

ax.legend(handles=[p1line,p2line,p3line],fontsize=25,framealpha=0.5)
f.savefig('/home/ajp25/Desktop/Wcs.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')

#%% plume width & spreading rate
img1 = dn32_upper.read_frame2d(0)
img2 = dn32_upper2.read_frame2d(0)
dn32_upper_img = (img1+img2)/2
z1 = (np.arange(0,2560)-outlet)*upper_cal/D0

img1 = dn32_lower.read_frame2d(0)
img2 = dn32_lower2.read_frame2d(0)
dn32_lower_img = (img1+img2)/2
z2 = ((np.arange(2560,2560*2)-outlet)*lower_cal-overlap)/D0

img1 = bidn32_upper.read_frame2d(0)
img2 = bidn32_upper2.read_frame2d(0)
bidn32_upper_img = (img1+img2)/2

img1 = bidn32_lower.read_frame2d(0)
img2 = bidn32_lower2.read_frame2d(0)
bidn32_lower_img = (img1+img2)/2

img1 = dn45_upper.read_frame2d(0)
img2 = dn45_upper2.read_frame2d(0)
dn45_upper_img = (img1+img2)/2

img1 = dn45_lower.read_frame2d(0)
img2 = dn45_lower2.read_frame2d(0)
dn45_lower_img = (img1+img2)/2

cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]/D0))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax  = plt.subplots(figsize=(6,10))
kernel=10
dn32_r1 = []
z = []
for p in range(500,2560,kernel):
    c_prof = windowed_average(np.mean(dn32_upper_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
    dn32_r1.append(np.max([r1,r2]))
    z.append((z1[p]+z1[p+kernel-1])/2)
z = np.array(z)
dpdz_dn32_upper = np.polyfit(z[z>20],windowed_average(np.array(dn32_r1)[z>20],kernel)*upper_cal/D0,1)
ax.plot(windowed_average(np.array(dn32_r1)*upper_cal/D0,30),z,'-',linewidth=4,color=cs(100))

dn32_r2 = []
z = []
for p in range(0,2560,kernel):
    c_prof = windowed_average(np.mean(dn32_lower_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    if a2==975:
        a2 = a2-1
        bs[1] = bs[1]-1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    try:
        r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
        dn32_r2.append(np.min([r1,r2]))
    except:
        dn32_r2.append(r1)
    z.append((z2[p]+z2[p+kernel-1])/2)

z = np.array(z)
dpdz_dn32_lower = np.polyfit(z[z<60],windowed_average(np.array(dn32_r2)[z<60],kernel)*lower_cal/D0,1)
ax.plot(windowed_average(np.array(dn32_r2)*lower_cal/D0,30),z,'-',linewidth=4,color=cs(100))

#z = np.linspace(z1[500]*D0,2,10000)
#beta0 = rho_to_phi(windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)[0],2500,1.225)
#lazy1 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.5)
#ax.plot(lazy1[3]/1.225,z/D0,'--',color=cs(100),linewidth=2)

bidn32_r1 = []
z = []
for p in range(330,2560,kernel):
    c_prof = windowed_average(np.mean(bidn32_upper_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
    bidn32_r1.append(np.max([r1,r2]))
    z.append((z1[p]+z1[p+kernel-1])/2)

z = np.array(z)
ax.plot(windowed_average(np.array(bidn32_r1)*upper_cal/D0,30),z,'-',linewidth=4,color=cs(0))
dpdz_bidn32_upper = np.polyfit(z[z>20],np.array(bidn32_r1)[z>20]*lower_cal/D0,1)


bidn32_r2 = []
z = []
for p in range(0,2560,kernel):
    c_prof = windowed_average(np.mean(bidn32_lower_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    if a2==975:
        a2 = a2-1
        bs[1] = bs[1]-1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    try:
        r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
        bidn32_r2.append(np.mean([r1,r2]))
    except:
        bidn32_r2.append(r1)
    z.append((z2[p]+z2[p+kernel-1])/2)

z = np.array(z)
ax.plot(windowed_average(np.array(bidn32_r2)*lower_cal/D0,30),z,'-',linewidth=4,color=cs(0))
dpdz_bidn32_lower = np.polyfit(z[z<60],np.array(bidn32_r2)[z<60]*lower_cal/D0,1)


#z = np.linspace(z1[330]*D0,2,10000)
#beta0 = rho_to_phi(windowed_average(bidn32_upper_img[bidn32_centerline1,np.arange(330,2560)],100)[0],2500,1.225)
#lazy2 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.7)
#ax.plot(lazy2[3]/1.225,z/D0,'--',color=cs(0),linewidth=2)

dn45_r1 = []
z = []
for p in range(1330,2560,kernel):
    c_prof = windowed_average(np.mean(dn45_upper_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
    dn45_r1.append(np.max([r1,r2]))
    z.append((z1[p]+z1[p+kernel-1])/2)
z = np.array(z)
ax.plot(windowed_average(np.array(dn45_r1)*upper_cal/D0,30),z,'-',linewidth=4,color=cs(500))
dpdz_dn45_upper = np.polyfit(z[z>15],np.array(dn45_r1)[z>15]*upper_cal/D0,1)


dn45_r2 = []
z= []
for p in range(0,2560,kernel):
    c_prof = windowed_average(np.mean(dn45_lower_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    if a2==975:
        a2 = a2-1
        bs[1] = bs[1]-1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    try:
        r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
        dn45_r2.append(np.min([r1,r2]))
    except:
        dn45_r2.append(r1)
    z.append((z2[p]+z2[p+kernel-1])/2)
z = np.array(z)
ax.plot(windowed_average(np.array(dn45_r2)*lower_cal/D0,30),z,'-',linewidth=4,color=cs(500))
dpdz_dn45_lower = np.polyfit(z[z<60],np.array(dn45_r2)[z<60]*lower_cal/D0,1)

#z = np.linspace(z1[1330]*D0,2,10000)
#beta0 = rho_to_phi(windowed_average(dn45_upper_img[dn45_centerline1,np.arange(1330,2560)],100)[0],2500,1.225)
#lazy3 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.64)
#ax.plot(lazy3[3]/1.225,z/D0,'--',color=cs(500),linewidth=2)

ylab = ax.set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
xlab = ax.set_xlabel(r'$b_p/D_0$',fontsize=28,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=18);

ax.set_ylim(75,0);
ax.set_xlim(0,5.5);
ax.xaxis.tick_top()
yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
ax.xaxis.set_label_position('top')
#mline = mlines.Line2D([],[],color='k',marker='None',linestyle='--',label='Lazy Plume Model')
#dline = mlines.Line2D([],[],color='k',marker='None',linestyle='-',label='Experimental Data')

#ax.legend(handles=[dline,mline],fontsize=25,framealpha=0.5)

p1line = mlines.Line2D([],[],color=cs(0),marker='None',linestyle='-',label='$P_1$',linewidth=4)
p2line = mlines.Line2D([],[],color=cs(100),marker='None',linestyle='-',label='$P_2$',linewidth=4)
p3line = mlines.Line2D([],[],color=cs(500),marker='None',linestyle='-',label='$P_3$',linewidth=4)

ax.legend(handles=[p1line,p2line,p3line],fontsize=25,framealpha=0.5,loc=1)
f.savefig('/home/ajp25/Desktop/bps.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')


#%% Plume widths based on velocity
cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]/D0))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax  = plt.subplots(figsize=(6,10))
dn32_r1 = []
kernel=5
W = ((dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]) +(dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]))/2
start = np.where(zpiv1>3)[0][0]
for p in range(start,dn32_upper_piv1[1].shape[1]-1):
    c_prof = windowed_average(np.mean(W[:,p:p+kernel],axis=1),3)
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
    dn32_r1.append(np.mean([r1,r2]))
ax.plot(np.array(dn32_r1)*pivdn32_upper.dx*upper_cal/D0,zpiv1[start:-1],'-',linewidth=4,color=cs(100))

dn32_r2 = []
W = (dn32_lower_piv1[1] + dn32_lower_piv2[1])/2
for p in range(0,W.shape[1]-1):
    c_prof = windowed_average(np.mean(W[:,p:p+kernel],axis=1),3)
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
    dn32_r2.append(np.mean([r1,r2]))
ax.plot(np.array(dn32_r2)*pivdn32_lower.dx*upper_cal/D0,zpiv2[:-1],'-',linewidth=4,color=cs(100))

#z = np.linspace(z1[500]*D0,2,10000)
#beta0 = rho_to_phi(windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)[0],2500,1.225)
#lazy1 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.5)
#ax.plot(lazy1[3]/1.225,z/D0,'--',color=cs(100),linewidth=2)

bidn32_r1 = []
for p in range(330,2560,kernel):
    c_prof = windowed_average(np.mean(bidn32_upper_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
    bidn32_r1.append(np.max([r1,r2]))
ax.plot(np.array(bidn32_r1)*upper_cal/D0,z1[330::kernel],'-',linewidth=4,color=cs(0))

bidn32_r2 = []
for p in range(0,2560,kernel):
    c_prof = windowed_average(np.mean(bidn32_lower_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    if a2==975:
        a2 = a2-1
        bs[1] = bs[1]-1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    try:
        r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
        bidn32_r2.append(np.mean([r1,r2]))
    except:
        bidn32_r2.append(r1)
ax.plot(np.array(bidn32_r2)*lower_cal/D0,z2[::kernel],'-',linewidth=4,color=cs(0))


#z = np.linspace(z1[330]*D0,2,10000)
#beta0 = rho_to_phi(windowed_average(bidn32_upper_img[bidn32_centerline1,np.arange(330,2560)],100)[0],2500,1.225)
#lazy2 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.7)
#ax.plot(lazy2[3]/1.225,z/D0,'--',color=cs(0),linewidth=2)

dn45_r1 = []
for p in range(1330,2560,kernel):
    c_prof = windowed_average(np.mean(dn45_upper_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
    dn45_r1.append(np.max([r1,r2]))
ax.plot(np.array(dn45_r1)*upper_cal/D0,z1[1330::kernel],'-',linewidth=4,color=cs(500))

dn45_r2 = []
for p in range(0,2560,kernel):
    c_prof = windowed_average(np.mean(dn45_lower_img[:,p:p+kernel],axis=1),25)[:-25]-1.225
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    if a2==975:
        a2 = a2-1
        bs[1] = bs[1]-1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    try:
        r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
        dn45_r2.append(np.min([r1,r2]))
    except:
        dn45_r2.append(r1)
ax.plot(np.array(dn45_r2)*lower_cal/D0,z2[::kernel],'-',linewidth=4,color=cs(500))


#z = np.linspace(z1[1330]*D0,2,10000)
#beta0 = rho_to_phi(windowed_average(dn45_upper_img[dn45_centerline1,np.arange(1330,2560)],100)[0],2500,1.225)
#lazy3 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.64)
#ax.plot(lazy3[3]/1.225,z/D0,'--',color=cs(500),linewidth=2)

ylab = ax.set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
xlab = ax.set_xlabel(r'$b_p/D_0$',fontsize=28,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=18);

ax.set_ylim(75,0);
ax.set_xlim(0,5.5);
ax.xaxis.tick_top()
yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
ax.xaxis.set_label_position('top')
#mline = mlines.Line2D([],[],color='k',marker='None',linestyle='--',label='Lazy Plume Model')
#dline = mlines.Line2D([],[],color='k',marker='None',linestyle='-',label='Experimental Data')

#ax.legend(handles=[dline,mline],fontsize=25,framealpha=0.5)

p1line = mlines.Line2D([],[],color=cs(0),marker='None',linestyle='-',label='$P_1$',linewidth=4)
p2line = mlines.Line2D([],[],color=cs(100),marker='None',linestyle='-',label='$P_2$',linewidth=4)
p3line = mlines.Line2D([],[],color=cs(500),marker='None',linestyle='-',label='$P_3$',linewidth=4)

ax.legend(handles=[p1line,p2line,p3line],fontsize=25,framealpha=0.5,loc=1)
f.savefig('/home/ajp25/Desktop/bps.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')

