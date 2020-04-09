#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:17:35 2019

@author: alec
"""


import numpy as np
from PLPlumes.pio import imgio,pivio

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

from PLPlumes.plume_processing.plume_functions import windowed_average
from PLPlumes import cmap_sunset

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
#%%
dn32_upper_piv1 = pivdn32_upper.read_frame2d(0)
dn32_upper_piv2 = pivdn32_upper2.read_frame2d(0)
zpiv1 = (np.arange(0,dn32_upper_piv1[0].shape[1])*pivdn32_upper.dx-outlet)*upper_cal/D0
dn32_lower_piv1 = pivdn32_lower.read_frame2d(0)
dn32_lower_piv2 = pivdn32_lower2.read_frame2d(0)
zpiv2 = ((np.linspace(2560,2*2560,dn32_lower_piv1[1].shape[1])-outlet)*lower_cal-overlap)/D0


bidn32_upper_piv1 = pivbidn32_upper.read_frame2d(0)
bidn32_upper_piv2 = pivdn32_upper2.read_frame2d(0)
bidn32_lower_piv1 = pivbidn32_lower.read_frame2d(0)
bidn32_lower_piv2 = pivdn32_lower2.read_frame2d(0)

dn45_upper_piv1 = pivdn45_upper.read_frame2d(0)
dn45_upper_piv2 = pivdn32_upper2.read_frame2d(0)
dn45_lower_piv1 = pivdn45_lower.read_frame2d(0)
dn45_lower_piv2 = pivdn32_lower2.read_frame2d(0)

#%% Self Similarity
# take velocity profiles at certain locations along streamwise distance of plume and fit Gaussian over
# full width (views 1 & 2) or reflecting across half width. Then plot W/W_max vs r/r_1/2
z2 = ((np.arange(2560,2560*2)-outlet)*lower_cal-overlap)/D0
cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax  = plt.subplots(figsize=(5,5))
#View 1 centerline velocity and r_1/2 location
prof_locs = [25,35,45,55,65,75,85,95]
W = ((dn32_upper_piv1[1]*R[0,0] - dn32_upper_piv1[2]*R[0,1]) +(dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]))/2
bin_s = 3
for p in prof_locs:
    w_prof = windowed_average(np.mean(W[:-1,p:p+bin_s],axis=1),3)
    wmax_loc = np.where(w_prof==w_prof.max())[0][0]
    data = np.flip(w_prof[0:wmax_loc])
    data = data
    r = np.arange(0,len(data))
    b = (np.where(data <= .5*np.max(data))[0][0])
    a = b-1
    interp = np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])
    rhalf = a +(np.where(np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])<=.5*np.max(data))[0][0]*1e-6)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(zpiv1[int((p+p+bin_s)/2)]))
    

#View 2 centerline velocity and r_1/2 location
prof_locs = [10,20,30,40,50,60,70,80]
W = (dn32_lower_piv1[1] + dn32_lower_piv2[1])/2

for p in prof_locs:
    w_prof = windowed_average(np.mean(W[:-1,p:p+bin_s],axis=1),3)
    wmax_loc = np.where(w_prof==w_prof.max())[0][0]
    data = np.flip(w_prof[0:wmax_loc])
    data = data
    r = np.arange(0,len(data))
    b = (np.where(data <= .5*np.max(data))[0][0])
    a = b-1
    interp = np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])
    rhalf = a +(np.where(np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])<=.5*np.max(data))[0][0]*1e-6)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(zpiv2[int((p+p+bin_s)/2)]))
xlab = ax.set_xlabel('$r / r_{1/2}$',fontsize=20,labelpad=15)
ylab = ax.set_ylabel('$W / W_{max}$',fontsize=20,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=16);
cbar_ax = f.add_axes([.91, 0.15, 0.025, 0.72])
cb = plt.colorbar(scalarMap,format='%d',cax=cbar_ax);
cbl=cb.set_label('$z/D_0$',size=20,labelpad=15);
cb.ax.tick_params(labelsize=15);
yticks = ax.yaxis.get_major_ticks();
yticks[0].label1.set_visible(False);
ax.set_ylim(0,1.1);
ax.set_xlim(0,3);

f.savefig('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/figures/self_sim_W_dn32.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
 
    
cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax  = plt.subplots(figsize=(5,5)) 
prof_locs = [25,30,45,55,65,75,85,95]
W = ((bidn32_upper_piv1[1]*R[0,0] - bidn32_upper_piv1[2]*R[0,1]) +(bidn32_upper_piv2[1]*R[0,0] - bidn32_upper_piv2[2]*R[0,1]))/2
bin_s = 5
for p in prof_locs:
    w_prof = windowed_average(np.mean(W[:-1,p:p+bin_s],axis=1),3)
    wmax_loc = np.where(w_prof==w_prof.max())[0][0]
    data = np.flip(w_prof[0:wmax_loc])
    data = data
    r = np.arange(0,len(data))
    b = (np.where(data <= .5*np.max(data))[0][0])
    a = b-1
    interp = np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])
    rhalf = a +(np.where(np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])<=.5*np.max(data))[0][0]*1e-6)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(zpiv1[int((p+p+bin_s)/2)]))
    

#View 2 centerline velocity and r_1/2 location
prof_locs = [20,40,50,60,70,80]
W = (bidn32_lower_piv1[1] + bidn32_lower_piv2[1])/2

for p in prof_locs:
    w_prof = windowed_average(np.mean(W[:-1,p:p+bin_s],axis=1),3)
    wmax_loc = np.where(w_prof==w_prof.max())[0][0]
    data = np.flip(w_prof[0:wmax_loc])
    data = data
    r = np.arange(0,len(data))
    b = (np.where(data <= .5*np.max(data))[0][0])
    a = b-1
    interp = np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])
    rhalf = a +(np.where(np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])<=.5*np.max(data))[0][0]*1e-6)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(zpiv2[int((p+p+bin_s)/2)]))
xlab = ax.set_xlabel('$r / r_{1/2}$',fontsize=20,labelpad=15)
ylab = ax.set_ylabel('$W / W_{max}$',fontsize=20,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=16);
cbar_ax = f.add_axes([.91, 0.15, 0.025, 0.72])
cb = plt.colorbar(scalarMap,format='%d',cax=cbar_ax);
cbl=cb.set_label('$z/D_0$',size=20,labelpad=15);
cb.ax.tick_params(labelsize=15);
ax.set_ylim(0,1.1);
ax.set_xlim(0,3);
yticks = ax.yaxis.get_major_ticks();
yticks[0].label1.set_visible(False);
f.savefig('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/figures/self_sim_W_bidn32.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')


cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax  = plt.subplots(figsize=(5,5))    
prof_locs = [25,35,45,55,65,75,85,95]
W = ((dn45_upper_piv1[1]*R[0,0] - dn45_upper_piv1[2]*R[0,1]) +(dn45_upper_piv2[1]*R[0,0] - dn45_upper_piv2[2]*R[0,1]))/2
bin_s = 3
for p in prof_locs:
    w_prof = windowed_average(np.mean(W[:-1,p:p+bin_s],axis=1),3)
    wmax_loc = np.where(w_prof==w_prof.max())[0][0]
    data = np.flip(w_prof[0:wmax_loc])
    data = data
    r = np.arange(0,len(data))
    b = (np.where(data <= .5*np.max(data))[0][0])
    a = b-1
    interp = np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])
    rhalf = a +(np.where(np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])<=.5*np.max(data))[0][0]*1e-6)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(zpiv1[int((p+p+bin_s)/2)]))
    

#View 2 centerline velocity and r_1/2 location
prof_locs = [10,23,30,40,50,60,70,80]
W = (dn45_lower_piv1[1] + dn45_lower_piv2[1])/2

for p in prof_locs:
    w_prof = windowed_average(np.mean(W[:-1,p:p+bin_s],axis=1),3)
    wmax_loc = np.where(w_prof==w_prof.max())[0][0]
    data = np.flip(w_prof[0:wmax_loc])
    data = data
    r = np.arange(0,len(data))
    b = (np.where(data <= .5*np.max(data))[0][0])
    a = b-1
    interp = np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])
    rhalf = a +(np.where(np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])<=.5*np.max(data))[0][0]*1e-6)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(zpiv2[int((p+p+bin_s)/2)]))
      
xlab = ax.set_xlabel('$r / r_{1/2}$',fontsize=20,labelpad=15)
ylab = ax.set_ylabel('$W / W_{max}$',fontsize=20,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=16);
cbar_ax = f.add_axes([.91, 0.15, 0.025, 0.72])
cb = plt.colorbar(scalarMap,format='%d',cax=cbar_ax);
cbl=cb.set_label('$z/D_0$',size=20,labelpad=15);
cb.ax.tick_params(labelsize=15);
ax.set_ylim(0,1.1);
ax.set_xlim(0,3);
yticks = ax.yaxis.get_major_ticks();
yticks[0].label1.set_visible(False);
f.savefig('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/figures/self_sim_W_dn45.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')

# single view
cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=zpiv2[0],vmax=zpiv2[14])
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f2,ax2  = plt.subplots(figsize=(5,5))
prof_locs = np.arange(5,15,1)
W = (dn32_lower_piv1[1] + dn32_lower_piv2[1])/2

for p in prof_locs:
    w_prof = windowed_average(np.mean(W[:-1,p:p+bin_s],axis=1),3)
    wmax_loc = np.where(w_prof==w_prof.max())[0][0]
    data = np.flip(w_prof[0:wmax_loc])
    data = data
    r = np.arange(0,len(data))
    b = (np.where(data <= .5*np.max(data))[0][0])
    a = b-1
    interp = np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])
    rhalf = a +(np.where(np.interp(np.arange(a,b,1e-6),[a,b],[data[a],data[b]])<=.5*np.max(data))[0][0]*1e-6)
    ax2.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(zpiv2[int((p+p+bin_s)/2)]))
    
    
xlab = ax2.set_xlabel('$r / r_{1/2}$',fontsize=20,labelpad=15)
ylab = ax2.set_ylabel('$W / W_{max}$',fontsize=20,labelpad=15)
ax2.tick_params(axis='both',which='major',labelsize=10);
cbar_ax = f2.add_axes([.91, 0.15, 0.025, 0.72])
cb = plt.colorbar(scalarMap,format='%d',cax=cbar_ax);
cbl=cb.set_label('$z/D_0$',size=20,labelpad=15);
cb.ax.tick_params(labelsize=15);
ax2.set_ylim(0,1.1);
ax2.set_xlim(0,3);
yticks = ax.yaxis.get_major_ticks();
yticks[0].label1.set_visible(False);
f2.savefig('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/figures/self_sim_W_dn32_subset.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')

#%% Self Similarity concentration
# take concentration profiles at certain locations along streamwise distance of plume and fit Gaussian over
# full width (views 1 & 2) or reflecting across half width. Then plot C/C_max vs r/r_1/2
z2 = ((np.arange(2560,2560*2)-outlet)*lower_cal-overlap)/D0

f,ax  = plt.subplots(figsize=(5,5))

cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=z2[-1])
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
#dn32 upper view
prof_locs = np.arange(600,2500,200)
c = (np.flipud(dn32_upper.read_frame2d(0))-1.225 + np.flipud(dn32_upper2.read_frame2d(0))-1.225)/2
rhalfs = []
cmax_locs = []
bin_s = 50
kernel = 25
hs = np.linspace(-outlet*upper_cal,(2560-outlet)*upper_cal,c.shape[1])/D0
for p in prof_locs:
    c_prof = windowed_average(np.mean(c[:,p:p+bin_s],axis=1),kernel)
    cmax_loc = np.where(c_prof==c_prof.max())[0][0]
    cmax_locs.append(np.where(c_prof==c_prof.max())[0][0])
    data = c_prof[cmax_loc:]
    r = np.arange(0,len(data))
    rhalf = np.where(data <= .5*np.max(data))[0][0]
    rhalfs.append(rhalf)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(hs[int((p+p+bin_s)/2)]))
    
    

#dn32 lower view
prof_locs = np.arange(100,2500,200)
c = (np.flipud(dn32_lower.read_frame2d(0))-1.225 + np.flipud(dn32_lower2.read_frame2d(0))-1.225)/2
rhalfs = []
cmax_locs = []

hs = z2
for p in prof_locs:
    c_prof = windowed_average(np.mean(c[:,p:p+bin_s],axis=1),kernel)
    cmax_loc = np.where(c_prof==c_prof.max())[0][0]
    cmax_locs.append(np.where(c_prof==c_prof.max())[0][0])
    data = c_prof[cmax_loc:]
    r = np.arange(0,len(data))
    rhalf = np.where(data <= .5*np.max(data))[0][0]
    rhalfs.append(rhalf)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(hs[int((p+p+bin_s)/2)]))

xlab = ax.set_xlabel('$r / r_{1/2}$',fontsize=20,labelpad=15)
ylab = ax.set_ylabel('$C / C_{max}$',fontsize=20,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=16);
cbar_ax = f.add_axes([.91, 0.15, 0.025, 0.72])
cb = plt.colorbar(scalarMap,format='%d',cax=cbar_ax);
cbl=cb.set_label('$z/D_0$',size=20,labelpad=15);
cb.ax.tick_params(labelsize=15);
ax.set_ylim(0,1.1);
ax.set_xlim(0,3);
yticks = ax.yaxis.get_major_ticks();
yticks[0].label1.set_visible(False);
f.savefig('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/figures/self_sim_C_dn32.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
 


f,ax  = plt.subplots(figsize=(5,5))

cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=z2[-1])
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
#bidn32 upper view
prof_locs = np.arange(800,2500,200)
c = (np.flipud(bidn32_upper.read_frame2d(0))-1.225 + np.flipud(bidn32_upper2.read_frame2d(0))-1.225)/2
rhalfs = []
cmax_locs = []
bin_s = 50
kernel = 25
hs = np.linspace(-outlet*upper_cal,(2560-outlet)*upper_cal,c.shape[1])/D0
for p in prof_locs:
    c_prof = windowed_average(np.mean(c[:,p:p+bin_s],axis=1),kernel)
    cmax_loc = np.where(c_prof==c_prof.max())[0][0]
    cmax_locs.append(np.where(c_prof==c_prof.max())[0][0])
    data = c_prof[cmax_loc:]
    r = np.arange(0,len(data))
    rhalf = np.where(data <= .5*np.max(data))[0][0]
    rhalfs.append(rhalf)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(hs[int((p+p+bin_s)/2)]))
    
    

#bidn32 lower view
prof_locs = np.arange(100,2500,200)
c = (np.flipud(bidn32_lower.read_frame2d(0))-1.225 + np.flipud(bidn32_lower2.read_frame2d(0))-1.225)/2
rhalfs = []
cmax_locs = []

hs = z2
for p in prof_locs:
    c_prof = windowed_average(np.mean(c[:,p:p+bin_s],axis=1),kernel)
    cmax_loc = np.where(c_prof==c_prof.max())[0][0]
    cmax_locs.append(np.where(c_prof==c_prof.max())[0][0])
    data = c_prof[cmax_loc:]
    r = np.arange(0,len(data))
    rhalf = np.where(data <= .5*np.max(data))[0][0]
    rhalfs.append(rhalf)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(hs[int((p+p+bin_s)/2)]))

xlab = ax.set_xlabel('$r / r_{1/2}$',fontsize=20,labelpad=15)
ylab = ax.set_ylabel('$C / C_{max}$',fontsize=20,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=16);
cbar_ax = f.add_axes([.91, 0.15, 0.025, 0.72])
cb = plt.colorbar(scalarMap,format='%d',cax=cbar_ax);
cbl=cb.set_label('$z/D_0$',size=20,labelpad=15);
cb.ax.tick_params(labelsize=15);
ax.set_ylim(0,1.1);
ax.set_xlim(0,3);
yticks = ax.yaxis.get_major_ticks();
yticks[0].label1.set_visible(False);
f.savefig('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/figures/self_sim_C_bidn32.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
 


f,ax  = plt.subplots(figsize=(5,5))

cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=z2[-1])
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
#dn45 upper view
prof_locs = np.arange(600,2500,200)
c = (np.flipud(dn45_upper.read_frame2d(0))-1.225 + np.flipud(dn45_upper2.read_frame2d(0))-1.225)/2
rhalfs = []
cmax_locs = []
bin_s = 50
kernel = 25
hs = np.linspace(-outlet*upper_cal,(2560-outlet)*upper_cal,c.shape[1])/D0
for p in prof_locs:
    c_prof = windowed_average(np.mean(c[:,p:p+bin_s],axis=1),kernel)
    cmax_loc = np.where(c_prof==c_prof.max())[0][0]
    cmax_locs.append(np.where(c_prof==c_prof.max())[0][0])
    data = c_prof[cmax_loc:]
    r = np.arange(0,len(data))
    rhalf = np.where(data <= .5*np.max(data))[0][0]
    rhalfs.append(rhalf)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(hs[int((p+p+bin_s)/2)]))
    
    

#dn45 lower view
prof_locs = np.arange(100,2500,200)
c = (np.flipud(dn45_lower.read_frame2d(0))-1.225 + np.flipud(dn45_lower2.read_frame2d(0))-1.225)/2
rhalfs = []
cmax_locs = []

hs = z2
for p in prof_locs:
    c_prof = windowed_average(np.mean(c[:,p:p+bin_s],axis=1),kernel)
    cmax_loc = np.where(c_prof==c_prof.max())[0][0]
    cmax_locs.append(np.where(c_prof==c_prof.max())[0][0])
    data = c_prof[cmax_loc:]
    r = np.arange(0,len(data))
    rhalf = np.where(data <= .5*np.max(data))[0][0]
    rhalfs.append(rhalf)
    ax.plot(r/rhalf,data/data.max(),color=scalarMap.to_rgba(hs[int((p+p+bin_s)/2)]))

xlab = ax.set_xlabel('$r / r_{1/2}$',fontsize=20,labelpad=15)
ylab = ax.set_ylabel('$C / C_{max}$',fontsize=20,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=16);
cbar_ax = f.add_axes([.91, 0.15, 0.025, 0.72])
cb = plt.colorbar(scalarMap,format='%d',cax=cbar_ax);
cbl=cb.set_label('$z/D_0$',size=20,labelpad=15);
cb.ax.tick_params(labelsize=15);
ax.set_ylim(0,1.1);
ax.set_xlim(0,3);
yticks = ax.yaxis.get_major_ticks();
yticks[0].label1.set_visible(False);
f.savefig('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/figures/self_sim_C_dn45.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
 
