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
from matplotlib.ticker import FormatStrFormatter


from PLPlumes.plume_processing.plume_functions import windowed_average
from PLPlumes import cmap_sunset

from PLPlumes.plume_processing.plume_functions import rho_to_phi, phi_to_rho
from PLPlumes.modeling import pure_plume, lazy_plume
from scipy.optimize import curve_fit

from matplotlib import ticker
import numpy as np

def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:   
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

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
dn32_upper_img = (img1+img2)/2/1.5
z1 = (np.arange(0,2560)-outlet)*upper_cal/D0

img1 = dn32_lower.read_frame2d(0)
img2 = dn32_lower2.read_frame2d(0)
dn32_lower_img = (img1+img2)/2/1.5
z2 = ((np.arange(2560,2560*2)-outlet)*lower_cal-overlap)/D0

img1 = bidn32_upper.read_frame2d(0)
img2 = bidn32_upper2.read_frame2d(0)
bidn32_upper_img = (img1+img2)/2/1.5

img1 = bidn32_lower.read_frame2d(0)
img2 = bidn32_lower2.read_frame2d(0)
bidn32_lower_img = (img1+img2)/2/1.5

img1 = dn45_upper.read_frame2d(0)
img2 = dn45_upper2.read_frame2d(0)
dn45_upper_img = (img1+img2)/2/1.25/1.5

img1 = dn45_lower.read_frame2d(0)
img2 = dn45_lower2.read_frame2d(0)
dn45_lower_img = (img2+img2)/2*np.linspace(1.65,1.25,2560)/1.25/1.5

#%% modeling
#from PLPlumes.modeling import liu_model
#from PLPlumes.modeling import lazy_plume,pure_plume
from PLPlumes.plume_processing.plume_functions import phi_to_rho,rho_to_phi
r0 = 1.905e-2/2
tau_p = 7.4e-3
w0 = tau_p*9.81
z = np.arange(0,2,5e-6)
rho_p = 2500
rho_f = 1.225
dp = 30e-6
mu_f = 1.825e-5
"""
# Lai z,bf_0,bp_0,wf_0,ws_0,phi_v0,dp,rho_p,rho_f,mu_f=1.825e-5
mfr = 1.42e-3 #g/s
rhob_0 = 25*1.225
phi_v0 = rho_to_phi(rhob_0*2,rho_p,rho_f)
wp_0 = mfr/(rhob_0*np.pi*r0**2) *(1-phi_v0*.5)/(0.5 - phi_v0/3)
wf_0 = wp_0-w0
bf_0 = r0
bp_0 = r0
ws_0 = w0

lai_bidn32 = lai_model(z,bf_0,bp_0,wf_0,ws_0,phi_v0,dp,rho_p,rho_f,mu_f)

mfr = 3e-3 #g/s
rhob_0 = 50*1.225
phi_v0 = rho_to_phi(rhob_0,rho_p,rho_f)
wp_0 = mfr/(rhob_0*np.pi*r0**2) *(1-phi_v0*.5)/(0.5 - phi_v0/3)
wf_0 = wp_0-w0
bf_0 = r0
bp_0 = r0
ws_0 = w0
lai_dn32 = lai_model(z,bf_0,bp_0,wf_0,ws_0,phi_v0,dp,rho_p,rho_f,mu_f)

mfr = 8e-3 #g/s
rhob_0 = 120*1.225
phi_v0 = rho_to_phi(rhob_0,rho_p,rho_f)
wp_0 = mfr/(rhob_0*np.pi*r0**2)# *(1-phi_v0*.5)/(0.5 - phi_v0/3)
wf_0 = wp_0-w0
bf_0 = r0
bp_0 = r0
ws_0 = w0
lai_dn45 = lai_model(z,bf_0,bp_0,wf_0,ws_0,phi_v0,dp,rho_p,rho_f,mu_f)
"""

mfr = 1.42e-3 #g/s
rhob_0 = 14.4#21.5*1.225/1.5/2
phi_v0 = rho_to_phi(rhob_0,rho_p,rho_f)
wp_0 = mfr/(rhob_0*np.pi*r0**2)
pure_bidn32 = pure_plume_model(z,r0,rho_f,rhob_0,phi_v0,wp_0,alpha=0.042)

mfr = 3e-3 #g/s
rhob_0 = 68#46.7*1.225/1.5/2
phi_v0 = rho_to_phi(rhob_0,rho_p,rho_f)
wp_0 = mfr/(rhob_0*np.pi*r0**2)
pure_dn32 = pure_plume_model(z,r0,rho_f,rhob_0,phi_v0,wp_0,alpha=0.044)

mfr = 8e-3 #g/s
rhob_0 = 128#93.2/2
phi_v0 = rho_to_phi(rhob_0,rho_p,rho_f)
wp_0 = mfr/(rhob_0*np.pi*r0**2)
pure_dn45 = pure_plume_model(z,r0,rho_f,rhob_0,phi_v0,wp_0,alpha=0.035)

# Liu w_p0,w_a0,beta_0,rho_p,rho_a,ra_0,rp_0,dp,mu_a,alpha,mfr,z
mfr = 1.42e-3 #g/s
rhob_0 = 14.4#21.5*1.225/1.5
wp_0 = mfr/(rhob_0*np.pi*r0**2)
wf_0 = wp_0-w0/2
bf_0 = r0
alpha = 0.042
liu_bidn32 = liu_model(wp_0,wf_0,rho_to_phi(rhob_0,rho_p,rho_f),rho_p,rho_f,r0,r0,dp,mu_f,alpha,mfr,z)

mfr = 3e-3 #g/s
rhob_0 = 68#46.7*1.225/1.5
wp_0 = mfr/(rhob_0*np.pi*r0**2)
wf_0 = wp_0-w0/2
bf_0 = r0
alpha = 0.044
liu_dn32 = liu_model(wp_0,wf_0,2*rho_to_phi(rhob_0,rho_p,rho_f),rho_p,rho_f,r0,r0,dp,mu_f,alpha,mfr,z)

mfr = 8e-3 #g/s
rhob_0 = 128#93.2
wp_0 = mfr/(rhob_0*np.pi*r0**2)
wf_0 = wp_0-w0/2
bf_0 = r0
alpha = 0.035
liu_dn45 = liu_model(wp_0,wf_0,rho_to_phi(rhob_0,rho_p,rho_f),rho_p,rho_f,r0,r0,dp,mu_f,alpha,mfr,z)

# lazy single-phase z,r0,rho_a,rho_p,beta_0,wp0,alpha=0.03
mfr = 1.42e-3 #g/s
rhob_0 = 14.4#21.5*1.225/1.5/2
wp_0 = mfr/(rhob_0*np.pi*r0**2)
alpha = 0.042
lazy_bidn32 = lazy_plume(z,r0,rho_f,rhob_0,rho_to_phi(rhob_0,rho_p,rho_f),wp_0,alpha)

mfr = 3e-3 #g/s
rhob_0 = 68#46.7*1.225/1.5/2
wp_0 = mfr/(rhob_0*np.pi*r0**2)
alpha = 0.044
lazy_dn32 = lazy_plume(z,r0,rho_f,rhob_0,rho_to_phi(rhob_0,rho_p,rho_f),wp_0,alpha)

mfr = 8e-3 #g/s
rhob_0 = 128#93.2/2
wp_0 = mfr/(rhob_0*np.pi*r0**2)
alpha = 0.035
lazy_dn45 = lazy_plume(z,r0,rho_f,rhob_0,rho_to_phi(rhob_0,rho_p,rho_f),wp_0,alpha)

# forced single-phase z,r0,rho_a,rho_p,beta_0,wp0,alpha=0.03
mfr = 1.42e-3 #g/s
rhob_0 = 14.4#21.5*1.225/1.5/2
wp_0 = mfr/(rhob_0*np.pi*r0**2)
alpha = 0.042
Gamma_0 = .16
forced_bidn32 = lazy_plume(z,r0,rho_f,rhob_0,rho_to_phi(rhob_0,rho_p,rho_f),wp_0,alpha,Gamma_0)

mfr = 3e-3 #g/s
rhob_0 = 68#46.7*1.225/1.5/2
wp_0 = mfr/(rhob_0*np.pi*r0**2)
alpha = 0.044
Gamma_0 = .052
forced_dn32 = lazy_plume(z,r0,rho_f,rhob_0,rho_to_phi(rhob_0,rho_p,rho_f),wp_0,alpha,Gamma_0)

mfr = 8e-3 #g/s
rhob_0 = 128#93.2/2
wp_0 = mfr/(rhob_0*np.pi*r0**2)
alpha = 0.035
Gamma_0 = .015
forced_dn45 = lazy_plume(z,r0,rho_f,rhob_0,rho_to_phi(rhob_0,rho_p,rho_f),wp_0,alpha,Gamma_0)


#%%
lai_bidn32 = np.load('/home/ajp25/Desktop/lai_bidn32.npz')['arr_0']
lai_dn32 = np.load('/home/ajp25/Desktop/lai_dn32.npz')['arr_0']
lai_dn45 = np.load('/home/ajp25/Desktop/lai_dn45.npz')['arr_0']
lazy_bidn32 = np.load('/home/ajp25/Desktop/lazy_bidn32.npz')['arr_0']
lazy_dn32 = np.load('/home/ajp25/Desktop/lazy_dn32.npz')['arr_0']
lazy_dn45 = np.load('/home/ajp25/Desktop/lazy_dn45.npz')['arr_0']
#%% power law decay
def linlaw(x, a, b) :
    return a + x * b

def curve_fit_log(xdata, ydata) :
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)
#%% Centerline Concentration

cs = plt.get_cmap('inferno')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]/D0))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax  = plt.subplots(figsize=(6,10))
fc,axc = plt.subplots(1,3,figsize=(18,10),sharey=True)
dn32_centerline1 = []
d = 100
for p in range(500,2560):
    c_prof = dn32_upper_img[:,p]
    dn32_centerline1.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot((windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)/1.225)[::d],z1[500:][::d],'o',linewidth=4,color=cs(100))
axc[1].plot((windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)/1.225)[::d],z1[500:][::d],'o',linewidth=4,color=cs(100))

dn32_centerline2 = []
for p in range(0,2560):
    c_prof = dn32_lower_img[:,p]
    dn32_centerline2.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot((windowed_average(dn32_lower_img[dn32_centerline2,np.arange(0,2560)],100)/1.225)[::d],z2[:][::d],'o',linewidth=4,color=cs(100))
axc[1].plot((windowed_average(dn32_lower_img[dn32_centerline2,np.arange(0,2560)],100)/1.225)[::d],z2[:][::d],'o',linewidth=4,color=cs(100))

x = np.hstack((z1[z1>=15],z2[z2<60]))
y1 = windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)
y2 = windowed_average(dn32_lower_img[dn32_centerline2,np.arange(0,2560)],100)
y = np.hstack((y1[z1[500:]>=15],y2[z2<60]))
fit_dn32 = curve_fit_log(x,y)
#ax.loglog(np.power(10,linlaw(np.log10(np.arange(15,60)),*fit_dn32[0])),np.arange(15,60),'--',color=cs(100))

axc[1].plot(2*lazy_dn32[3]/1.225,z/D0,'--',color='k',linewidth=2)
axc[1].plot(2*forced_dn32[3]/1.225,z/D0,'-.',color='k',linewidth=2)
axc[1].plot(2*pure_dn32[1]/1.225,z/D0,'-',color='k',linewidth=2)
axc[1].plot(2*liu_dn32[6]/1.225,z/D0,':',color='k',linewidth=2)
axc[1].set_xlim(0,40);

bidn32_centerline1 = []
for p in range(330,2560):
    c_prof = bidn32_upper_img[:,p]
    bidn32_centerline1.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(bidn32_upper_img[bidn32_centerline1,np.arange(330,2560)],100)[::d]/1.225,z1[330:][::d],'o',linewidth=4,color=cs(0))
axc[0].plot(windowed_average(bidn32_upper_img[bidn32_centerline1,np.arange(330,2560)],100)[::d]/1.225,z1[330:][::d],'o',linewidth=4,color=cs(0))

bidn32_centerline2 = []
for p in range(0,2560):
    c_prof = dn32_lower_img[:,p]
    bidn32_centerline2.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(bidn32_lower_img[bidn32_centerline2,np.arange(0,2560)],100)[::d]/1.225,z2[:][::d],'o',linewidth=4,color=cs(0))
axc[0].plot(windowed_average(bidn32_lower_img[bidn32_centerline2,np.arange(0,2560)],100)[::d]/1.225,z2[:][::d],'o',linewidth=4,color=cs(0))

x = np.hstack((z1[z1>=20],z2[z2<60]))
y1 = windowed_average(bidn32_upper_img[bidn32_centerline1,np.arange(330,2560)],100)
y2 = windowed_average(bidn32_lower_img[bidn32_centerline2,np.arange(0,2560)],100)
y = np.hstack((y1[z1[330:]>=20],y2[z2<60]))
fit_bidn32 = curve_fit_log(x,y)
#ax.loglog(np.power(10,linlaw(np.log10(np.arange(20,60)),*fit_bidn32[0])),np.arange(20,60),'--',color=cs(0))

axc[0].plot(2*lazy_bidn32[3]/1.225,z/D0,'--',color='k',linewidth=2)
axc[0].plot(2*forced_bidn32[3]/1.225,z/D0,'-.',color='k',linewidth=2)
axc[0].plot(2*pure_bidn32[1]/1.225,z/D0,'-',color='k',linewidth=2)
axc[0].plot(2*liu_bidn32[6]/1.225,z/D0,':',color='k',linewidth=2)
axc[0].set_xlim(0,15)

dn45_centerline1 = []
for p in range(1330,2560):
    c_prof = dn45_upper_img[:,p]
    dn45_centerline1.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(dn45_upper_img[dn45_centerline1,np.arange(1330,2560)],100)[::d]/1.225,z1[1330:][::d],'o',linewidth=4,color=cs(220))
axc[2].plot(windowed_average(dn45_upper_img[dn45_centerline1,np.arange(1330,2560)],100)[::d]/1.225,z1[1330:][::d],'o',linewidth=4,color=cs(220))
#ax.plot(windowed_average(np.mean(dn45_upper_img[480:530,1330:],axis=0),100)/1.225,z1[1330:],'-',linewidth=4,color=cs(500))
#axc[2].plot(windowed_average(np.mean(dn45_upper_img[480:530,1330:],axis=0),100)/1.225,z1[1330:],'-',linewidth=4,color=cs(500))
dn45_centerline2 = []
for p in range(0,2560):
    c_prof = dn45_lower_img[:,p]
    dn45_centerline2.append(np.where(c_prof == np.max(c_prof))[0][0])
ax.plot(windowed_average(dn45_lower_img[dn45_centerline2,np.arange(0,2560)],100)[::d]/1.225,z2[:][::d],'o',linewidth=4,color=cs(220))
axc[2].plot(windowed_average(dn45_lower_img[dn45_centerline2,np.arange(0,2560)],100)[::d]/1.225,z2[:][::d],'o',linewidth=4,color=cs(220))

x = np.hstack((z1[z1>=20],z2[z2<60]))
y1 = windowed_average(dn45_upper_img[dn45_centerline1,np.arange(1330,2560)],100)
y2 = windowed_average(dn45_lower_img[dn45_centerline2,np.arange(0,2560)],100)
y = np.hstack((y1[z1[1330:]>=20],y2[z2<60]))
fit_dn45 = curve_fit_log(x,y)
#ax.loglog(np.power(10,linlaw(np.log10(np.arange(20,60)),*fit_dn45[0])),np.arange(20,60),'--',color=cs(220),subsy=[10,20,30,60])

axc[2].plot(2*lazy_dn45[3]/1.225,z/D0,'--',color='k',linewidth=2)
axc[2].plot(2*forced_dn45[3]/1.225,z/D0,'-.',color='k',linewidth=2)
axc[2].plot(2*pure_dn45[1]/1.225,z/D0,'-',color='k',linewidth=2)
axc[2].plot(2*liu_dn45[6]/1.225,z/D0,':',color='k',linewidth=2)
axc[2].set_xlim(0,50)

ylabc = axc[0].set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
xlabc = axc[1].set_xlabel(r'$\rho_b/\rho_f$',fontsize=34,labelpad=20)
axc[0].tick_params(axis='both',which='major',labelsize=18);
axc[1].tick_params(axis='both',which='major',labelsize=18);
axc[2].tick_params(axis='both',which='major',labelsize=18);
axc[0].xaxis.tick_top()
axc[1].xaxis.tick_top()
axc[2].xaxis.tick_top()
yticks = axc[0].yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks = axc[1].yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks = axc[2].yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
axc[1].xaxis.set_label_position('top')
axc[0].set_ylim(70,0)
p1line = mlines.Line2D([],[],color='k',marker='None',linestyle='-',label='Single-phase Pure Plume Model',linewidth=4)
p2line = mlines.Line2D([],[],color='k',marker='None',linestyle='--',label='Single-phase Lazy Plume Model',linewidth=4)
p3line = mlines.Line2D([],[],color='k',marker='None',linestyle='-.',label='Single-phase Forced Plume Model',linewidth=4)
p4line = mlines.Line2D([],[],color='k',marker='None',linestyle=':',label='Multiphase Liu 2003 Model',linewidth=4)
legend = axc[1].legend(handles=[p1line,p2line,p3line,p4line],fontsize=25,framealpha=0.5,loc='upper center',fancybox=True,bbox_to_anchor=(0.5,-0.001))
#fc.savefig('/home/ajp25/Desktop/rhobs_c_models2.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlabc,ylabc,legend], bbox_inches='tight')
#axc[0].set_yscale('log')
#axc[0].set_xscale('log')
#axc[1].set_yscale('log')
#axc[1].set_xscale('log')
#axc[2].set_yscale('log')
#axc[2].set_xscale('log')

ylab = ax.set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
xlab = ax.set_xlabel(r'$\rho_b/\rho_f$',fontsize=34,labelpad=20)

ax.tick_params(axis='both',which='major',labelsize=18);
ax.set_ylim(61,0);
ax.set_xlim(0,40);
ax.xaxis.tick_top()
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.yaxis.set_ticklabels(['','','','','$10^1$','$2\times 10^1$','$3\times 10^1$','','','$6\times 10^1$'])

yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
ax.xaxis.set_label_position('top')
#ax.text(1.65,30,'$-1.6$',fontsize=20);
#ax.text(4,50,'$-1.7$',fontsize=20);
#ax.text(15,40,'$-1.15$',fontsize=20);

#mline = mlines.Line2D([],[],color='k',marker='None',linestyle='--',label='Lazy Plume Model')
#dline = mlines.Line2D([],[],color='k',marker='None',linestyle='-',label='Experimental Data')

#ax.legend(handles=[dline,mline],fontsize=25,framealpha=0.5)

p1line = mlines.Line2D([],[],color=cs(0),linestyle='None',marker='o',label='$P_1$',ms=8)
p2line = mlines.Line2D([],[],color=cs(100),linestyle='None',marker='o',label='$P_2$',ms=8)
p3line = mlines.Line2D([],[],color=cs(220),linestyle='None',marker='o',label='$P_3$',ms=8)

ax.legend(handles=[p1line,p2line,p3line],fontsize=25,framealpha=0.5,loc=4)
#f.savefig('/home/ajp25/Desktop/rhobs_c.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')


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


#pivdn45_upper = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper.0064.def.msk.ave.piv')
#pivdn45_upper2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper2.0064.def.msk.ave.piv')
#pivdn45_lower = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower.0048.def.msk.ave.piv')
#pivdn45_lower2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower2.0048.def.msk.ave.piv')

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
cs = plt.get_cmap('inferno')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]/D0))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
#fc,axc = plt.subplots(1,3,figsize=(18,10),sharey=True)
d = 5
dn32_centerline1 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W1 = ((dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]) +(dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]))/2
for p in range(0,dn32_upper_piv1[1].shape[1]):
    w_prof = W1[:,p]
    dn32_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
ax.plot(windowed_average(np.mean(W1[20:22,np.where(zpiv1>5)[0].astype('int')],axis=0)[:-1],10)[::d]*upper_cal/deltat/w0,zpiv1[zpiv1>5][:-1][::d],'o',linewidth=4,color=cs(100))
#axc[1].plot(windowed_average(np.mean(W1[20:22,np.where(zpiv1>5)[0].astype('int')],axis=0)[:-1],10)[::d]*upper_cal/deltat/w0,zpiv1[zpiv1>5][:-1][::d],'o',linewidth=4,color=cs(100))
w1_p2 = windowed_average(np.mean(W1[20:22,15:],axis=0)[:-1],10)*upper_cal/deltat
#ax.plot(w1_p2[::d]/w0,zpiv1[15:-1][::d],'o',linewidth=4,color=cs(100))
#axc[1].plot(w1_p2[::d]/.31,zpiv1[15:-1][::d],'o',linewidth=4,color=cs(100))

dn32_centerline2 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W2 = (dn32_lower_piv1[1] + dn32_lower_piv2[1])/2
for p in range(0,dn32_lower_piv1[1].shape[1]):
    w_prof = W2[:,p]
    dn32_centerline2[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#ax.plot(windowed_average(W2[[r for r in dn32_centerline2],[c for c in range(0,pivdn32_lower2.nx)]][1:-1],10)[::d]*lower_cal/deltat/w0,zpiv2[1:-1][::d],'o',linewidth=4,color=cs(100))
#axc[1].plot(windowed_average(W2[[r for r in dn32_centerline2],[c for c in range(0,pivdn32_lower2.nx)]][1:-1],10)[::d]*lower_cal/deltat/w0,zpiv2[1:-1][::d],'o',linewidth=4,color=cs(100))
w2_p2 = windowed_average(np.mean(W2[20:22,:],axis=0)[:-1],10)*lower_cal/deltat
ax.plot(w2_p2[::d]/w0,zpiv2[:-1][::d],'o',linewidth=4,color=cs(100))
#axc[1].plot(w2_p2[::d]/.31,zpiv2[:-1][::d],'o',linewidth=4,color=cs(100))

x = np.hstack((zpiv1[15:-1][zpiv1[15:-1]>5],zpiv2[:-1][zpiv2[:-1]<65]))
y = np.hstack((w1_p2[zpiv1[15:-1]>5],w2_p2[zpiv2[:-1]<65]))
fit_dn32 = curve_fit_log(x,y)
#ax.loglog(np.power(1300,linlaw(np.log10(np.arange(10,65)),*fit_dn32[0])),np.arange(10,65),'--',color=cs(100))
#ax.loglog(np.power(10,linlaw(np.log10(np.arange(10,50)),*fit_dn32[0])),np.arange(10,50),'--',color=cs(100))

#axc[1].plot(2*lazy_dn32[1]/.31,z/D0,'--',color='k',linewidth=2)
#axc[1].plot(2*forced_dn32[1]/.31,z/D0,'-.',color='k',linewidth=2)
#axc[1].plot(2*pure_dn32[2]/.31,z/D0,'-',color='k',linewidth=2)
#axc[1].plot(liu_dn32[0]/.31+.5,z/D0,':',color='k',linewidth=2)

bidn32_centerline1 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W1 = ((bidn32_upper_piv1[1]*R[0,0] - bidn32_upper_piv1[2]*R[0,1]) +(bidn32_upper_piv2[1]*R[0,0] - bidn32_upper_piv2[2]*R[0,1]))/2
for p in range(0,dn32_upper_piv1[1].shape[1]):
    w_prof = W1[:,p]
    bidn32_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
w1_p1 = windowed_average(np.mean(W1[20:22,15:],axis=0),5)*upper_cal/deltat
ax.plot(w1_p1[:-1][::d]/w0,zpiv1[15:-1][::d],'o',linewidth=4,color=cs(0))
axc[0].plot(w1_p1[:-1][::d]/.69,zpiv1[15:-1][::d],'o',linewidth=4,color=cs(0))
#ax.plot(windowed_average(np.mean(W1[20:22,np.where(zpiv1>3)[0].astype('int')],axis=0)[:-1],10)[::d]*upper_cal/deltat/w0,zpiv1[zpiv1>3][:-1][::d],'o',linewidth=4,color=cs(0))
#axc[0].plot(windowed_average(np.mean(W1[20:22,np.where(zpiv1>3)[0].astype('int')],axis=0)[:-1],10)[::d]*upper_cal/deltat/w0,zpiv1[zpiv1>3][:-1][::d],'o',linewidth=4,color=cs(0))

bidn32_centerline2 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W2 = (bidn32_lower_piv1[1]+ bidn32_lower_piv2[1])/2
for p in range(0,dn32_lower_piv1[1].shape[1]):
    w_prof = W2[:,p]
    bidn32_centerline2[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#ax.plot(windowed_average(W2[[r for r in bidn32_centerline2],[c for c in range(0,pivbidn32_lower2.nx)]][1:-1],10)[::d]*lower_cal/deltat/w0,zpiv2[1:-1][::d],'o',linewidth=4,color=cs(0))
#axc[0].plot(windowed_average(W2[[r for r in bidn32_centerline2],[c for c in range(0,pivbidn32_lower2.nx)]][1:-1],10)[::d]*lower_cal/deltat/w0,zpiv2[1:-1][::d],'o',linewidth=4,color=cs(0))
w2_p1 = windowed_average(np.mean(W2[20:22,:],axis=0),5)*lower_cal/deltat
ax.plot(w2_p1[:-1][::d]/w0,zpiv2[:-1][::d],'o',linewidth=4,color=cs(0))
axc[0].plot(w2_p1[:-1][::d]/.69,zpiv2[:-1][::d],'o',linewidth=4,color=cs(0))

#axc[0].plot(2*lazy_bidn32[1]/.69,z/D0,'--',color='k',linewidth=2)
#axc[0].plot(2*forced_bidn32[1]/.69,z/D0,'-.',color='k',linewidth=2)
#axc[0].plot(2*pure_bidn32[2]/.69,z/D0,'-',color='k',linewidth=2)
#axc[0].plot(liu_bidn32[0]/.69+.5,z/D0,':',color='k',linewidth=2)

x = np.hstack((zpiv1[15:][zpiv1[15:]>5],zpiv2[zpiv2[:]<65]))
y = np.hstack((w1_p1[zpiv1[15:]>5],w2_p1[zpiv2[:]<65]))
fit_bidn32 = curve_fit_log(x,y)
#ax.loglog(np.power(10,linlaw(np.log10(np.arange(10,65)),*fit_bidn32[0])),np.arange(10,65),'--',color=cs(0))

dn45_centerline1 = np.zeros(np.arange(0,dn45_upper_piv1[1].shape[1]).shape).astype('int')
W1 = ((dn45_upper_piv2[1]*R[0,0] - dn45_upper_piv2[2]*R[0,1]) +(dn45_upper_piv1[1]*R[0,0] - dn45_upper_piv1[2]*R[0,1]))/2
for p in range(0,dn45_upper_piv1[1].shape[1]):
    w_prof = W1[:,p]
    dn45_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
w1_p3 = windowed_average(np.mean(W1[20:22,15:],axis=0),5)*upper_cal/deltat
ax.plot(w1_p3[:-1][::d]/w0,zpiv1[15:-1][::d],'o',linewidth=4,color=cs(220))
axc[2].plot(w1_p3[:-1][::d]/.44,zpiv1[15:-1][::d],'o',linewidth=4,color=cs(220))
#ax.plot(windowed_average(W1[[r for r in dn45_centerline1[np.where(zpiv1a>6)[0][0]:]],
#                            [c for c in range(np.where(zpiv1a>6)[0][0],pivdn45_upper2.nx)]][1:-1],10)[::d]*upper_cal/deltat/w0,
 #                           zpiv1a[zpiv1a>6][1:-1][::d],'o',linewidth=4,color=cs(220))
#axc[2].plot(windowed_average(W1[[r for r in dn45_centerline1[np.where(zpiv1a>6)[0][0]:]],
#                            [c for c in range(np.where(zpiv1a>6)[0][0],pivdn45_upper2.nx)]][1:-1],10)[::d]*upper_cal/deltat/w0,
#                            zpiv1a[zpiv1a>6][1:-1][::d],'o',linewidth=4,color=cs(220))

dn45_centerline2 = np.zeros(np.arange(0,dn45_lower_piv1[1].shape[1]).shape).astype('int')
W = (dn45_lower_piv1[1] + dn45_lower_piv2[1])/2
for p in range(0,dn45_lower_piv1[1].shape[1]):
    w_prof = W[:,p]
    dn45_centerline2[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
    
#wc1 = W[[r for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
#wc2 = W[[r+1 for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
#wc3 = W[[r-1 for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
#wc4 = W[[r+2 for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
#wc5 = W[[r-2 for r in dn45_centerline2],[c for c in range(0,pivdn45_lower.nx)]][1:-1]
#wc = np.mean([wc1,wc2,wc3],axis=0)
#ax.plot(windowed_average(np.mean(W[17:23,:],axis=0),10)[::d]*lower_cal/deltat/w0,zpiv2[1:-1][::d],'o',linewidth=4,color=cs(220))
#axc[2].plot(windowed_average(np.mean(W[17:23,:],axis=0),10)[::d]*lower_cal/deltat/w0,zpiv2[1:-1][::d],'o',linewidth=4,color=cs(220))
w2_p3 = windowed_average(np.mean(W[20:22,:],axis=0),5)*lower_cal/deltat
ax.plot(w2_p3[::d]/w0,zpiv2[1:-1][::d],'o',linewidth=4,color=cs(220))
axc[2].plot(w2_p3[::d]/.44,zpiv2[1:-1][::d],'o',linewidth=4,color=cs(220))

x = np.hstack((zpiv1[15:][zpiv1[15:]>15],zpiv2[zpiv2[:]<65]))
y = np.hstack((w1_p3[zpiv1[15:]>15],w2_p3[zpiv2[:]<65]))
fit_dn45 = curve_fit_log(x,y)
#ax.loglog(np.power(70,linlaw(np.log10(np.arange(15,65)),*fit_dn45[0])),np.arange(15,65),'--',color=cs(220))

#axc[2].plot(2*lazy_dn45[1]/.44,z/D0,'--',color='k',linewidth=2)
#axc[2].plot(2*forced_dn45[1]/.44,z/D0,'-.',color='k',linewidth=2)
#axc[2].plot(2*pure_dn45[2]/.44,z/D0,'-',color='k',linewidth=2)
#axc[2].plot(liu_dn45[0]/.44 +.5,z/D0,':',color='k',linewidth=2)

ylabc = axc[0].set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
xlabc = axc[1].set_xlabel(r'$W_c/W_0$',fontsize=34,labelpad=20)
axc[0].tick_params(axis='both',which='major',labelsize=18);
axc[1].tick_params(axis='both',which='major',labelsize=18);
axc[2].tick_params(axis='both',which='major',labelsize=18);
axc[0].xaxis.tick_top()
axc[1].xaxis.tick_top()
axc[2].xaxis.tick_top()
yticks = axc[0].yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks = axc[1].yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks = axc[2].yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
axc[1].xaxis.set_label_position('top')
axc[0].set_ylim(65,0)
#axc[0].set_xlim(0,35)
#axc[1].set_xlim(0,35)
#axc[2].set_xlim(0,35)

p1line = mlines.Line2D([],[],color='k',marker='None',linestyle='-',label='Single-phase Pure Plume Model',linewidth=4)
p2line = mlines.Line2D([],[],color='k',marker='None',linestyle='--',label='Single-phase Lazy Plume Model',linewidth=4)
p3line = mlines.Line2D([],[],color='k',marker='None',linestyle='-.',label='Single-phase Forced Plume Model',linewidth=4)
p4line = mlines.Line2D([],[],color='k',marker='None',linestyle=':',label='Multiphase Liu 2003 Model',linewidth=4)
legend = axc[1].legend(handles=[p1line,p2line,p3line,p4line],fontsize=25,framealpha=0.5,loc='upper center',fancybox=True,bbox_to_anchor=(0.5,-0.001))
#fc.savefig('/home/ajp25/Desktop/Wcs_models.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlabc,ylabc,legend], bbox_inches='tight')
#ax.plot(z/D0,lazy3[1]/w0,'--',color=cs(500),linewidth=2)
#axc[0].set_yscale('log')
#axc[0].set_xscale('log')
#axc[1].set_xscale('log')
#axc[2].set_xscale('log')

xlab = ax.set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
ylab = ax.set_xlabel(r'$W_c / \tau_pg$',fontsize=28,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=18);
ax.set_xlim(10,35)
ax.set_ylim(60,3);
#ax.set_xlim(.5,10);
ax.xaxis.tick_top()
yticks = ax.yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
ax.xaxis.set_label_position('top')
ax.set_yscale('log')
ax.set_xscale('log')
#ax.text(1.4,40,'$-0.40$',fontsize=20)
#ax.text(7,55,'$-0.1$',fontsize=20)
#ax.text(2.9,25,'$0.08$',fontsize=20)

#mline = mlines.Line2D([],[],color='k',marker='None',linestyle='--',label='Lazy Plume Model')
#dline = mlines.Line2D([],[],color='k',marker='None',linestyle='-',label='Experimental Data')

#ax.legend(handles=[dline,mline],fontsize=20)

p1line = mlines.Line2D([],[],color=cs(0),linestyle='None',marker='o',label='$P_1$',ms=8)
p2line = mlines.Line2D([],[],color=cs(100),linestyle='None',marker='o',label='$P_2$',ms=8)
p3line = mlines.Line2D([],[],color=cs(220),linestyle='None',marker='o',label='$P_3$',ms=8)

ax.legend(handles=[p1line,p2line,p3line],fontsize=25,framealpha=0.5)
#f.savefig('/home/ajp25/Desktop/Wcs_loglog.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')

#%% plume width & spreading rate
dn32_upper = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper.avg.img')
dn32_upper2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper2.avg.img')
dn32_lower = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower.avg.img')
dn32_lower2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower2.avg.img')



bidn32_upper = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/upper/bidn32_upper.avg.img')
bidn32_upper2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/upper/bidn32_upper2.avg.img')
bidn32_lower = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/lower/bidn32_lower.avg.img')
bidn32_lower2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_bi_dn32/whole_plume/lower/bidn32_lower2.avg.img')


dn45_upper = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper.avg.img')
dn45_upper2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/upper/dn45_upper2.avg.img')
dn45_lower = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower.avg.img')
dn45_lower2 = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn45/whole_plume/lower/dn45_lower2.avg.img')

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
dn45_lower_img = (img2+img2)/2

cs = plt.get_cmap('inferno')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]/D0))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax  = plt.subplots(figsize=(6,10))
fc,axc = plt.subplots(1,3,figsize=(18,10),sharey=True)

kernel=10
dn32_r1 = []
z = []
for p in range(500,2560,kernel):
    c_prof = windowed_average(np.mean(dn32_upper_img[:,p:p+kernel],axis=1),25)[:-25]
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
    z.append((z1[p]+z1[p+kernel-1])/2)
z = np.array(z)
dpdz_dn32_upper = np.polyfit(z[z>20],windowed_average(np.array(dn32_r1)[z>20],kernel)*upper_cal/D0,1)
ax.plot(windowed_average(np.array(dn32_r1)*upper_cal/D0,30)[::10],z[::10],'o',linewidth=4,color=cs(100))
axc[1].plot(windowed_average(np.array(dn32_r1)*upper_cal/D0,30)[::10],z[::10],'o',linewidth=4,color=cs(100))
r1_p2 = binned_statistic(z,windowed_average(np.array(dn32_r1)*upper_cal,30),statistic=np.nanmean,bins=zpiv1[15:])
dn32_r2 = []
z = []
for p in range(0,2560,kernel):
    c_prof = windowed_average(np.mean(dn32_lower_img[:,p:p+kernel],axis=1),25)[:-25]
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
        dn32_r2.append(np.mean([r1,r2]))
    except:
        dn32_r2.append(r1)
    z.append((z2[p]+z2[p+kernel-1])/2)

z = np.array(z)
dpdz_dn32_lower = np.polyfit(z[z<60],windowed_average(np.array(dn32_r2)[z<60],kernel)*lower_cal/D0,1)
ax.plot(windowed_average(np.array(dn32_r2)*lower_cal/D0-.3,30)[::10],z[::10],'o',linewidth=4,color=cs(100))
axc[1].plot(windowed_average(np.array(dn32_r2)*lower_cal/D0-.3,30)[::10],z[::10],'o',linewidth=4,color=cs(100))
r2_p2 = binned_statistic(z,windowed_average(np.array(dn32_r2)*lower_cal,30),statistic=np.nanmean,bins=zpiv2)

z = np.arange(0,2,5e-6)
#axc[1].plot(lazy_dn32[2]/D0/np.sqrt(np.log(2)),z/D0,'--',color='k',linewidth=2)
#axc[1].plot(forced_dn32[2]/D0/np.sqrt(np.log(2)),z/D0,'-.',color='k',linewidth=2)
#axc[1].plot(pure_dn32[3]/D0/np.sqrt(np.log(2)),z/D0,'-',color='k',linewidth=2)
#axc[1].plot(liu_dn32[5]/D0/np.sqrt(np.log(2)),z/D0,':',color='k',linewidth=2)
#z = np.linspace(z1[500]*D0,2,10000)
#beta0 = rho_to_phi(windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)[0],2500,1.225)
#lazy1 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.5)
#ax.plot(lazy1[3]/1.225,z/D0,'--',color=cs(100),linewidth=2)

bidn32_r1 = []
z = []
for p in range(500,2560,kernel):
    c_prof = windowed_average(np.mean(bidn32_upper_img[:,p:p+kernel],axis=1),25)[:-25]
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
    bidn32_r1.append(np.mean([r1,r2]))
    z.append((z1[p]+z1[p+kernel-1])/2)

z = np.array(z)
ax.plot(windowed_average(np.array(bidn32_r1)*upper_cal/D0,30)[::10],z[::10],'o',linewidth=4,color=cs(0))
dpdz_bidn32_upper = np.polyfit(z[z>20],np.array(bidn32_r1)[z>20]*lower_cal/D0,1)
axc[0].plot(windowed_average(np.array(bidn32_r1)*upper_cal/D0,30)[::10],z[::10],'o',linewidth=4,color=cs(0))


bidn32_r2 = []
z = []
for p in range(0,2560,kernel):
    c_prof = windowed_average(np.mean(bidn32_lower_img[:,p:p+kernel],axis=1),25)[:-25]
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
ax.plot(windowed_average(np.array(bidn32_r2)*lower_cal/D0,30)[::10],z[::10],'o',linewidth=4,color=cs(0))
dpdz_bidn32_lower = np.polyfit(z[z<60],np.array(bidn32_r2)[z<60]*lower_cal/D0,1)
axc[0].plot(windowed_average(np.array(bidn32_r2)*lower_cal/D0,30)[::10],z[::10],'o',linewidth=4,color=cs(0))
z = np.arange(0,2,5e-6)
#axc[0].plot(lazy_bidn32[2]/D0/np.sqrt(np.log(2)),z/D0,'--',color='k',linewidth=2)
#axc[0].plot(forced_bidn32[2]/D0/np.sqrt(np.log(2)),z/D0,'-.',color='k',linewidth=2)
#axc[0].plot(pure_bidn32[3]/D0/np.sqrt(np.log(2)),z/D0,'-',color='k',linewidth=2)
#axc[0].plot(liu_bidn32[5]/D0/np.sqrt(np.log(2)),z/D0,':',color='k',linewidth=2)


dn45_r1 = []
z = []
for p in range(1330,2560,kernel):
    c_prof = windowed_average(np.mean(dn45_upper_img[:,p:p+kernel],axis=1),25)[:-25]
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    b = (np.where(c_prof >= 1/np.e*np.max(c_prof))[0])
    bs = [b[0],b[-1]]
    a1 = bs[0]-1
    a2 = bs[1]+1
    interp = np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])
    r1 = c_max - (a1 + np.where(np.interp(np.arange(a1,bs[0],1e-6),[a1,bs[0]],[c_prof[a1],c_prof[bs[0]]])>=1/np.e*np.max(c_prof))[0][0]*1e-6)
    interp = np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])
    r2 = ((bs[1] + np.where(np.interp(np.arange(bs[1],a2,1e-6),[bs[1],a2],[c_prof[bs[1]],c_prof[a2]])<=1/np.e*np.max(c_prof))[0][0]*1e-6))- c_max
    dn45_r1.append(np.mean([r1,r2]))
    z.append((z1[p]+z1[p+kernel-1])/2)
z = np.array(z)
ax.plot(windowed_average(np.array(dn45_r1)*upper_cal/D0,30)[::10],z[::10],'o',linewidth=4,color=cs(220))
dpdz_dn45_upper = np.polyfit(z[z>15],np.array(dn45_r1)[z>15]*upper_cal/D0,1)
axc[2].plot(windowed_average(np.array(dn45_r1)*upper_cal/D0,30)[::10],z[::10],'o',linewidth=4,color=cs(220))


dn45_r2 = []
z= []
for p in range(0,2560,kernel):
    c_prof = windowed_average(np.mean(dn45_lower_img[:,p:p+kernel],axis=1),25)[:-25]
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
        dn45_r2.append(np.mean([r1,r2]))
    except:
        dn45_r2.append(r1)
    z.append((z2[p]+z2[p+kernel-1])/2)
z = np.array(z)
ax.plot(windowed_average(np.array(dn45_r2)*lower_cal/D0-.4,30)[::10],z[::10],'o',linewidth=4,color=cs(220))
dpdz_dn45_lower = np.polyfit(z[z<60],np.array(dn45_r2)[z<60]*lower_cal/D0,1)
axc[2].plot(windowed_average(np.array(dn45_r2)*lower_cal/D0-.4,30)[::10],z[::10],'o',linewidth=4,color=cs(220))
z = np.arange(0,2,5e-6)
#axc[2].plot(lazy_dn45[2]/D0/np.sqrt(np.log(2)),z/D0,'--',color='k',linewidth=2)
#axc[2].plot(forced_dn45[2]/D0/np.sqrt(np.log(2)),z/D0,'-.',color='k',linewidth=2)
#axc[2].plot(pure_dn45[3]/D0/np.sqrt(np.log(2)),z/D0,'-',color='k',linewidth=2)
#axc[2].plot(liu_dn45[5]/D0/np.sqrt(np.log(2)),z/D0,':',color='k',linewidth=2)

ylabc = axc[0].set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
xlabc = axc[1].set_xlabel(r'$b_p/D_0$',fontsize=34,labelpad=20)
axc[0].tick_params(axis='both',which='major',labelsize=18);
axc[1].tick_params(axis='both',which='major',labelsize=18);
axc[2].tick_params(axis='both',which='major',labelsize=18);
axc[0].xaxis.tick_top()
axc[1].xaxis.tick_top()
axc[2].xaxis.tick_top()
yticks = axc[0].yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks = axc[1].yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
yticks = axc[2].yaxis.get_major_ticks()
yticks[0].label1.set_visible(False)
axc[1].xaxis.set_label_position('top')
axc[0].set_ylim(65,0)
axc[0].set_xlim(0,4.5)
axc[1].set_xlim(0,4.5)
axc[2].set_xlim(0,4.5)
p1line = mlines.Line2D([],[],color='k',marker='None',linestyle='-',label='Single-phase Pure Plume Model',linewidth=4)
p2line = mlines.Line2D([],[],color='k',marker='None',linestyle='--',label='Single-phase Lazy Plume Model',linewidth=4)
p3line = mlines.Line2D([],[],color='k',marker='None',linestyle='-.',label='Single-phase Forced Plume Model',linewidth=4)
p4line = mlines.Line2D([],[],color='k',marker='None',linestyle=':',label='Multiphase Liu 2003 Model',linewidth=4)
legend = axc[1].legend(handles=[p1line,p2line,p3line,p4line],fontsize=25,framealpha=0.5,loc='upper center',fancybox=True,bbox_to_anchor=(0.5,-0.001))
#fc.savefig('/home/ajp25/Desktop/bps_models.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlabc,ylabc,legend], bbox_inches='tight')

ylab = ax.set_ylabel('$z / D_{0}$',fontsize=28,labelpad=15)
xlab = ax.set_xlabel(r'$b_p/D_0$',fontsize=28,labelpad=15)
ax.tick_params(axis='both',which='major',labelsize=18);

ax.set_ylim(70,0);
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
p3line = mlines.Line2D([],[],color=cs(220),marker='None',linestyle='-',label='$P_3$',linewidth=4)

ax.legend(handles=[p1line,p2line,p3line],fontsize=25,framealpha=0.5,loc=1)
#f.savefig('/home/ajp25/Desktop/bps.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')


#%% Plume widths based on velocity
from PLPlumes.plume_processing.plume_functions import gaussian
from scipy.optimize import curve_fit
cs = plt.get_cmap('sunset')
cNorm = colors.Normalize(vmin=0,vmax=(z2[-1]/D0))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax  = plt.subplots(figsize=(6,10))
dn32_r1 = []
kernel=1
W = ((dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]) +(dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]))/2
start = np.where(zpiv1>3)[0][0]
for p in range(start,dn32_upper_piv1[1].shape[1]-1):
    c_prof = windowed_average(np.mean(W[:,p:p+kernel],axis=1),3)
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    fitx = np.arange(0,len(c_prof),.1)
    fity = curve_fit(gaussian,np.arange(0,len(c_prof)),c_prof,p0=[np.max(c_prof),c_max,10,0],bounds=([np.max(c_prof), c_max-5,1,0],[2*np.max(c_prof),c_max+5,len(c_prof),1]))[0]
    width = np.where(gaussian(fitx,fity[0],fity[1],fity[2],fity[3])>1/np.e*fity[0])[0]
    r1 = fity[1] - width[0]*.1
    dn32_r1.append(r1)
ax.plot(np.array(dn32_r1)*pivdn32_upper.dx*upper_cal/D0,zpiv1[start:-1],'-',linewidth=4,color=cs(100))

dn32_r2 = []
W = (dn32_lower_piv1[1] + dn32_lower_piv2[1])/2
for p in range(0,W.shape[1]-1):
    c_prof = windowed_average(np.mean(W[:,p:p+kernel],axis=1),3)
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    fitx = np.arange(0,len(c_prof),.1)
    fity = curve_fit(gaussian,np.arange(0,len(c_prof)),c_prof,p0=[np.max(c_prof),c_max,10,0],bounds=([np.max(c_prof), c_max-5,1,0],[2*np.max(c_prof),c_max+5,len(c_prof),1]))[0]
    width = np.where(gaussian(fitx,fity[0],fity[1],fity[2],fity[3])>1/np.e*fity[0])[0]
    r1 = fity[1] - width[0]*.1
    dn32_r2.append(r1)
ax.plot(np.array(dn32_r2)*pivdn32_lower.dx*lower_cal/D0,zpiv2[:-1],'-',linewidth=4,color=cs(100))

#z = np.linspace(z1[500]*D0,2,10000)
#beta0 = rho_to_phi(windowed_average(dn32_upper_img[dn32_centerline1,np.arange(500,2560)],100)[0],2500,1.225)
#lazy1 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.5)
#ax.plot(lazy1[3]/1.225,z/D0,'--',color=cs(100),linewidth=2)

W = ((bidn32_upper_piv2[1]*R[0,0] - bidn32_upper_piv2[2]*R[0,1]) +(bidn32_upper_piv2[1]*R[0,0] - bidn32_upper_piv2[2]*R[0,1]))/2
start = np.where(zpiv1>3)[0][0]
bidn32_r1 = []
for p in range(start,bidn32_upper_piv1[1].shape[1]-1):
    c_prof = windowed_average(np.mean(W[:,p:p+kernel],axis=1),3)
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    fitx = np.arange(0,len(c_prof),.1)
    fity = curve_fit(gaussian,np.arange(0,len(c_prof)),c_prof,p0=[np.max(c_prof),c_max,10,0],bounds=([np.max(c_prof), c_max-5,1,0],[2*np.max(c_prof),c_max+5,len(c_prof),1]))[0]
    width = np.where(gaussian(fitx,fity[0],fity[1],fity[2],fity[3])>1/np.e*fity[0])[0]
    r1 = fity[1] - width[0]*.1
    bidn32_r1.append(r1)
ax.plot(np.array(bidn32_r1)*pivbidn32_lower.dx*upper_cal/D0,zpiv1[start:-1],'-',linewidth=4,color=cs(0))

bidn32_r2 = []
W = (bidn32_lower_piv1[1] + bidn32_lower_piv2[1])/2
for p in range(0,bidn32_upper_piv1[1].shape[1]-1):
    c_prof = windowed_average(np.mean(W[:,p:p+kernel],axis=1),3)
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    c_prof = windowed_average(np.mean(W[:,p:p+kernel],axis=1),3)
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    fitx = np.arange(0,len(c_prof),.1)
    fity = curve_fit(gaussian,np.arange(0,len(c_prof)),c_prof,p0=[np.max(c_prof),c_max,10,0],bounds=([np.max(c_prof), c_max-5,1,0],[2*np.max(c_prof),c_max+5,len(c_prof),1]))[0]
    width = np.where(gaussian(fitx,fity[0],fity[1],fity[2],fity[3])>1/np.e*fity[0])[0]
    r1 = fity[1] - width[0]*.1
    bidn32_r2.append(r1)
ax.plot(np.array(bidn32_r2)*pivbidn32_lower.dx*lower_cal/D0,zpiv2[:-1],'-',linewidth=4,color=cs(0))


#z = np.linspace(z1[330]*D0,2,10000)
#beta0 = rho_to_phi(windowed_average(bidn32_upper_img[bidn32_centerline1,np.arange(330,2560)],100)[0],2500,1.225)
#lazy2 = lazy_plume.lazy_plume_model(z,2e-2/2,1.225,2500,beta0,1.7)
#ax.plot(lazy2[3]/1.225,z/D0,'--',color=cs(0),linewidth=2)

dn45_r1 = []
W = ((dn45_upper_piv2[1]*R[0,0] - dn45_upper_piv2[2]*R[0,1]) +(dn45_upper_piv2[1]*R[0,0] - dn45_upper_piv2[2]*R[0,1]))/2
start = np.where(zpiv1a>6)[0][0]
for p in range(start,dn45_upper_piv1[1].shape[1]-1):
    c_prof = windowed_average(np.mean(W[:,p:p+kernel],axis=1),3)
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    fitx = np.arange(0,len(c_prof),.1)
    fity = curve_fit(gaussian,np.arange(0,len(c_prof)),c_prof,p0=[np.max(c_prof),c_max,10,0],bounds=([np.max(c_prof), c_max-5,1,0],[2*np.max(c_prof),c_max+5,len(c_prof),1]))[0]
    width = np.where(gaussian(fitx,fity[0],fity[1],fity[2],fity[3])>1/np.e*fity[0])[0]
    r1 = fity[1] - width[0]*.1
    dn45_r1.append(r1)
ax.plot(np.array(dn45_r1)*pivdn45_upper.dx*upper_cal/D0,zpiv1a[start:-1],'-',linewidth=4,color=cs(500))

dn45_r2 = []
W = (dn45_lower_piv1[1] + dn45_lower_piv2[1])/2
for p in range(0,W.shape[1]-1):
    c_prof = windowed_average(np.mean(W[:,p:p+kernel],axis=1),3)
    c_max = np.where(c_prof==np.max(c_prof))[0][0]
    fitx = np.arange(0,len(c_prof),.1)
    fity = curve_fit(gaussian,np.arange(0,len(c_prof)),c_prof,p0=[np.max(c_prof),c_max,10,0],bounds=([np.max(c_prof), c_max-5,1,0],[2*np.max(c_prof),c_max+5,len(c_prof),1]))[0]
    width = np.where(gaussian(fitx,fity[0],fity[1],fity[2],fity[3])>1/np.e*fity[0])[0]
    r1 = fity[1] - width[0]*.1
    dn45_r2.append(r1)
ax.plot(np.array(dn45_r2)*pivdn45_lower.dx*lower_cal/D0,zpiv2[:-1],'-',linewidth=4,color=cs(500))

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
f.savefig('/home/ajp25/Desktop/bps_w.pdf',dpi=1200,format='pdf',bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
#%% Gamma plume flux parameter
from scipy.integrate import simps

Beta = (1.225-2500)/1.225
zpiv = np.hstack((zpiv1,zpiv2))
g = 9.81
x = np.arange(24,dn32_upper.ix-24,24)
y = np.arange(24,dn32_upper.iy-24,24)
slices = []
for i in x:
   for j in y:
       slices.append((slice(int(j),int(j+24)),slice(int(i),int(i+24))))
       
dn32_upper_phi = rho_to_phi(dn32_upper_img,2500,1.225)
dn32_lower_phi = rho_to_phi(dn32_lower_img,2500,1.225)
dn32_upper_phi[dn32_upper_phi<0] = 0
dn32_lower_phi[dn32_lower_phi<0] = 0

dn32_upper_buoy = dn32_upper_phi*Beta*g
dn32_lower_buoy = dn32_lower_phi*Beta*g

W1 = -((dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]) +(dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]))/2*upper_cal/deltat
W2 = -(dn32_lower_piv1[1] + dn32_lower_piv2[1])/2*lower_cal/deltat

Bflux_frame_upper = np.zeros(W1.shape)
Bflux_frame_lower = np.zeros(W2.shape)
phi_frame_upper = np.zeros(W1.shape)
phi_frame_lower = np.zeros(W2.shape)
for s in slices:
    Bflux_frame_upper[int((s[0].start+24/2)/(24)-1),int((s[1].start+24/2)/(24)-1)] = np.mean(dn32_upper_buoy[s])
    Bflux_frame_lower[int((s[0].start+24/2)/(24)-1),int((s[1].start+24/2)/(24)-1)] = np.mean(dn32_lower_buoy[s])
    phi_frame_upper[int((s[0].start+24/2)/(24)-1),int((s[1].start+24/2)/(24)-1)] = np.mean(dn32_upper_phi[s])
    phi_frame_lower[int((s[0].start+24/2)/(24)-1),int((s[1].start+24/2)/(24)-1)] = np.mean(dn32_lower_phi[s])

Bflux_upper = Bflux_frame_lower*W1
Bflux_lower = Bflux_frame_lower*W2

Bflux_dn32= []
b = Bflux_lower
mps = np.zeros((105,))
for c in range(0,105):
    try:
        data2 = b[:,c]
        data = data2
        max_loc = np.where(data==np.min(data))[0][0]
        ds = np.arange(0,len(data))*24*upper_cal
        ds = ds+np.diff(ds)[0]/2 - ds[max_loc]
        mps[c] = 4*simps(data*np.abs(ds),dx=np.diff(ds)[0])

    except:
        mps[c] = np.nan
Bflux_dn32.append(mps)

Bflux_dn32 = np.hstack((Bflux_dn32[0],Bflux_dn32[1]))

Mflux_dn32= []
b = W2
phi_frame = phi_frame_lower
mps = np.zeros((105,))
for c in range(0,105):
    try:
        data2 = b[:,c]
        data = data2
        max_loc = np.where(data==np.min(data))[0][0]
        ds = np.arange(0,len(data))*24*upper_cal
        ds = ds+np.diff(ds)[0]/2 - ds[max_loc]
        mps[c] = 4*simps((((data+w0)**2*(1-phi_frame[:,c]))+2500/1.225*data**2*phi_frame[:,c])*np.abs(ds),dx=np.diff(ds)[0])

    except:
        mps[c] = np.nan
Mflux_dn32.append(mps)

Mflux_dn32 = np.hstack((Mflux_dn32[0],Mflux_dn32[1]))


Qflux_dn32= []
b = W2
phi_frame = phi_frame_lower
mps = np.zeros((105,))
for c in range(0,105):
    try:
        data2 = b[:,c]
        data = -data2
        max_loc = np.where(data==np.max(data))[0][0]
        ds = np.arange(0,len(data))*24*upper_cal
        ds = ds+np.diff(ds)[0]/2 - ds[max_loc]
        mps[c] = (4*simps(data*(phi_frame[:,c])*np.abs(ds),dx=np.diff(ds)[0])) + (4*simps((data-w0)*(1-phi_frame[:,c])*np.abs(ds),dx=np.diff(ds)[0]))

    except:
        mps[c] = np.nan
Qflux_dn32.append(mps)

Qflux_dn32 = np.hstack((Qflux_dn32[0],Qflux_dn32[1]))

Gamma_dn32 = (5*np.abs(Qflux_dn32)**2*np.abs(Bflux_dn32))/(4*.025*np.abs(Mflux_dn32)**(5/2))