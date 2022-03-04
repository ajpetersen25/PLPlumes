#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 06:43:00 2021

@author: apetersen
"""


import numpy as np
from PLPlumes.pio import imgio,pivio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.lines as mlines
from matplotlib import rc
from matplotlib import ticker

#from matplotlib.ticker import *
#import matplotlib.gridspec as gridspec
#import matplotlib.patches as patches
#from matplotlib.lines import Line2D
#from matplotlib.font_manager import FontProperties
#from mpl_toolkits.axes_grid1 import make_axes_locatable
rc('text', usetex=True)
from matplotlib.ticker import FormatStrFormatter


from PLPlumes import cmap_sunset

from PLPlumes.plume_processing.plume_functions import rho_to_phi, phi_to_rho
def liu_model(w_p0,w_a0,beta_0,rho_p,rho_a,ra_0,rp_0,dp,mu_a,alpha,mfr,z):
    g = 9.81
    steps = len(z)
    wp = np.zeros((steps,))
    wa = np.zeros((steps,))
    Qa = np.zeros((steps,))
    Ma = np.zeros((steps,))
    ra = np.zeros((steps,))
    beta = np.zeros((steps,))
    rhob = np.zeros((steps,))
    mp_dot = np.zeros((steps,))
    rp = np.zeros((steps,))
    C = np.zeros((steps,))
    wp[0] = w_p0
    wa[0] = w_a0
    ra[0] = ra_0
    rp[0] = rp_0
    Qa[0] = np.pi*ra[0]**2*wa[0]
    Ma[0] = w_a0*Qa[0]*rho_a
    rhob[0] = rho_p*beta_0+rho_a*(1-beta_0)
    mp_dot[0] = mfr
    C[0] = ((rho_p-rhob[0])/(rho_p-rho_a))**(-4.7)
    beta[0] = beta_0
    Rep = np.zeros((steps,))
    Cd = np.zeros((steps,))
    Fd = np.zeros((steps,))
    B = np.zeros((steps,))
    for i in range(0,len(z)-1):
        i=i+1
        step_size = abs(z[i]-z[i-1])
        Rep[i] = (rho_a*abs(wp[i-1]-wa[i-1])*dp)/mu_a
        Cd[i] = 24*(1+0.15*Rep[i]**.687)/Rep[i]
        Fd[i] = rho_a*(wp[i-1]-wa[i-1])**2/2*Cd[i]/4*np.pi*dp**2*C[i-1]*6*beta[i-1]*wp[i-1]*rp[i-1]**2/dp**3
        B[i] = (rho_p - rho_a)*np.pi*g*beta[i-1]*wp[i-1]*rp[i-1]**2
        wp[i] = wp[i-1] + step_size * (B[i]-Fd[i])/(mp_dot[0]*wp[i-1])
        Qa[i] = Qa[i-1] + step_size * (2*np.pi*ra[i-1]*abs(wa[i-1])*alpha)
        Ma[i] = Ma[i-1] + step_size * (Fd[i])/wp[i]
        wa[i] = Ma[i]/(Qa[i]*rho_a)
        ra[i] = np.sqrt(Qa[i]/(np.pi*abs(wa[i])))

        rp[i] = rp[i-1] + np.diff(ra)[i-1]*wa[i-1]/wp[i-1]
        rhob[i] = mp_dot[0]/(wp[i]*np.pi*rp[i]**2) + rho_a
        
        beta[i] = 1-((rho_p-rhob[i])/(rho_p-rho_a))
        C[i] = ((rho_p-rhob[i])/(rho_p-rho_a))**(-4.7)
        mp_dot[i] = np.pi*(rp[i])**2*rho_p*(wp[i])*(beta[i])
    return(wp,wa,Qa,Ma,ra,rp,rhob,mp_dot)

#%%
r0 = 1.905e-2/2
tau_p = 7.4e-3
w0 = tau_p*9.81
z = np.arange(0,10,1e-5)
rho_p = 2500
rho_f = 1.225
dp = 30e-6
mu_f = 1.825e-5

#%% vary alpha

#alphas = np.linspace(0,1,20)
alphas = np.linspace(0,.01,5) 
alphas = np.append(alphas,np.logspace(-2,0,20))
alphas = np.append(alphas,np.logspace(0,3,25))
alphas = np.append(alphas,np.logspace(3.33,6,5))
mfr = 3e-3 #g/s
rhob_0 = 68#46.7*1.225/1.5
wp_0 = mfr/(rhob_0*np.pi*r0**2)
wf_0 = wp_0-w0/2
bf_0 = r0
vary_a_results = []
for alpha in alphas:
    liu_dn32 = liu_model(wp_0,wf_0,rho_to_phi(rhob_0,rho_p,rho_f),rho_p,rho_f,r0,r0,dp,mu_f,alpha,mfr,z)
    vary_a_results.append(liu_dn32)
    
#%% varying entrainment plot
vary_a_results = np.load('/home/apetersen/Desktop/vary_alpha_results.npy')
cs = plt.get_cmap('sunset')
def _inverse(x):
    #return np.exp(x)
    return(x**10)

def _forward(x):
    #return np.log(x)
    return(x**(1/10))
cNorm = colors.FuncNorm((_forward, _inverse), vmin=0, vmax=1000)

#cNorm = colors.Normalize(vmin=0,vmax=1000)
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax = plt.subplots(figsize=(12,8))
for i in range(0,len(vary_a_results)):
    ax.plot(z[1:]/(r0*2),vary_a_results[i][0][1:]/(tau_p*9.81),color=scalarMap.to_rgba(alphas[i]),lw=2)
ax.set_xlabel(r'$z/D_0$',fontsize=20,labelpad=15);
ax.set_ylabel(r'$W_p/\tau_p g$',fontsize=20,labelpad=15);
cbar=plt.colorbar(scalarMap);
ticks = [0,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,2,5,7,10,15,20,30,50,100,200,500,1000]
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels(["{:.3f}".format(i) for i in ticks])
cbar.set_label(r'$\alpha$',fontsize=20);
plt.tight_layout();

#%% where does wp reach peak?
z_star = []
z_star_loc = []
for i in range(0,len(vary_a_results)):
    wp = vary_a_results[i][0]
    m = np.where(wp==wp.max())[0]
    z_star_loc.append(m[0])
    if m[0]==499999:
        z_star.append(np.inf)
    else:
        z_star.append(z[m[0]]/1.905e-2)
        
# what is concentration at peak
f,ax = plt.subplots(figsize=(12,8))

f2,ax2 = plt.subplots(figsize=(12,8))

for i in range(0,len(vary_a_results)):
    ax.plot(alphas[i],vary_a_results[i][-2][z_star_loc[i]]/vary_a_results[i][-2][0],'ko')#,color=scalarMap.to_rgba(alphas[i]),lw=2)
    ax2.semilogx(alphas[i],vary_a_results[i][-2][z_star_loc[i]]/vary_a_results[i][-2][0],'ko')#,color=scalarMap.to_rgba(alphas[i]),lw=2)

ax.set_xlabel(r'$\alpha$',fontsize=20,labelpad=15);
ax.set_ylabel(r'$\rho_b(z^{*})/\rho_{b}(0)$',fontsize=20,labelpad=15);
ax2.set_xlabel(r'$\alpha$',fontsize=20,labelpad=15);
ax2.set_ylabel(r'$\rho_b(z^{*})/\rho_{b}(0)$',fontsize=20,labelpad=15);
"""cbar=plt.colorbar(scalarMap);
ticks = [0,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,2,5,7,10,15,20,30,50,100,200,500,1000]
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels(["{:.3f}".format(i) for i in ticks])
cbar.set_label(r'$\alpha$',fontsize=20);"""
f.tight_layout()
f2.tight_layout()


#%% vary tau_p

dps = [5e-6,6e-6,7e-6,8e-6,8e-6,10e-6,15e-6,20e-6,25e-6,30e-6,50e-6,75e-6,100e-6,200e-6,300e-6,500e-6,1e-3]
alpha = 0.044
mfr = 3e-3 #g/s
rhob_0 = 68#46.7*1.225/1.5
wp_0 = mfr/(rhob_0*np.pi*r0**2)

bf_0 = r0
vary_dp_results = []
tau_ps = []
unphys = []
for dp in dps:
    tau_p = dp**2*(rho_p-rho_f)/(18*mu_f)
    e =1
    while abs(e)>1e-12:
        Rep = dp*(tau_p*9.81)/(mu_f/rho_f)
        tau_p_temp =dp**2*(rho_p-rho_f)/(18*mu_f*(1+.15*Rep**(.687)))
        e = tau_p - tau_p_temp
        tau_p = tau_p_temp
    tau_ps.append(tau_p)
    w0 = tau_p*9.81
    wf_0 = wp_0-w0/2
    liu_dn32 = liu_model(wp_0,wf_0,rho_to_phi(rhob_0,rho_p,rho_f),rho_p,rho_f,r0,r0,dp,mu_f,alpha,mfr,z)
    vary_dp_results.append(liu_dn32)

#%% varying dp plot
cs = plt.get_cmap('sunset')
cNorm = colors.LogNorm(vmin=1e-6,vmax=np.max(dps))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax = plt.subplots(figsize=(12,8))
for i in range(1,len(vary_dp_results)):
    ax.plot(z/(r0*2),vary_dp_results[i][0]/wp_0,color=scalarMap.to_rgba(dps[i]),lw=2)
ax.set_xlabel(r'$z/D_0$',fontsize=20,labelpad=15);
ax.set_ylabel(r'$W_p/W_0$',fontsize=20,labelpad=15);
cbar=plt.colorbar(scalarMap);
cbar.set_label(r'$d_p$',fontsize=20);
plt.tight_layout();

cs = plt.get_cmap('sunset')
cNorm = colors.LogNorm(vmin=np.min(tau_ps),vmax=np.max(tau_ps)/2)
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax = plt.subplots(figsize=(12,8))
for i in range(0,len(vary_dp_results)):
    ax.plot(z/(r0*2),vary_dp_results[i][0]/wp_0,color=scalarMap.to_rgba(tau_ps[i]),lw=2)
ax.set_xlabel(r'$z/D_0$',fontsize=20,labelpad=15);
ax.set_ylabel(r'$W_p/W_0$',fontsize=20,labelpad=15);
cbar=plt.colorbar(scalarMap);
cbar.set_label(r'$\tau_p$',fontsize=20);
plt.tight_layout();

#%% vary volume fraction for single mp
phi_vs = np.linspace(1e-6,1e-1,15)
mfr = 3e-3 #g/s
dp = 30e-3
rhobs = phi_to_rho(phi_vs,2500,1.225)
alpha = 0.044
tau_p = 7.4e-3
w0 = tau_p*9.81
vary_phi_results = []
for phi_v in phi_vs:
    wp_0 = mfr/(phi_to_rho(phi_v,2500,1.225)*np.pi*r0**2)
    wf_0 = wp_0-w0/2
    liu_dn32 = liu_model(wp_0,wf_0,phi_v,rho_p,rho_f,r0,r0,dp,mu_f,alpha,mfr,z)
    vary_phi_results.append(liu_dn32)
    
#%% varying volume fraction plot
cs = plt.get_cmap('sunset')
cNorm = colors.LogNorm(vmin=np.min(phi_vs),vmax=np.max(phi_vs))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cs)
f,ax = plt.subplots(figsize=(12,8))
for i in range(0,len(vary_phi_results)):
    ax.plot(z/(r0*2),vary_phi_results[i][0]/wp_0,color=scalarMap.to_rgba(phi_vs[i]),lw=2)
ax.set_xlabel(r'$z/D_0$',fontsize=20,labelpad=15);
ax.set_ylabel(r'$W_p/W_0$',fontsize=20,labelpad=15);
cbar=plt.colorbar(scalarMap);
cbar.set_label(r'$\phi_v$',fontsize=20);
plt.tight_layout();
