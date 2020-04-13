#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:07:57 2020
Model adapted from Lai et al. " Spreading Hypothesis of
a Particle Plume", J. Hydraul. Eng., 2016
@author: alec
"""
import numpy as np

def lai_model(z,bf_0,bp_0,wf_0,ws_0,phi_v0,dp,rho_p,rho_f,mu_f=1.825e-5):
    g = 9.81
    steps = len(z)
    bf = np.zeros((steps,))
    bp = np.zeros((steps,))
    wf = np.zeros((steps,))
    wp = np.zeros((steps,))
    ws = np.zeros((steps,))
    phi_v = np.zeros((steps,))
    Mp = np.zeros((steps,))
    Mf = np.zeros((steps,))
    Qf = np.zeros((steps,))
    Qp = np.zeros((steps,))
    
    bf[0] = bf_0
    bp[0] = bp_0
    wf[0] = wf_0
    ws[0] = ws_0
    wp[0] = wf_0 + ws_0
    phi_v[0] = phi_v0
    Qf[0] = np.pi*bf[0]**2*rho_f*wf_0*(1-phi_v0*(bp_0/bf_0)**2/(1+(bp_0/bf_0)**2))
    Qp[:] = np.pi*rho_p*bp_0**2*phi_v0*(wf_0*1/(1+(bp_0/bf_0)**2)+ws_0)
    Mp[0] = np.pi*rho_p*bp[0]**2*phi_v0*(wf_0**2*1/(2+(bp_0/bf_0)**2) +2*wf_0*ws_0*1/(1+(bp_0/bf_0)**2)+ws_0**2)
    Mf[0] = np.pi*rho_f*bf[0]**2*wf[0]**2*(0.5-phi_v[0]*(bp_0/bf_0)**2/(1+2*(bp_0/bf_0)**2))
    
    for i in range(0,len(z)-1):        
        i=i+1
        step_size = abs(z[i]-z[i-1])
        Rep = rho_f*ws[i-1]*dp/mu_f
        Cd = 24/Rep * (1+0.15*Rep**(.687))
        u_avg = wf[i-1]*(0.5 - phi_v[i-1]*(bp[i-1]/bf[i-1])**2/(1+2*(bp[i-1]/bf[i-1])**2))/(1-phi_v[i-1]*(bp[i-1]/bf[i-1])**2/(1+(bp[i-1]/bf[i-1])**2))
        bf[i] = bf[i-1] + 0.11*step_size
        bp[i] = bp[i-1] + 0.11*u_avg/wp[i-1]*step_size
       # phi_v[i] = rho_p*np.pi*(2*bp[0])**2*wp[0]*phi_v[0]/4 * 1/(wf[i-1]*1/(1+(bp[i]/bf[i])**2) + ws[i-1])
        phi_v[i] = Qp[0]/(np.pi*rho_p*bp[i]**2*(wf[i-1]/(1+(bp[i]/bf[i])**2) +ws[i-1]))
        Mp[i] = Mp[i-1] + step_size * (np.pi*bp[i]**2*phi_v[i]*((rho_p-rho_f)*g - 3*rho_f*Cd*ws[i-1]**2/(8*dp/2)))
        Mf[i] = Mf[i-1] + step_size * (np.pi*bp[i-1]**2*phi_v[i-1]*3*rho_f*Cd*ws[i-1]**2/(4*dp))
        
        wf[i] = np.sqrt(Mf[i]/(np.pi*rho_f*bf[i]**2*(0.5 - phi_v[i]*(bp[i-1]/bf[i-1])**2/(1+2*(bp[i-1]/bf[i-1])**2))))
        ws[i] = (Mp[i]/(np.pi*rho_p*bp[i]**2*phi_v[i]) - wf[i]**2/(2+(bp[i]/bf[i])**2)) * 1/(2*wf[i]/(1+(bp[i]/bf[i])**2)+1)
        wp[i] = wf[i]+ws[i]
        Qf[i] = np.pi*rho_f*bf[i]**2*wf[i]*(1-phi_v[i]*(bp[i]/bf[i])**2/(1+(bp[i]/bf[i])**2))
    return(bf,bp,wf,wp,phi_v,Qf)