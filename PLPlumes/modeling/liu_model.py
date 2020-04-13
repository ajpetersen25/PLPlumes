#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:35:43 2020

@author: ajp25
"""
import numpy as np
from PLPlumes.plume_processing.plume_functions import rho_to_phi

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
        Rep[i] = (rho_a*(wp[i-1]-wa[i-1])*dp)/mu_a
        Cd[i] = 24*(1+0.15*Rep[i]**.687)/Rep[i]
        Fd[i] = rho_a*(wp[i-1]-wa[i-1])**2/2*Cd[i]/4*np.pi*dp**2*C[i-1]*6*beta[i-1]*wp[i-1]*ra[i-1]**2/dp**3
        B[i] = (rho_p - rho_a)*np.pi*g*beta[i-1]*wp[i-1]*ra[i-1]**2
        wp[i] = wp[i-1] + step_size * (B[i]-Fd[i])/(mp_dot[0]*wp[i-1])
        Qa[i] = Qa[i-1] + step_size * (2*np.pi*ra[i-1]*abs(wa[i-1])*alpha)
        Ma[i] = Ma[i-1] + step_size * (Fd[i])/wp[i-1]
        wa[i] = Ma[i]/(Qa[i]*rho_a)
        ra[i] = np.sqrt(Qa[i]/(np.pi*abs(wa[i])))
        #u_avg = wa[i]*(0.5 - rho_to_phi(rhob[i-1],2500,1.225)*(rp[i-1]/ra[i-1])**2/(1+2*(rp[i-1]/ra[i-1])**2))/(1-rho_to_phi(rhob[i-1],2500,1.225)*(rp[i-1]/ra[i-1])**2/(1+(rp[i-1]/ra[i-1])**2))
        #rp[i] = rp[i-1] + np.diff(ra)[i-1]*u_avg/(u_avg+(wp[i]-wa[i]))
        rp[i] = rp[i-1] + np.diff(ra)[i-1]*wa[i]/wp[i]
        rhob[i] = mp_dot[0]/(wp[i]*np.pi*rp[i]**2) + rho_a
        C[i] = ((rho_p-rhob[i])/(rho_p-rho_a))**(-4.7)
        beta[i] = 1-(rho_p-rhob[i])/(rho_p-rho_a)

    return(wp,wa,Qa,Ma,ra,rp,rhob)