#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:51:22 2020
Model adapted from T.S. van den Bremer & G.R. Hunt
"Universal solutions for Boussinesq and non-Boussinesq plumes", JFM 2010
@author: alec
"""
from scipy.integrate import solve_ivp
import numpy as np

def f(z,G,Gamma_0):
    #G = G.astype('complex')
    dGdz = G*(1-G)*np.sqrt(Gamma_0/G)*((1-G)/(1-Gamma_0))**(3/10)
    return np.real(dGdz)

def lazy_plume_model(z,r0,rho_a,rho_p,beta_0,wp0,alpha=0.03):
    """# Initial parameters
    z -- vector of streamwise distances from plume outlet 
            (zeta = 4*alpha*z/r0)
    r_0 -- initial plume radius
    rho_a -- fluid density
    rho_p -- particle density
    beta_0 -- initial volume fraction
    wp0 --  initial plume velocity
    alpha -- entrainment ratio
    """
    rho_b0 = rho_p*beta_0+rho_a*(1-beta_0)
    Delta_0 = (1-rho_a/rho_b0)/(rho_a/rho_b0)
    Gamma_0 = (5*9.81*(1-rho_b0/rho_a)*(r0)*np.sqrt(rho_b0/rho_a))/(8*alpha*-(wp0)*2*(rho_b0/rho_a))    
    zeta = 4*alpha*z/(r0)
    sol = solve_ivp(lambda z,G: f(z,G,Gamma_0), [zeta[0],zeta[-1]],[Gamma_0],t_eval=zeta)
    #w = np.real(wp0*np.sqrt(Gamma_0/(sol.y.astype('complex')))*((1-(sol.y.astype('complex')))/(1-Gamma_0))**(1/10))
    #r = np.real(r0*np.sqrt((sol.y)/Gamma_0)*((1-Gamma_0)/(1-(sol.y.astype('complex'))))**(3/10))
    #Delta = np.real(Delta_0*np.sqrt(Gamma_0/(sol.y.astype('complex')))*np.sqrt((1-(sol.y.astype('complex')))/(1-Gamma_0)))
    
    w = np.real(wp0*np.sqrt(Gamma_0/(sol.y))*((1-(sol.y))/(1-Gamma_0))**(1/10))
    r = np.real(r0*np.sqrt((sol.y)/Gamma_0)*((1-Gamma_0)/(1-(sol.y)))**(3/10))
    Delta = np.real(Delta_0*np.sqrt(Gamma_0/(sol.y))*np.sqrt((1-(sol.y))/(1-Gamma_0)))
    
    rho_b = rho_a*(Delta+1)
    phi_v = (rho_a-rho_b)/(rho_a-rho_p)
    print(w.shape,r.shape,Delta.shape,Gamma_0,rho_b0,Delta_0)
    return(zeta,w.reshape(w.shape[1],),r.reshape(r.shape[1],),
           rho_b.reshape(rho_b.shape[1],),phi_v.reshape(phi_v.shape[1],))