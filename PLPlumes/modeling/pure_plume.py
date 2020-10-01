#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:16:43 2020
Model adapted from T.S. van den Bremer & G.R. Hunt
"Universal solutions for Boussinesq and non-Boussinesq plumes", JFM 2010
@author: alec
"""

def pure_plume_model(z,r_0,rho_a,rho_p,beta_0,wp0,alpha=0.03):
    """# Initial parameters
    z -- vector of streamwise distances from plume outlet
    r_0 -- initial plume radius
    rho_a -- fluid density
    rho_p -- particle density
    beta_0 -- initial volume fraction
    wp0 --  initial plume velocity
    alpha -- entrainment ratio
    """
    rho_b0 = rho_p#rho_p*beta_0+rho_a*(1-beta_0)
    Delta_0 = (1-rho_a/rho_b0)/(rho_a/rho_b0)
    r_pnb = r_0*3/10*(10/3+(4*alpha*z)/r_0)
    w_pnb = wp0*(10/3)**(1/3)*(10/3+(4*alpha*z)/(r_0))**(-1/3)
    Delta_pnb = Delta_0*(10/3)**(5/3)*(10/3+(4*alpha*z)/r_0)**(-5/3)
    rho_b_pnb = rho_a*(Delta_pnb+1)
    beta_pnb = (rho_a -rho_b_pnb)/(rho_a-rho_p)
    
    return(Delta_pnb,rho_b_pnb,w_pnb,r_pnb,beta_pnb)