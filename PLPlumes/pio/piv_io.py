#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:51:25 2020

@author: alec
"""

import numpy as np

def load_piv(piv_txt_file,piv_arr_shape,full=False):
    piv = np.loadtxt(piv_txt_file)
    if full==True:
        x = piv[:,0].reshape(piv_arr_shape)
        y = piv[:,1].reshape(piv_arr_shape)
        u = piv[:,2].reshape(piv_arr_shape)
        v = piv[:,3].reshape(piv_arr_shape)
        mask = piv[:,4].reshape(piv_arr_shape)
        return(x,y,u,v,mask)
    if full==False:
        u = piv[:,2].reshape(piv_arr_shape)
        v = piv[:,3].reshape(piv_arr_shape)
        mask = piv[:,4].reshape(piv_arr_shape)
        return(u,v,mask)