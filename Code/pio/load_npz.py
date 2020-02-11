#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:50:55 2020

@author: alec
"""
import numpy as np

def load_npz(file_name):
    arr = np.load(file_name)
    array = arr['arr_0']
    return array
