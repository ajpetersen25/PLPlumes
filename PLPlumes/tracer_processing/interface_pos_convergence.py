#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:48:36 2021

@author: apetersen
"""
#%%
import numpy as np
import scipy.stats
import h5py
from PLPlumes.pio import pivio, imgio
from PLPlumes.plume_processing.plume_functions import windowed_average as windowed_average
from PLPlumes.plume_processing.plume_functions import convergence as convergence
import time 
import glob
import matplotlib.pyplot as plt
v1_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View1/*.hdf5')
v2_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View2/*.hdf5')
v3_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View3/*.hdf5')
v4_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View4/*.hdf5')
v5_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View5/*.hdf5')
v6_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View6/*.hdf5')
v7_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View7/*.hdf5')
v8_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View8/*.hdf5')

#%% View 1 single convergence test
ctests = []
positions = [20,1000,2000]
for v in v1_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_p = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0]
            pts = v1[f][position[0],0:2]
            all_p.append(pts[1])
        ctests.append(all_p)
        
ctests = np.array(ctests)
v1_convergence_test1 = convergence(ctests[:,0])
v1_convergence_test2 = convergence(ctests[:,1])
v1_convergence_test3 = convergence(ctests[:,2])
plt.figure();plt.plot(abs(1 - v1_convergence_test1/(v1_convergence_test1[-1]) )* 100,'k')
plt.plot(100*(1 - v1_convergence_test2/(v1_convergence_test2[-1])),'r');
plt.plot(100*(1 - v1_convergence_test3/(v1_convergence_test3[-1])),'b');
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466*4);plt.axvline(12466*5);
#%% View 2 convergence tests
ctests = []
positions = [20,1000,2000]
for v in v2_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_p = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0]
            pts = v1[f][position[0],0:2]
            all_p.append(pts[1])
        ctests.append(all_p)
     
ctests = np.array(ctests)
v2_convergence_test1 = convergence(ctests[:,0])
v2_convergence_test2 = convergence(ctests[:,1])
v2_convergence_test3 = convergence(ctests[:,2])
plt.figure();plt.plot(100*(1 - v2_convergence_test1/(v2_convergence_test1[-1])),'k');
plt.plot(100*(1 - v2_convergence_test2/(v2_convergence_test2[-1])),'r');
plt.plot(100*(1 - v2_convergence_test3/(v2_convergence_test3[-1])),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);
#%% View 3 convergence tests
ctests = []
positions = [20,1000,2000]
for v in v3_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_p = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0]            
            pts = v1[f][position[0],0:2]
            all_p.append(pts[1])
        ctests.append(all_p)
     
ctests = np.array(ctests)
v3_convergence_test1 = convergence(ctests[:,0])
v3_convergence_test2 = convergence(ctests[:,1])
v3_convergence_test3 = convergence(ctests[:,2])
plt.figure();plt.plot(100*(1 - v3_convergence_test1/(v3_convergence_test1[-1])),'k');
plt.plot(100*(1 - v3_convergence_test2/(v3_convergence_test2[-1])),'r');
plt.plot(100*(1 - v3_convergence_test3/(v3_convergence_test3[-1])),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);
#%% View 4 convergence tests
ctests = []
positions = [20,1000,2000]
for v in v4_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_p = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0]
            pts = v1[f][position[0],0:2]
            all_p.append(pts[1])
        ctests.append(all_p)
     
ctests = np.array(ctests)
v4_convergence_test1 = convergence(ctests[:,0])
v4_convergence_test2 = convergence(ctests[:,1])
v4_convergence_test3 = convergence(ctests[:,2])
plt.figure();plt.plot(100*(1 - v4_convergence_test1/(v4_convergence_test1[-1])),'k');
plt.plot(100*(1 - v4_convergence_test2/(v4_convergence_test2[-1])),'r');
plt.plot(100*(1 - v4_convergence_test3/(v4_convergence_test3[-1])),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);
#%% View 5 convergence tests
ctests = []
positions = [20,1000,2000]
for v in v5_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_p = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0]
            pts = v1[f][position[0],0:2]
            all_p.append(pts[1])
        ctests.append(all_p)
     
ctests = np.array(ctests)
v5_convergence_test1 = convergence(ctests[:,0])
v5_convergence_test2 = convergence(ctests[:,1])
v5_convergence_test3 = convergence(ctests[:,2])
plt.figure();plt.plot(100*(1 - v5_convergence_test1/(v5_convergence_test1[-1])),'k');
plt.plot(100*(1 - v5_convergence_test2/(v5_convergence_test2[-1])),'r');
plt.plot(100*(1 - v5_convergence_test3/(v5_convergence_test3[-1])),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);

#%% View 6 convergence tests
ctests = []
positions = [20,1000,2000]
for v in v6_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_p = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0]
            pts = v1[f][position[0],0:2]
            all_p.append(pts[1])
        ctests.append(all_p)
     
ctests = np.array(ctests)
v6_convergence_test1 = convergence(ctests[:,0])
v6_convergence_test2 = convergence(ctests[:,1])
v6_convergence_test3 = convergence(ctests[:,2])
plt.figure();plt.plot(100*(1 - v6_convergence_test1/(v6_convergence_test1[-1])),'k');
plt.plot(100*(1 - v6_convergence_test2/(v6_convergence_test2[-1])),'r');
plt.plot(100*(1 - v6_convergence_test3/(v6_convergence_test3[-1])),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);
#%% View 7 convergence tests
ctests = []
positions = [20,1000,2000]
for v in v7_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_p = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0]
            pts = v1[f][position[0],0:2]
            all_p.append(pts[1])
        ctests.append(all_p)
     
ctests = np.array(ctests)
v7_convergence_test1 = convergence(ctests[:,0])
v7_convergence_test2 = convergence(ctests[:,1])
v7_convergence_test3 = convergence(ctests[:,2])
plt.figure();plt.plot(100*(1 - v7_convergence_test1/(v7_convergence_test1[-1])),'k');
plt.plot(100*(1 - v7_convergence_test2/(v7_convergence_test2[-1])),'r');
plt.plot(100*(1 - v7_convergence_test3/(v7_convergence_test3[-1])),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);
#%% View 8 convergence tests
ctests = []
positions = [20,1000,2000]
for v in v8_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_p = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0]
            pts = v1[f][position[0],0:2]
            all_p.append(pts[1])
        ctests.append(all_p)
     
ctests = np.array(ctests)
v8_convergence_test1 = convergence(ctests[:,0])
v8_convergence_test2 = convergence(ctests[:,1])
v8_convergence_test3 = convergence(ctests[:,2])
plt.figure();plt.plot(100*(1 - v8_convergence_test1/(v8_convergence_test1[-1])),'k');
plt.plot(100*(1 - v8_convergence_test2/(v8_convergence_test2[-1])),'r');
plt.plot(100*(1 - v8_convergence_test3/(v8_convergence_test3[-1])),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);
