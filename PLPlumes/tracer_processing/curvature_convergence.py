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
from PLPlumes.plume_processing.plume_functions import curvature as curvature
import time 
import glob
import matplotlib.pyplot as plt
v1_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View1/*u_e.npy')
v2_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View2/*u_e.npy')
v3_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View3/*u_e.npy')
v4_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View4/*u_e.npy')
v5_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View5/*u_e.npy')
v6_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View6/*u_e.npy')
v7_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View7/*u_e.npy')
v8_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View8/*u_e.npy')

v1_ppi = 412.0012 #pix per in
v1_range_in = np.array([235/v1_ppi, 2560/v1_ppi]) #inches
v1_range = v1_range_in*0.0254 # meters
v1_cal = v1_ppi/.0254 # pix/m


v2_ppi = np.sqrt((2278-1872)**2+(1241-1245)**2) #pix per in
v2_range_in = np.array([7-(2560-2278)/v2_ppi, (7-(2560-2278)/v2_ppi) + (2560/v2_ppi)]) #inches
v2_range = v2_range_in*0.0254 # meters
v2_cal = v2_ppi/.0254 # pix/m


v3_ppi = np.sqrt((2359-1945)**2+(429-430)**2) #pix per in
v3_range_in = np.array([18-6-(2560-2359)/v3_ppi, (18-6-(2560-2359)/v3_ppi) + (2560/v3_ppi)]) #inches
v3_range = v3_range_in*0.0254 # meters
v3_cal = v3_ppi/.0254 # pix/m


v4_ppi = np.sqrt((2325-1915)**2+(419-419)**2) #pix per in
v4_range_in = np.array([24-6-(2560-2325)/v4_ppi, (24-6-(2560-2325)/v4_ppi) + (2560/v4_ppi)]) #inches
v4_range = v4_range_in*0.0254 # meters
v4_cal = v4_ppi/.0254 # pix/m

v5_ppi = np.sqrt((1906-1496)**2+(992-999)**2) #pix per in
v5_range_in = np.array([30-6-(2560-2313)/v5_ppi, (30-6-(2560-2313)/v5_ppi) + (2560/v5_ppi)]) #inches
v5_range = v5_range_in*0.0254 # meters
v5_cal = v5_ppi/.0254 # pix/m

v6_ppi = np.sqrt((1122-1532)**2+(606-600)**2) #pix per in
v6_range_in = np.array([36-6-(2560-2354)/v6_ppi, (36-6-(2560-2354)/v6_ppi) + (2560/v6_ppi)]) #inches
v6_range = v6_range_in*0.0254 # meters
v6_cal = v6_ppi/.0254 # pix/m


v7_ppi = np.sqrt((2171-1771)**2+(891-894)**2) #pix per in
v7_range_in = np.array([49-6-(2560-2171)/v7_ppi, (49-6-(2560-2171)/v7_ppi) + (2560/v7_ppi)]) #inches
v7_range = v7_range_in*0.0254 # meters
v7_cal = v7_ppi/.0254 # pix/m

v8_ppi = np.sqrt((2453-2061)**2+(1260-1263)**2) #pix per in
v8_range_in = np.array([57-6-(2560-2453)/v8_ppi, (57-6-(2560-2453)/v8_ppi) + (2560/v8_ppi)]) #inches
v8_range = v8_range_in*0.0254 # meters
v8_cal = v8_ppi/.0254 # pix/m

#%% View 1 single convergence test
ctests = []
positions = [20,1000,2000]
for v in v1_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_k = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            pts = v1[f][:,0:2]
            z = 2560 - pts[:,0][::-1]
            r = pts[:,1][::-1] - (1600-1532)
            k = curvature(z,r,pts)[position]
            all_k.append(k)
        ctests.append(all_k)
     
ctests = np.array(ctests)
v1_convergence_test1 = convergence(ctests[:,0])
v1_convergence_test2 = convergence(ctests[:,1])
v1_convergence_test3 = convergence(ctests[:,2])
plt.figure();plt.plot(100*(1 - v1_convergence_test1/(v1_convergence_test1[-1])),'k');
plt.plot(100*(1 - v1_convergence_test2/(v1_convergence_test2[-1])),'r');
plt.plot(100*(1 - v1_convergence_test3/(v1_convergence_test3[-1])),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);
#%% View 2 convergence tests
ctests = []
positions = [20,1000,2000]
for v in v2_list:
    v1 = np.load(v,allow_pickle=True)
    for f in range(0,len(v1)):
        all_k = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            pts = v1[f][:,0:2]
            z = 2560 - pts[:,0][::-1]
            r = pts[:,1][::-1] - (1600-1450)
            k = curvature(z,r,pts)[position]
            all_k.append(k)
        ctests.append(all_k)
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
        all_k = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            pts = v1[f][:,0:2]
            z = 2560 - pts[:,0][::-1]
            r = pts[:,1][::-1] + (3*.0254)*v3_cal - 799
            k = curvature(z,r,pts)[position]
            all_k.append(k)
        ctests.append(all_k)
     
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
        all_k = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            pts = v1[f][:,0:2]
            z = 2560 - pts[:,0][::-1]
            r = pts[:,1][::-1] + (3*.0254)*v4_cal - 783
            k = curvature(z,r,pts)[position]
            all_k.append(k)
        ctests.append(all_k)
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
        all_k = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            pts = v1[f][:,0:2]
            z = 2560 - pts[:,0][::-1]
            r = pts[:,1][::-1] + (4.5*.0254)*v5_cal - 1002
            k = curvature(z,r,pts)[position]
            all_k.append(k)
        ctests.append(all_k)
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
        all_k = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            pts = v1[f][:,0:2]
            z = 2560 - pts[:,0][::-1]
            r = pts[:,1][::-1] + (4.5*.0254)*v6_cal - 967
            k = curvature(z,r,pts)[position]
            all_k.append(k)
        ctests.append(all_k)
     
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
        all_k = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            pts = v1[f][:,0:2]
            z = 2560 - pts[:,0][::-1]
            r = pts[:,1][::-1] + (4.6*.0254)*v7_cal - 899
            k = curvature(z,r,pts)[position]
            all_k.append(k)
        ctests.append(all_k)
     
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
        all_k = []
        for p in positions:
            d = np.abs(v1[f][:,0]-p)
            position = np.where(d==np.min(d))[0][0]
            pts = v1[f][:,0:2]
            z = 2560 - pts[:,0][::-1]
            r = pts[:,1][::-1] + (4.6*.0254)*v8_cal - 918
            k = curvature(z,r,pts)[position]
            all_k.append(k)
        ctests.append(all_k)
     
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
