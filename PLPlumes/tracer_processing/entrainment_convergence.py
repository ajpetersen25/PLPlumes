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
v1_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View1/*u_e.npy')
v2_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View2/*u_e.npy')
v3_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View3/*u_e.npy')
v4_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View4/*u_e.npy')
v5_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View5/*u_e.npy')
v6_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View6/*u_e.npy')
v7_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View7/*u_e.npy')
v8_list = glob.glob('/media/apetersen/Backup4/P2_entrainment/View8/*u_e.npy')

#%% View 1 single convergence test
num_bins = 2560/16
ctests = []
positions = np.array([160/16, 1000/16, 2400/16], dtype=int)
for v in v1_list:
    v1 = np.load(v, allow_pickle=True)
    tic = time.time()

    for f in range(0, len(v1)):
        all_ue = []
        all_ue_counts = []
        pts = v1[f][:, 0:2]
        uf = v1[f][:, 2:4]
        up = v1[f][:, 4:6]
        n = v1[f][:, 6:8]
        ufmag = -np.sum(uf*n, axis=1)
        upmag = -np.sum(up*n, axis=1)
        ue = ufmag/3 - upmag
        ue_binned = scipy.stats.binned_statistic(
            pts[:, 0], ue, statistic=np.nansum, bins=num_bins, range=(0, 2560)).statistic
        ue_c = scipy.stats.binned_statistic(pts[:, 0], ~np.isnan(
            ue), statistic=np.sum, bins=num_bins, range=(0, 2560)).statistic

        for position in positions:
            all_ue.append(ue_binned[position])
            all_ue_counts.append(ue_c[position])
        ctests.append(np.array(all_ue)/np.array(all_ue_counts))

ctests = np.array(ctests)
v1_convergence_test1 = convergence(ctests[:, 0])[::10]
v1_convergence_test2 = convergence(ctests[:, 1])[::10]
v1_convergence_test3 = convergence(ctests[:, 2])[::10]
"""
plt.figure();plt.plot(np.linspace(0,len(ctests[:,0]),len(v1_convergence_test1)),100*(v1_convergence_test1-v1_convergence_test1[-1])/(v1_convergence_test1[-1]),'k');
plt.plot(np.linspace(0,len(ctests[:,1]),len(v1_convergence_test1)),100*(v1_convergence_test2-v1_convergence_test2[-1])/(v1_convergence_test2[-1]),'r');
plt.plot(np.linspace(0,len(ctests[:,2]),len(v1_convergence_test1)),100*(v1_convergence_test3-v1_convergence_test3[-1])/(v1_convergence_test3[-1]),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);"""
plt.figure()
plt.plot(np.linspace(0, len(ctests[:, 0]), len(
    v1_convergence_test1)), (v1_convergence_test1), 'k')
plt.plot(np.linspace(0, len(ctests[:, 1]), len(
    v1_convergence_test1)), (v1_convergence_test2), 'r')
plt.plot(np.linspace(0, len(ctests[:, 2]), len(
    v1_convergence_test1)), (v1_convergence_test3), 'b')
plt.xlabel('frames', fontsize=15, labelpad=15)
plt.ylabel(r'$u_e$', fontsize=15, labelpad=15)
plt.axvline(12466)
plt.axvline(12466*2)
plt.axvline(12466*3)
plt.axvline(12466*4)
plt.axvline(12466*5)
plt.title(r'$V_1$ $\langle u_e \rangle$', fontsize=20, y=1.03)
#%% View 2 convergence tests
ctests = []
positions = np.array([160/16, 1000/16, 2400/16], dtype=int)
for v in v2_list:
    v1 = np.load(v, allow_pickle=True)
    tic = time.time()

    for f in range(0, len(v1)):
        all_ue = []
        all_ue_counts = []
        pts = v1[f][:, 0:2]
        uf = v1[f][:, 2:4]
        up = v1[f][:, 4:6]
        n = v1[f][:, 6:8]
        ufmag = -np.sum(uf*n, axis=1)
        upmag = -np.sum(up*n, axis=1)
        ue = ufmag/3 - upmag
        ue_binned = scipy.stats.binned_statistic(
            pts[:, 0], ue, statistic=np.nansum, bins=num_bins, range=(0, 2560)).statistic
        ue_c = scipy.stats.binned_statistic(pts[:, 0], ~np.isnan(
            ue), statistic=np.sum, bins=num_bins, range=(0, 2560)).statistic

        for position in positions:
            all_ue.append(ue_binned[position])
            all_ue_counts.append(ue_c[position])
        ctests.append(np.array(all_ue)/np.array(all_ue_counts))
ctests = np.array(ctests)
v2_convergence_test1 = convergence(ctests[:, 0])[::10]
v2_convergence_test2 = convergence(ctests[:, 1])[::10]
v2_convergence_test3 = convergence(ctests[:, 2])[::10]
"""
plt.figure();plt.plot(np.linspace(0,len(ctests[:,0]),len(v2_convergence_test1)),100*(v2_convergence_test1-v2_convergence_test1[-1])/(v2_convergence_test1[-1]),'k');
plt.plot(np.linspace(0,len(ctests[:,1]),len(v2_convergence_test1)),100*(v2_convergence_test2-v2_convergence_test2[-1])/(v2_convergence_test2[-1]),'r');
plt.plot(np.linspace(0,len(ctests[:,2]),len(v2_convergence_test1)),100*(v2_convergence_test3-v2_convergence_test3[-1])/(v2_convergence_test3[-1]),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);"""
plt.figure()
plt.plot(np.linspace(0, len(ctests[:, 0]), len(
    v2_convergence_test1)), (v2_convergence_test1), 'k')
plt.plot(np.linspace(0, len(ctests[:, 1]), len(
    v2_convergence_test1)), (v2_convergence_test2), 'r')
plt.plot(np.linspace(0, len(ctests[:, 2]), len(
    v2_convergence_test1)), (v2_convergence_test3), 'b')
plt.xlabel('frames', fontsize=15, labelpad=15)
plt.ylabel(r'$u_e$', fontsize=15, labelpad=15)
plt.axvline(12466)
plt.axvline(12466*2)
plt.axvline(12466*3)
plt.axvline(12466*4)
plt.axvline(12466*5)
plt.title(r'$V_2$ $\langle u_e \rangle$', fontsize=20, y=1.03)
#%% View 3 convergence tests
ctests = []
positions = np.array([160/16, 1000/16, 2400/16], dtype=int)
for v in v3_list:
    v1 = np.load(v, allow_pickle=True)
    tic = time.time()

    for f in range(0, len(v1)):
        all_ue = []
        all_ue_counts = []
        pts = v1[f][:, 0:2]
        uf = v1[f][:, 2:4]
        up = v1[f][:, 4:6]
        n = v1[f][:, 6:8]
        ufmag = -np.sum(uf*n, axis=1)
        upmag = -np.sum(up*n, axis=1)
        ue = ufmag/3 - upmag
        ue_binned = scipy.stats.binned_statistic(
            pts[:, 0], ue, statistic=np.nansum, bins=num_bins, range=(0, 2560)).statistic
        ue_c = scipy.stats.binned_statistic(pts[:, 0], ~np.isnan(
            ue), statistic=np.sum, bins=num_bins, range=(0, 2560)).statistic

        for position in positions:
            all_ue.append(ue_binned[position])
            all_ue_counts.append(ue_c[position])
        ctests.append(np.array(all_ue)/np.array(all_ue_counts))
ctests = np.array(ctests)

v3_convergence_test1 = convergence(ctests[:, 0])[::10]
v3_convergence_test2 = convergence(ctests[:, 1])[::10]
v3_convergence_test3 = convergence(ctests[:, 2])[::10]
"""plt.figure();plt.plot(np.linspace(0,len(ctests[:,0]),len(v3_convergence_test1)),100*(v3_convergence_test1-v3_convergence_test1[-1])/(v3_convergence_test1[-1]),'k');
plt.plot(np.linspace(0,len(ctests[:,1]),len(v3_convergence_test1)),100*(v3_convergence_test2-v3_convergence_test2[-1])/(v3_convergence_test2[-1]),'r');
plt.plot(np.linspace(0,len(ctests[:,2]),len(v3_convergence_test1)),100*(v3_convergence_test3-v3_convergence_test3[-1])/(v3_convergence_test3[-1]),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);"""
plt.figure()
plt.plot(np.linspace(0, len(ctests[:, 0]), len(
    v3_convergence_test1)), (v3_convergence_test1), 'k')
plt.plot(np.linspace(0, len(ctests[:, 1]), len(
    v3_convergence_test1)), (v3_convergence_test2), 'r')
plt.plot(np.linspace(0, len(ctests[:, 2]), len(
    v3_convergence_test1)), (v3_convergence_test3), 'b')
plt.xlabel('frames', fontsize=15, labelpad=15)
plt.ylabel(r'$u_e$', fontsize=15, labelpad=15)
plt.axvline(12466)
plt.axvline(12466*2)
plt.axvline(12466*3)
plt.axvline(12466*4)
plt.axvline(12466*5)
plt.title(r'$V_3$ $\langle u_e \rangle$', fontsize=20, y=1.03)
#%% View 4 convergence tests
ctests = []
positions = np.array([160/16, 1000/16, 2400/16], dtype=int)
for v in v4_list:
    v1 = np.load(v, allow_pickle=True)
    tic = time.time()

    for f in range(0, len(v1)):
        all_ue = []
        all_ue_counts = []
        pts = v1[f][:, 0:2]
        uf = v1[f][:, 2:4]
        up = v1[f][:, 4:6]
        n = v1[f][:, 6:8]
        ufmag = -np.sum(uf*n, axis=1)
        upmag = -np.sum(up*n, axis=1)
        ue = ufmag/3 - upmag
        ue_binned = scipy.stats.binned_statistic(
            pts[:, 0], ue, statistic=np.nansum, bins=num_bins, range=(0, 2560)).statistic
        ue_c = scipy.stats.binned_statistic(pts[:, 0], ~np.isnan(
            ue), statistic=np.sum, bins=num_bins, range=(0, 2560)).statistic

        for position in positions:
            all_ue.append(ue_binned[position])
            all_ue_counts.append(ue_c[position])
        ctests.append(np.array(all_ue)/np.array(all_ue_counts))
ctests = np.array(ctests)
v4_convergence_test1 = convergence(ctests[:, 0])[::10]
v4_convergence_test2 = convergence(ctests[:, 1])[::10]
v4_convergence_test3 = convergence(ctests[:, 2])[::10]
"""plt.figure();plt.plot(np.linspace(0,len(ctests[:,0]),len(v4_convergence_test1)),100*(v4_convergence_test1-v4_convergence_test1[-1])/(v4_convergence_test1[-1]),'k');
plt.plot(np.linspace(0,len(ctests[:,1]),len(v4_convergence_test1)),100*(v4_convergence_test2-v4_convergence_test2[-1])/(v4_convergence_test2[-1]),'r');
plt.plot(np.linspace(0,len(ctests[:,2]),len(v4_convergence_test1)),100*(v4_convergence_test3-v4_convergence_test3[-1])/(v4_convergence_test3[-1]),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);"""
plt.figure()
plt.plot(np.linspace(0, len(ctests[:, 0]), len(
    v4_convergence_test1)), (v4_convergence_test1), 'k')
plt.plot(np.linspace(0, len(ctests[:, 1]), len(
    v4_convergence_test1)), (v4_convergence_test2), 'r')
plt.plot(np.linspace(0, len(ctests[:, 2]), len(
    v4_convergence_test1)), (v4_convergence_test3), 'b')
plt.xlabel('frames', fontsize=15, labelpad=15)
plt.ylabel(r'$u_e$', fontsize=15, labelpad=15)
plt.axvline(12466)
plt.axvline(12466*2)
plt.axvline(12466*3)
plt.axvline(12466*4)
plt.axvline(12466*5)
plt.title(r'$V_4$ $\langle u_e \rangle$', fontsize=20, y=1.03)
#%% View 5 convergence tests
ctests = []
positions = np.array([160/16, 1000/16, 2400/16], dtype=int)
for v in v5_list:
    v1 = np.load(v, allow_pickle=True)
    tic = time.time()

    for f in range(0, len(v1)):
        all_ue = []
        all_ue_counts = []
        pts = v1[f][:, 0:2]
        uf = v1[f][:, 2:4]
        up = v1[f][:, 4:6]
        n = v1[f][:, 6:8]
        ufmag = -np.sum(uf*n, axis=1)
        upmag = -np.sum(up*n, axis=1)
        ue = ufmag/3 - upmag
        ue_binned = scipy.stats.binned_statistic(
            pts[:, 0], ue, statistic=np.nansum, bins=num_bins, range=(0, 2560)).statistic
        ue_c = scipy.stats.binned_statistic(pts[:, 0], ~np.isnan(
            ue), statistic=np.sum, bins=num_bins, range=(0, 2560)).statistic

        for position in positions:
            all_ue.append(ue_binned[position])
            all_ue_counts.append(ue_c[position])
        ctests.append(np.array(all_ue)/np.array(all_ue_counts))
ctests = np.array(ctests)
v5_convergence_test1 = convergence(ctests[:, 0])[::10]
v5_convergence_test2 = convergence(ctests[:, 1])[::10]
v5_convergence_test3 = convergence(ctests[:, 2])[::10]
"""plt.figure();plt.plot(np.linspace(0,len(ctests[:,0]),len(v5_convergence_test1)),100*(v5_convergence_test1-v5_convergence_test1[-1])/(v5_convergence_test1[-1]),'k');
plt.plot(np.linspace(0,len(ctests[:,1]),len(v5_convergence_test1)),100*(v5_convergence_test2-v5_convergence_test2[-1])/(v5_convergence_test2[-1]),'r');
plt.plot(np.linspace(0,len(ctests[:,2]),len(v5_convergence_test1)),100*(v5_convergence_test3-v5_convergence_test3[-1])/(v5_convergence_test3[-1]),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);"""
plt.figure()
plt.plot(np.linspace(0, len(ctests[:, 0]), len(
    v5_convergence_test1)), (v5_convergence_test1), 'k')
plt.plot(np.linspace(0, len(ctests[:, 1]), len(
    v5_convergence_test1)), (v5_convergence_test2), 'r')
plt.plot(np.linspace(0, len(ctests[:, 2]), len(
    v5_convergence_test1)), (v5_convergence_test3), 'b')
plt.xlabel('frames', fontsize=15, labelpad=15)
plt.ylabel(r'$u_e$', fontsize=15, labelpad=15)
plt.axvline(12466)
plt.axvline(12466*2)
plt.axvline(12466*3)
plt.axvline(12466*4)
plt.axvline(12466*5)
plt.title(r'$V_5$ $\langle u_e \rangle$', fontsize=20, y=1.03)
#%% View 6 convergence tests
ctests = []
positions = np.array([160/16, 1000/16, 2400/16], dtype=int)
for v in v6_list:
    v1 = np.load(v, allow_pickle=True)
    tic = time.time()

    for f in range(0, len(v1)):
        all_ue = []
        all_ue_counts = []
        pts = v1[f][:, 0:2]
        uf = v1[f][:, 2:4]
        up = v1[f][:, 4:6]
        n = v1[f][:, 6:8]
        ufmag = -np.sum(uf*n, axis=1)
        upmag = -np.sum(up*n, axis=1)
        ue = ufmag/3 - upmag
        ue_binned = scipy.stats.binned_statistic(
            pts[:, 0], ue, statistic=np.nansum, bins=num_bins, range=(0, 2560)).statistic
        ue_c = scipy.stats.binned_statistic(pts[:, 0], ~np.isnan(
            ue), statistic=np.sum, bins=num_bins, range=(0, 2560)).statistic

        for position in positions:
            all_ue.append(ue_binned[position])
            all_ue_counts.append(ue_c[position])
        ctests.append(np.array(all_ue)/np.array(all_ue_counts))
ctests = np.array(ctests)
v6_convergence_test1 = convergence(ctests[:, 0])[::10]
v6_convergence_test2 = convergence(ctests[:, 1])[::10]
v6_convergence_test3 = convergence(ctests[:, 2])[::10]
"""plt.figure();plt.plot(np.linspace(0,len(ctests[:,0]),len(v6_convergence_test1)),100*(v6_convergence_test1-v6_convergence_test1[-1])/(v6_convergence_test1[-1]),'k');
plt.plot(np.linspace(0,len(ctests[:,1]),len(v6_convergence_test1)),100*(v6_convergence_test2-v6_convergence_test2[-1])/(v6_convergence_test2[-1]),'r');
plt.plot(np.linspace(0,len(ctests[:,2]),len(v6_convergence_test1)),100*(v6_convergence_test3-v6_convergence_test3[-1])/(v6_convergence_test3[-1]),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);"""
plt.figure()
plt.plot(np.linspace(0, len(ctests[:, 0]), len(
    v6_convergence_test1)), (v6_convergence_test1), 'k')
plt.plot(np.linspace(0, len(ctests[:, 1]), len(
    v6_convergence_test1)), (v6_convergence_test2), 'r')
plt.plot(np.linspace(0, len(ctests[:, 2]), len(
    v6_convergence_test1)), (v6_convergence_test3), 'b')
plt.xlabel('frames', fontsize=15, labelpad=15)
plt.ylabel(r'$u_e$', fontsize=15, labelpad=15)
plt.axvline(12466)
plt.axvline(12466*2)
plt.axvline(12466*3)
plt.axvline(12466*4)
plt.axvline(12466*5)
plt.title(r'$V_6$ $\langle u_e \rangle$', fontsize=20, y=1.03)
#%% View 7 convergence tests
ctests = []
positions = np.array([160/16, 1000/16, 2400/16], dtype=int)
for v in v7_list:
    v1 = np.load(v, allow_pickle=True)
    tic = time.time()

    for f in range(0, len(v1)):
        all_ue = []
        all_ue_counts = []
        pts = v1[f][:, 0:2]
        uf = v1[f][:, 2:4]
        up = v1[f][:, 4:6]
        n = v1[f][:, 6:8]
        ufmag = -np.sum(uf*n, axis=1)
        upmag = -np.sum(up*n, axis=1)
        ue = ufmag/3 - upmag
        ue_binned = scipy.stats.binned_statistic(
            pts[:, 0], ue, statistic=np.nansum, bins=num_bins, range=(0, 2560)).statistic
        ue_c = scipy.stats.binned_statistic(pts[:, 0], ~np.isnan(
            ue), statistic=np.sum, bins=num_bins, range=(0, 2560)).statistic

        for position in positions:
            all_ue.append(ue_binned[position])
            all_ue_counts.append(ue_c[position])
        ctests.append(np.array(all_ue)/np.array(all_ue_counts))
ctests = np.array(ctests)
v7_convergence_test1 = convergence(ctests[:, 0])[::10]
v7_convergence_test2 = convergence(ctests[:, 1])[::10]
v7_convergence_test3 = convergence(ctests[:, 2])[::10]
"""plt.figure();plt.plot(100*(np.linspace(0,len(ctests[:,0]),len(v7_convergence_test1)),v7_convergence_test1-v7_convergence_test1[-1])/(v7_convergence_test1[-1]),'k');
plt.plot(np.linspace(0,len(ctests[:,1]),len(v7_convergence_test1)),100*(v7_convergence_test2-v7_convergence_test2[-1])/(v7_convergence_test2[-1]),'r');
plt.plot(np.linspace(0,len(ctests[:,2]),len(v7_convergence_test1)),100*(v7_convergence_test3-v7_convergence_test3[-1])/(v7_convergence_test3[-1]),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);"""
plt.figure()
plt.plot(np.linspace(0, len(ctests[:, 0]), len(
    v7_convergence_test1)), (v7_convergence_test1), 'k')
plt.plot(np.linspace(0, len(ctests[:, 1]), len(
    v7_convergence_test1)), (v7_convergence_test2), 'r')
plt.plot(np.linspace(0, len(ctests[:, 2]), len(
    v7_convergence_test1)), (v7_convergence_test3), 'b')
plt.xlabel('frames', fontsize=15, labelpad=15)
plt.ylabel(r'$u_e$', fontsize=15, labelpad=15)
plt.axvline(12466)
plt.axvline(12466*2)
plt.axvline(12466*3)
plt.axvline(12466*4)
plt.axvline(12466*5)
plt.title(r'$V_7$ $\langle u_e \rangle$', fontsize=20, y=1.03)

#%% View 8 convergence tests
ctests = []
positions = np.array([160/16, 1000/16, 2400/16], dtype=int)
for v in v8_list:
    v1 = np.load(v, allow_pickle=True)
    tic = time.time()

    for f in range(0, len(v1)):
        all_ue = []
        all_ue_counts = []
        pts = v1[f][:, 0:2]
        uf = v1[f][:, 2:4]
        up = v1[f][:, 4:6]
        n = v1[f][:, 6:8]
        ufmag = -np.sum(uf*n, axis=1)
        upmag = -np.sum(up*n, axis=1)
        ue = ufmag/3 - upmag
        ue_binned = scipy.stats.binned_statistic(
            pts[:, 0], ue, statistic=np.nansum, bins=num_bins, range=(0, 2560)).statistic
        ue_c = scipy.stats.binned_statistic(pts[:, 0], ~np.isnan(
            ue), statistic=np.sum, bins=num_bins, range=(0, 2560)).statistic

        for position in positions:
            all_ue.append(ue_binned[position])
            all_ue_counts.append(ue_c[position])
        ctests.append(np.array(all_ue)/np.array(all_ue_counts))
ctests = np.array(ctests)
v8_convergence_test1 = convergence(ctests[:, 0])[::10]
v8_convergence_test2 = convergence(ctests[:, 1])[::10]
v8_convergence_test3 = convergence(ctests[:, 2])[::10]
"""plt.figure();plt.plot(np.linspace(0,len(ctests[:,0]),len(v8_convergence_test1)),100*(v8_convergence_test1-v8_convergence_test1[-1])/(v8_convergence_test1[-1]),'k');
plt.plot(np.linspace(0,len(ctests[:,1]),len(v8_convergence_test2)),100*(v8_convergence_test2-v8_convergence_test2[-1])/(v8_convergence_test2[-1]),'r');
plt.plot(np.linspace(0,len(ctests[:,2]),len(v8_convergence_test3)),100*(v8_convergence_test3-v8_convergence_test3[-1])/(v8_convergence_test3[-1]),'b');
plt.xlabel('frames',fontsize=15,labelpad=15);plt.ylabel(r'% change',fontsize=15,labelpad=15)
plt.axvline(12466);plt.axvline(12466*2);plt.axvline(12466*3);
plt.axvline(12466*4);plt.axvline(12466*5);"""
plt.figure()
plt.plot(np.linspace(0, len(ctests[:, 0]), len(
    v8_convergence_test1)), (v8_convergence_test1), 'k')
plt.plot(np.linspace(0, len(ctests[:, 1]), len(
    v8_convergence_test1)), (v8_convergence_test2), 'r')
plt.plot(np.linspace(0, len(ctests[:, 2]), len(
    v8_convergence_test1)), (v8_convergence_test3), 'b')
plt.xlabel('frames', fontsize=15, labelpad=15)
plt.ylabel(r'$u_e$', fontsize=15, labelpad=15)
plt.axvline(12466)
plt.axvline(12466*2)
plt.axvline(12466*3)
plt.axvline(12466*4)
plt.axvline(12466*5)
plt.title(r'$V_8$ $\langle u_e \rangle$', fontsize=20, y=1.03)
