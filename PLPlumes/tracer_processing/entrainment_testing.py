import numpy as np
import numpy.ma as nma
from scipy.ndimage import median_filter, gaussian_filter
from skimage.morphology import dilation,square,binary_erosion
import scipy.ndimage.measurements as measurements
from PLPlumes.pio import pivio,imgio
from scipy import interpolate
from scipy.ndimage import map_coordinates
import time
import argparse
import multiprocessing as mp
from itertools import repeat
import copy
import getpass
from datetime import datetime
import matplotlib.pyplot as plt
import os
from scipy.integrate import simps

def q_at_p(points, q, dstep):
    """
    Inputs:
    points --- (n,2) array of positions in pixel units
    q      --- 2d array of fluid field values
    dstep  --- int, pixel spacing between piv vectors
    Returns:
    q_at_ps --- n,1 array of interpolated values at points
    """
    X,Y = np.meshgrid(dstep * np.arange(0,q.shape[1]) + dstep,dstep * np.arange(0,q.shape[0]) + dstep)
    xi=X.ravel()/dstep
    yi=Y.ravel()/dstep
    z = q.ravel()
    #xi=xi[~np.isnan(z)]
    #yi=yi[~np.isnan(z)]
    xi=xi.reshape(len(xi),1)
    yi=yi.reshape(len(yi),1)
    #z = z[~np.isnan(z)]
    q_at_p = interpolate.griddata(np.concatenate((xi,yi),axis=1),z,points/dstep,method='linear')
    #print(q[int(points[0]/dstep),int(points[1]/dstep)])
    #if np.any(np.isnan(q_at_p)):
    #    q_at_p = fill_nans(xi,yi,z,points[:,0]/dstep,points[:,1]/dstep,q_at_p)
    return q_at_p
#%%
upper_cal = 1/3475.93 #m/pix
lower_cal = 1/3461.77 #m/pix
upper_angle = .00809 # rad
R = np.array([[np.cos(upper_angle),-np.sin(upper_angle)],[np.sin(upper_angle), np.cos(upper_angle)]])
deltat_w = 1/(600) #s
D0 = 1.905e-2

outlet = 130
overlap = 0.6731 - 0.6270625 #m
pivdn32_upper = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper.0048.def.msk.ave.piv')
pivdn32_upper2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/upper/dn32_upper2.0048.def.msk.ave.piv')
pivdn32_lower = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower.0048.def.msk.ave.piv')
pivdn32_lower2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/whole_plume/lower/dn32_lower2.0048.def.msk.ave.piv')

dn32_upper_piv1 = pivdn32_upper.read_frame2d(0)
dn32_upper_piv2 = pivdn32_upper2.read_frame2d(0)
zpiv1 = (np.arange(0,dn32_upper_piv1[0].shape[1])*pivdn32_upper.dx-outlet)*upper_cal/D0
dn32_lower_piv1 = pivdn32_lower.read_frame2d(0)
dn32_lower_piv2 = pivdn32_lower2.read_frame2d(0)
zpiv2 = ((np.linspace(2560,2*2560,dn32_lower_piv1[1].shape[1])-outlet)*lower_cal-overlap)/D0


tau_p = 7.4e-3
w0 = tau_p*9.81

dn32_centerline1 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W1 = ((dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]) +(dn32_upper_piv2[1]*R[0,0] - dn32_upper_piv2[2]*R[0,1]))/2
for p in range(0,dn32_upper_piv1[1].shape[1]):
    w_prof = W1[:,p]
    dn32_centerline1[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
#centerline1[:] = 36
Wc_dn32a = windowed_average(np.mean(W1[20:22,np.where(zpiv1>3)[0].astype('int')],axis=0)[:-1],10)*upper_cal/deltat_w
dn32_centerline2 = np.zeros(np.arange(0,dn32_upper_piv1[1].shape[1]).shape).astype('int')
W2 = (dn32_lower_piv1[1] + dn32_lower_piv2[1])/2
for p in range(0,dn32_lower_piv1[1].shape[1]):
    w_prof = W2[:,p]
    dn32_centerline2[p] = int(np.where(w_prof == np.max(w_prof))[0][0])
Wc_dn32b = windowed_average(W2[[r for r in dn32_centerline2],[c for c in range(0,pivdn32_lower2.nx)]][1:-1],10)*lower_cal/deltat_w
Wc_dn32 = np.hstack((Wc_dn32a,Wc_dn32b[4:]))
zpiv = np.hstack((zpiv1[zpiv1>3][:-1],zpiv2[4:][1:-1]))
#%%
v1_cal = 6.64e-5
v2_cal = 6.76e-5
v3_cal = 7.689e-5
v4_cal = 7.75e-5
v5_cal = 7.52e-5
v6_cal = 7.587e-5
v7_cal = 7.86e-5
v8_cal = 7.96e-5
v1_range = (1.25/39.37,7.9375/39.37)
v2_range = (.201,.373)
v3_range = (.357,.553)
v4_range = (.549,.746)
v5_range = (29.25/39.37,34/39.37)
v6_range = (34.3125/39.37,39.125/39.37)
v7_range = (39.25/39.37,44.25/39.37)
v8_range = (44.3125/39.37, 49.3125/39.37)
deltat = 1/(300) #s

upper_cal = 1/3475.93 #m/pix
lower_cal = 1/3461.77 #m/pix
upper_angle = .00809 # rad
R = np.array([[np.cos(upper_angle),-np.sin(upper_angle)],[np.sin(upper_angle), np.cos(upper_angle)]])
deltat_w = 1/(600) #s
D0 = 1.905e-2

outlet = 130
overlap = 0.6731 - 0.6270625 #m

v1_0 = 70
v2_0 = 76
v3_0 = 87 #85
v4_0 = 88 #

v5_0 = 29
v6_0 = 29
v7_0 = 28
v8_0 = 28

#%% import mean image
img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.avg.img')


v1 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/tracers_dn32_v1.tracers.bsub.0032.def.msk.piv')
v2 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.tracers.bsub.0032.def.msk.piv')
v3 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View3/processed/quiescent/tracers_dn32_v3.tracers.bsub.0032.def.msk.piv')
v4 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/tracers_dn32_v4.tracers.bsub.0032.def.msk.piv')
v5 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View5/processed/quiescent/tracers_dn32_v5.tracers.bsub.0032.def.msk.piv')
v6 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View6/processed/quiescent/tracers_dn32_v6.tracers.bsub.0032.def.msk.piv')
v7 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View7/processed/quiescent/tracers_dn32_v7.tracers.bsub.0032.def.msk.piv')
v8 = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View8/processed/quiescent/tracers_dn32_v8.tracers.bsub.0032.def.msk.piv')

v1_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View1/processed/quiescent/tracers_dn32_v1.tracers.bsub.0032.def.msk.ave.piv')
v2_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.tracers.bsub.0032.def.msk.ave.piv')
v3_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View3/processed/quiescent/tracers_dn32_v3.tracers.bsub.0032.def.msk.ave.piv')
v4_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View4/processed/quiescent/tracers_dn32_v4.tracers.bsub.0032.def.msk.ave.piv')
v5_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View5/processed/quiescent/tracers_dn32_v5.tracers.bsub.0032.def.msk.ave.piv')
v6_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View6/processed/quiescent/tracers_dn32_v6.tracers.bsub.0032.def.msk.ave.piv')
v7_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View7/processed/quiescent/tracers_dn32_v7.tracers.bsub.0032.def.msk.ave.piv')
v8_ave = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View8/processed/quiescent/tracers_dn32_v8.tracers.bsub.0032.def.msk.ave.piv')

v1_ave.cal = v1_cal
v2_ave.cal = v2_cal
v3_ave.cal = v3_cal
v4_ave.cal = v4_cal
v5_ave.cal = v5_cal
v6_ave.cal = v6_cal
v7_ave.cal = v7_cal
v8_ave.cal = v8_cal

v1_ave.range = v1_range
v2_ave.range = v2_range
v3_ave.range = v3_range
v4_ave.range = v4_range
v5_ave.range = v5_range
v6_ave.range = v6_range
v7_ave.range = v7_range
v8_ave.range = v8_range

v1.cal = v1_cal
v2.cal = v2_cal
v3.cal = v3_cal
v4.cal = v4_cal
v5.cal = v5_cal
v6.cal = v6_cal
v7.cal = v7_cal
v8.cal = v8_cal

z1_piv = np.linspace(v1_range[0],v1_range[1],v1.nx)
z2_piv = np.linspace(v2_range[0],v2_range[1],v2.nx)
z3_piv = np.linspace(v3_range[0],v3_range[1],v3.nx)
z4_piv = np.linspace(v4_range[0],v4_range[1],v4.nx)
z5_piv = np.linspace(v5_range[0],v5_range[1],v5.ny)
z6_piv = np.linspace(v6_range[0],v6_range[1],v6.ny)
z7_piv = np.linspace(v7_range[0],v7_range[1],v7.ny)
z8_piv = np.linspace(v8_range[0],v8_range[1],v8.ny)
v1_frac = np.loadtxt(os.path.dirname(v1.file_name)+'/v1_frac.txt')
v2_frac = np.loadtxt(os.path.dirname(v2.file_name)+'/v2_frac.txt')
v3_frac = np.loadtxt(os.path.dirname(v3.file_name)+'/v3_frac.txt')
v4_frac = np.loadtxt(os.path.dirname(v4.file_name)+'/v4_frac.txt')
v5_frac = np.loadtxt(os.path.dirname(v5.file_name)+'/v5_frac.txt')
v6_frac = np.loadtxt(os.path.dirname(v6.file_name)+'/v6_frac.txt')
v7_frac = np.loadtxt(os.path.dirname(v7.file_name)+'/v7_frac.txt')
v8_frac = np.loadtxt(os.path.dirname(v8.file_name)+'/v8_frac.txt')
v1_ave.frac = v1_frac
v2_ave.frac = v2_frac
v3_ave.frac = v3_frac
v4_ave.frac = v4_frac
v5_ave.frac = v5_frac
v6_ave.frac = v6_frac
v7_ave.frac = v7_frac
v8_ave.frac = v8_frac

v1_ave.r0 = v1_0
v2_ave.r0 = v2_0
v3_ave.r0 = v3_0
v4_ave.r0 = v4_0
v5_ave.r0 = v5_0
v6_ave.r0 = v6_0
v7_ave.r0 = v7_0
v8_ave.r0 = v8_0
first = [v1_ave,v2_ave,v3_ave,v4_ave]
last = [v5_ave,v6_ave,v7_ave,v8_ave]
#%% determine angle of spread
tmp = np.diff(v2_frac[0:80,:]<0.8,axis=0)
pts = np.where(tmp==True)
pts[0].sort();pts[1].sort()
p = np.polyfit(pts[1][::-1],pts[0][::-1],1)
m = p[0]
angle = np.arctan(m/1)*180/np.pi
#%% choose a radial location
r = pts[0][::-1]-5
z = pts[1]
r_m = r*v2.dx*v2.cal
n_vec = np.array([1,-1/m])/np.sqrt((-1/m)**2+1)

#%% calculate velocity normal to spread angle
v_vec = np.array([v2_ave.read_frame2d(0)[1][z[10],r[10]],v2_ave.read_frame2d(0)[2][z[10],r[10]]])
v_in = np.dot(v_vec,n_vec)
v_ins = []
for i in range(0,z.shape[0]):
    v_vec = np.array([v2_ave.read_frame2d(0)[1][z[i],r[i]],v2_ave.read_frame2d(0)[2][z[i],r[i]]])
    v_ins.append(np.dot(v_vec,n_vec)*v2.cal/deltat)
#%% calculate entrainment mass flux
me = 2*np.pi*1.225/np.cos(angle/180*np.pi)*simps(v_ins*r_m,dx=v2.dx*v2.cal)

#%% 0 -5 -15
r_offset = 0
mes = []
angles = []
k=2
z_coord = []
for v in first:
    tmp = np.diff(v.frac[0:80,:]<0.8,axis=0)
    pts = np.where(tmp==True)
    pts = np.array(sorted(zip(pts[1],pts[0])))
    #pts[0].sort();pts[1].sort()
    p = np.polyfit(pts[:,0],pts[:,1],1)
    m = p[0]
    angle = np.arctan(m/1)*180/np.pi
    angles.append(-angle)
    me_dots = []
    z = pts[:,0]
    r = np.round(np.polyval(p,z)+r_offset).astype('int')
    r = nma.masked_array(r,mask=(r<0))
    r_m = (v.r0 - r)*v.dx*v.cal
    v_ins = []
    n_vec = np.array([1,-1/m])/np.sqrt((-1/m)**2+1)
    zs = np.linspace(v.range[0],v.range[1],v.nx)
    #p = np.concatenate((z.reshape(len(z),1),r.reshape(len(r),1)),axis=1)
    #uz = q_at_p(p,v.read_frame2d(0)[1],v.dx)
    #ur = q_at_p(p,v.read_frame2d(0)[2],v.dx)
    #u = np.concatenate((uz.reshape(len(uz),1),ur.reshape(len(ur),1)),axis=1)
    for i in range(0,z.shape[0]):
        try:
            v_vec = np.array([v.read_frame2d(0)[1][r[i],z[i]],v.read_frame2d(0)[2][r[i],z[i]]])
            v_in = np.dot(v_vec,n_vec)*v.cal/deltat
            v_ins.append(v_in)
        except:
            v_ins.append(np.nan)
    v.v_ins = np.array(v_ins)
    v.zcoord = np.array(zs)
    for i in range(1,z.shape[0]-1,k):
        me = 2*np.pi*1.225/np.cos(angle/180*np.pi)*simps(v_ins[i:i+k]*r_m[i:i+k],dx=v.dx*v.cal)
        #me = 2*np.pi*r/np.cos(angle/180*np.pi)*1.225*np.array(v_ins)**2/2
        mes.append(me)
        z_coord.append(zs[i])


r_offset = -r_offset
for v in last:
    tmp = np.diff(v.frac[:,40:]<0.8,axis=1)
    pts = np.where(tmp==True)
    #pts = np.array(sorted(zip(pts[1],pts[0])))
    p = np.polyfit(pts[0][0:60],pts[1][0:60],1)
    m = p[0]
    angle = np.arctan(m/1)*180/np.pi
    angles.append(angle)
    z = pts[0]
    r = np.round(np.polyval(p,z)+r_offset).astype('int')+40
    r = nma.masked_array(r,mask=(r<0))    
    r_m = (r-v.r0)*v.dx*v.cal
    v_ins = []
    n_vec = np.array([-1/m,1])/np.sqrt((-1/m)**2+1)
    zs = np.linspace(v.range[0],v.range[1],v.ny)
    for i in range(0,z.shape[0]):
        try:
            v_vec = np.array([v.read_frame2d(0)[1][z[i],r[i]],v.read_frame2d(0)[2][z[i],r[i]]])
            v_ins.append(np.dot(v_vec,n_vec)*v.cal/deltat)
        except:
            v_ins.append(np.nan)
    v.v_ins = np.array(v_ins)
    for i in range(1,z.shape[0]-1,k):
        me = 2*np.pi*1.225/np.cos(angle/180*np.pi)*simps(v_ins[i:i+k]*r_m[i:i+k],dx=v.dx*v.cal)
        #me = 2*np.pi*r/np.cos(angle/180*np.pi)*1.225*np.array(v_ins)**2/2
        mes.append(me)
        z_coord.append(zs[i])
    v.zcoord = np.array(zs)
        
#%%
plt.figure(2);plt.quiver(v8_r,v8_z,(v.read_frame2d(0)[1])*(v.read_frame2d(0)[0]),
                         (v.read_frame2d(0)[2])*(v.read_frame2d(0)[0]),scale=250,color='blue');
plt.figure(2);plt.plot(r_m,v8_z[:-1],'r',linewidth=5)
plt.tick_params(axis='both',which='major',labelsize=15);
plt.figure(2);plt.title('$V_8$',fontsize=35,y=1.04)
#%% mean velocity fields
v1_r = (v1_0-np.arange(0,v1.ny))*v1.dx*v1_cal
v1_z = np.linspace(v1_range[0],v1_range[1],v1.nx)

v2_r = (v2_0-np.arange(0,v2.ny))*v2.dx*v2_cal
v2_z = np.linspace(v2_range[0],v2_range[1],v2.nx)

v3_r = (v3_0-np.arange(0,v3.ny))*v3.dx*v3_cal
v3_z = np.linspace(v3_range[0],v3_range[1],v3.nx)

v4_r = (v4_0-np.arange(0,v4.ny))*v4.dx*v4_cal
v4_z = np.linspace(v4_range[0],v4_range[1],v4.nx)

v5_r = (np.arange(0,v5.nx)-v5_0)*v5.dx*v5_cal
v5_z = np.linspace(v5_range[0],v5_range[1],v5.ny)

v6_r = (np.arange(0,v6.nx)-v6_0)*v6.dx*v6_cal
v6_z = np.linspace(v6_range[0],v6_range[1],v6.ny)

v7_r = (np.arange(0,v7.nx)-v7_0)*v7.dx*v7_cal
v7_z = np.linspace(v7_range[0],v7_range[1],v7.ny)

v8_r = (np.arange(0,v8.nx)-v8_0)*v8.dx*v8_cal
v8_z = np.linspace(v8_range[0],v8_range[1],v8.ny)

#%%
r_offset = 0
mes = []
angles = []
k=2
z_coord = []
for v in first:
    r_offsets = np.arange(-15,12)#[0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15]
    tmp = np.diff(v.frac[0:80,:]<0.8,axis=0)
    pts = np.where(tmp==True)
    pts = np.array(sorted(zip(pts[1],pts[0])))
    #pts[-1][1]=pts[-1][1]-15
    pts[:,1] = pts[-1,1]
    #pts[0].sort();pts[1].sort()
    #p = np.polyfit([pts[0,0],pts[-1,0]],[pts[0,1],pts[-1,1]],1)
    p = np.polyfit(pts[:,0],pts[:,1],1)
    m = 0#p[0]
    angle = np.arctan(m/1)*180/np.pi
    angles.append(-angle)
    me_dots = []
    r_0deg = []
    for ro in r_offsets:
        mes = []
        z_coord = []
        z = pts[:,0]
        #r = np.ones(z.shape)*27+ro
        r = np.round(np.polyval(p,z)+ro).astype('int')
        r = nma.masked_array(r,mask=(r<0))
        r_m = (v.r0 - r)*v.dx*v.cal
        v_ins = []
        #n_vec = np.array([1,-1/m])/np.sqrt((-1/m)**2+1)
        n_vec = np.array([0,1])/np.sqrt((0)**2+1)
        zs = np.linspace(v.range[0],v.range[1],v.nx)
        #p = np.concatenate((z.reshape(len(z),1),r.reshape(len(r),1)),axis=1)
        #uz = q_at_p(p,v.read_frame2d(0)[1],v.dx)
        #ur = q_at_p(p,v.read_frame2d(0)[2],v.dx)
        #u = np.concatenate((uz.reshape(len(uz),1),ur.reshape(len(ur),1)),axis=1)
        for i in range(0,z.shape[0]):
            try:
                v_vec = np.array([v.read_frame2d(0)[1][r[i],z[i]],v.read_frame2d(0)[2][r[i],z[i]]])
                v_in = np.dot(v_vec,n_vec)*v.cal/deltat
                v_ins.append(v_in)
            except:
                v_ins.append(np.nan)
        for i in range(1,z.shape[0]-1,k):
            me = 2*np.pi*1.225/np.cos(angle/180*np.pi)*simps(v_ins[i:i+k]*r_m[i:i+k],dx=v.dx*v.cal)
            #me = 2*np.pi*r/np.cos(angle/180*np.pi)*1.225*np.array(v_ins)**2/2
            mes.append(me)
            z_coord.append(zs[i])
        me_dot = np.gradient(mes,16*v.cal)
        me_dots.append(me_dot[-2])
        r_0deg.append(r_m[-2])
        
#%% interpolate plume PIV at zoomed in locations

v1_interp = np.interp(v1_ave.zcoord[:len(v1_ave.v_ins)],zpiv*D0,Wc_dn32)
v2_interp = np.interp(v2_ave.zcoord[:len(v2_ave.v_ins)],zpiv*D0,Wc_dn32)
v3_interp = np.interp(v3_ave.zcoord[:len(v3_ave.v_ins)],zpiv*D0,Wc_dn32)
v4_interp = np.interp(v4_ave.zcoord[:len(v4_ave.v_ins)],zpiv*D0,Wc_dn32)
v5_interp = np.interp(v5_ave.zcoord[:len(v5_ave.v_ins)],zpiv*D0,Wc_dn32)
v6_interp = np.interp(v6_ave.zcoord[:len(v6_ave.v_ins)],zpiv*D0,Wc_dn32)
v7_interp = np.interp(v7_ave.zcoord[:len(v7_ave.v_ins)],zpiv*D0,Wc_dn32)
v8_interp = np.interp(v8_ave.zcoord[:len(v8_ave.v_ins)],zpiv*D0,Wc_dn32)

alpha1 = v1_ave.v_ins/((v1_interp-w0)/2)
alpha2 = v2_ave.v_ins/((v2_interp-w0)/2)
alpha3 = v3_ave.v_ins/((v3_interp-w0)/2)
alpha4 = v4_ave.v_ins/((v4_interp-w0)/2)
alpha5 = v5_ave.v_ins/((v5_interp-w0)/2)
alpha6 = v6_ave.v_ins/((v6_interp-w0)/2)
alpha7 = v7_ave.v_ins/((v7_interp-w0)/2)
alpha8 = v8_ave.v_ins/((v8_interp-w0)/2)

all_alphas = np.hstack((alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8))
all_zcoords = np.hstack((v1_ave.zcoord[0:len(v1_ave.v_ins)],v2_ave.zcoord[0:len(v2_ave.v_ins)],v3_ave.zcoord[0:len(v3_ave.v_ins)],
                         v4_ave.zcoord[0:len(v4_ave.v_ins)],v5_ave.zcoord[0:len(v5_ave.v_ins)],v6_ave.zcoord[0:len(v6_ave.v_ins)],
                         v7_ave.zcoord[0:len(v7_ave.v_ins)],v8_ave.zcoord[0:len(v8_ave.v_ins)]))
f,ax = plt.subplots();
ax.plot(all_alphas,all_zcoords/D0,'ko');
ax.tick_params(axis='both',which='major',labelsize=15);
ax.set_title('$80\%$ $threshold$',fontsize=35,y=1.04)
ax.set_ylim(65,0);
ax.set_xlim(0,.15);
#ax.xaxis.set_label_position('top')
xlab = ax.set_xlabel(r'$\alpha = u_e/(W_c - \tau_p g)$',fontsize=20,labelpad=15)
ylab = ax.set_ylabel('$z/D_0$',fontsize=20,labelpad=15)
#ax.xaxis.tick_top()