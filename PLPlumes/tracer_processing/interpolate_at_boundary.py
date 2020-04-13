#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:29:53 2020

@author: ajp25
"""
import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import dilation,square,binary_erosion
import scipy.ndimage.measurements as measurements
from PLPlumes.pio import pivio,imgio
from scipy import interpolate
from scipy.ndimage import map_coordinates
import time
import argparse
import multiprocessing as mp
from itertools import repeat

import os

import h5py
import tables


def fill_nans(x, y, z, xp, yp, q_at_p):
    """"
    Fill in the NaNs or masked values on interpolated points using nearest
    neighbors.

    .. warning::

        Operation is performed in place. Replaces the NaN or masked values of
        the original array!

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the original data points (not
        interpolated).
    * v : 1D array
        Array with the scalar value assigned to the data points (not
        interpolated).
    * xp, yp : 1D arrays
        Points where the data values were interpolated.
    * vp : 1D array
        Interpolated data values (the one that has NaNs or masked values to
        replace).

    """
    if np.ma.is_masked(q_at_p):
        nans = q_at_p.mask
    else:
        nans = np.isnan(q_at_p)
    q_at_p[nans] = interpolate.griddata(((x.ravel()), y.ravel()), z, (xp[nans], yp[nans]), method='nearest').ravel()
    return q_at_p

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




def plume_outline(frame,dilation_size,dilation_iterations,threshold,med_filt_size,orientation='horz'):
    """
    img_outline = frame>threshold
    contours, hierarchy =   cv2.findContours(img_outline.copy().astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    c_size = ([c.size for c in contours])
    lc = np.where(c_size == np.max(c_size))[0][0]
    plume_contour = cv2.drawContours(image,contours[lc],-1, 255, 1)

    return plume_contour"""
    frame_d = frame.copy()
    i=0
    kernel = square(dilation_size)
    while i<dilation_iterations:
        frame_d = dilation(frame_d,kernel)
        i+=1
    frame_db = frame_d > threshold
    frame_mask = median_filter(frame_db,med_filt_size)
    f1lab, f1num = measurements.label(frame_mask)
    f1slices = measurements.find_objects(f1lab)
    obj_sizes = []
    img_objects = []
    for s, f1slice in enumerate(f1slices):
        # remove overlapping label objects from current slice
        img_object = ((f1lab==(s+1)))
        obj_size = (np.count_nonzero(img_object))
        obj_sizes.append(obj_size)
        img_objects.append(img_object)
    frame_mask = img_objects[np.where(obj_sizes==np.max(obj_sizes))[0][0]]
    plume_outline_pts = []
    if orientation == 'horz':
        for col in range(0,frame.shape[1]):
            rows = np.where(np.diff(frame_mask[:,col])==1)[0]
            rows = rows[rows<1200]
            for row in rows:
                plume_outline_pts.append(np.array([row,col]))
    elif orientation == 'vert':
        for row in range(0,frame.shape[0]):
            cols = np.where(np.diff(frame_mask[row,:])==1)[0]
            cols = cols[cols>500]
            for col in cols:
                plume_outline_pts.append(np.array([row,col]))
        
    return plume_outline_pts,frame_mask

def entrainment_vel(params):
    img,piv,dilation_kernel,dilation_iter,threshold,med_filt_size,orientation,frame = params
    plume_outline_pts,frame_mask = plume_outline(img.read_frame2d(frame),dilation_kernel,dilation_iter,
                                                 threshold,med_filt_size,orientation)
      
    pts = np.array([np.array([r[0]-piv.dx,r[1]]) for r in plume_outline_pts])    
    
    X,Y = np.meshgrid(piv.dx * np.arange(0,piv.read_frame2d(frame)[0].shape[1]) + piv.dx,
                      piv.dx * np.arange(0,piv.read_frame2d(frame)[0].shape[0]) + piv.dx)
    
    points = np.vstack([Y.ravel(),X.ravel()])
    vf_mask = binary_erosion(map_coordinates(frame_mask,points).reshape(piv.read_frame2d(frame)[0].shape))
    u = (piv.read_frame2d(frame)[1])
    v = (piv.read_frame2d(frame)[2])
    u[piv.read_frame2d(frame)[0]!=1]=np.nan
    v[piv.read_frame2d(frame)[0]!=1]=np.nan
    u[vf_mask!=0] = np.nan
    v[vf_mask!=0] = np.nan
    u_at_p = q_at_p(pts,u.transpose(),piv.dx)
    v_at_p = q_at_p(pts,v.transpose(),piv.dx)
    u_e = []

    for c in range(0,img.ix):
        cols = np.where(pts[:,1]==c)[0]
        for c2 in cols:
            vel_vec = [u_at_p[c2],v_at_p[c2]]
            try:
                if np.isnan(vel_vec).any():
                    u_e.append(np.nan)
                else:
                    fit = np.polyfit(pts[c-2:c+3,1],pts[c-2:c+3,0],1)
                    line= np.array([1,fit[0]])
                    l_norm = np.sqrt(np.sum(np.array(line)**2))
                    proj = (np.dot(vel_vec,line)/l_norm**2)*np.array(line)
                    perp_vel = np.linalg.norm(vel_vec - proj)
                    u_e.append(perp_vel)
            except:
                u_e.append(np.nan)


    try:
        u_at_boundary = np.hstack((pts,np.array(u_e).reshape(len(u_e),1)))
    except:
        print('frame %d ERROR' %frame)
    return(u_at_boundary)

def main():
    tic = time.time()
    # parse inputs
    parser = argparse.ArgumentParser(
               description='Program for parallel plume piv',
               formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('img_file',type=str, help='path to .img file')
    parser.add_argument('piv_file',type=str, help='Name of .piv file')
    parser.add_argument('dilation_kernel',type=int,help='size of dilation filter')
    parser.add_argument('dilation_iteration',type=int,help='iterations of dilation filter')
    parser.add_argument('threshold',type=int,help='intensity threshold for binarizing image')
    parser.add_argument('med_filt_size',type=int,help='size of median filter')
    parser.add_argument('start_frame',nargs='?',default=0,type=int, help='Frame to start separation from')
    parser.add_argument('end_frame',nargs='?', default=0,type=int, help='Number of frames to separate')
    parser.add_argument('orientation',type=str,nargs='?', default='horz',help='Number of frames to separate')
    parser.add_argument('cores',type=int,nargs='?',default=1,help='Optional - Force number of cores for node_separate')
    args = parser.parse_args() 
    img = imgio.imgio(args.img_file)
    piv = pivio.pivio(args.piv_file)
    if args.end_frame == 0:
        end_frame = img.it
    else:
        end_frame = args.end_frame
    frames = np.arange(args.start_frame,end_frame)
    f_tot = len(frames)
    
    piv_root,piv_ext = os.path.splitext(args.piv_file)
    h5 = h5py.File(piv_root+'.u_e.hdf5','a')
    frame_group = h5.create_group('frames')
    
    
    objList = list(zip(repeat(img,times=f_tot),
                   repeat(piv,times=f_tot),
                   repeat(args.dilation_kernel,times=f_tot),
                   repeat(args.dilation_iteration,times=f_tot),
                   repeat(args.threshold,times=f_tot),
                   repeat(args.med_filt_size,times=f_tot),
                   repeat(args.orientation,times=f_tot),
                   frames))

    pool = mp.Pool(processes=args.cores)
    results = pool.map(entrainment_vel,objList)
    results = np.array(results)
        
    for ff,f in enumerate(frames):
        h5['frames'].create_dataset('frame_%06d' %f,data=results[ff])
        
    h5.close()
    tables.file._open_files.close_all()
    
    print(('[FINISHED]: %f seconds elapsed' %(time.time()-tic)))
    
if __name__ == "__main__":
  main()    
    

#%%

'''img = imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.img')

piv = pivio.pivio('/media/cluster/msi/pet00105/Coletti/Data_2020/Plumes/Plume_dn32/View2/processed/quiescent/tracers_dn32_v2.t.bsub.0032.def.piv')

f=11000
u_at_boundary = np.zeros((12000,2560))
for f in range(0,12000):
    plume_outline_pts,frame_mask = plume_outline(img.read_frame2d(f),40,1,2800,100)
      
    pts = np.array([np.array([r[0]-piv.dx,r[1]]) for r in plume_outline_pts])    
    
    X,Y = np.meshgrid(piv.dx * np.arange(0,piv.read_frame2d(f)[0].shape[1]) + piv.dx,
                      piv.dx * np.arange(0,piv.read_frame2d(f)[0].shape[0]) + piv.dx)
    
    points = np.vstack([Y.ravel(),X.ravel()])
    vf_mask = binary_erosion(map_coordinates(frame_mask,points).reshape(piv.read_frame2d(f)[0].shape))
    u = (piv.read_frame2d(f)[1])
    v = (piv.read_frame2d(f)[2])
    
    u[piv.read_frame2d(f)[0]!=1]=np.nan
    v[piv.read_frame2d(f)[0]!=1]=np.nan
    u[vf_mask!=0] = np.nan
    v[vf_mask!=0] = np.nan
    
    plt.figure(31);plt.pcolormesh(img.read_frame2d(f),cmap='gray');  
    #plt.figure(31);plt.quiver(X,Y,u,v,color='b')
    #plt.figure(31);plt.plot(pts[:,1],pts[:,0],'r.')       
    
    u_at_p = q_at_p(pts,u.transpose(),piv.dx)
    v_at_p = q_at_p(pts,v.transpose(),piv.dx)
    for c in range(2,2560-3):
        vel_vec = [u_at_p[c],v_at_p[c]]
        if np.isnan(vel_vec).any():
            u_at_boundary[c] = np.nan
        else:
            fit = np.polyfit(pts[c-2:c+3,1],pts[c-2:c+3,0],1)
            line= np.array([1,fit[0]])
            l_norm = np.sqrt(np.sum(np.array(line)**2))
            proj = (np.dot(vel_vec,line)/l_norm**2)*np.array(line)
            perp_vel = np.linalg.norm(vel_vec - proj)
            u_at_boundary[c] = perp_vel
    u_at_boundary[u_at_boundary==0]=np.nan
    plt.figure(31);plt.quiver(pts[:,1],pts[:,0],u_at_p[:],v_at_p[:],color='r',scale=100)
    '''