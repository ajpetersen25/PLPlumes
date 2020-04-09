"""
Tracer/inertial particle separation tools

Created on Wed May 25 15:31:13 2016

@author: alec petersen
"""
import numpy as np
import copy
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morph
from scipy.ndimage import uniform_filter
from scipy.ndimage import generic_filter
import numba
from numba import cfunc, carray
from numba.types import intc, CPointer, float64, intp, voidptr
from scipy import LowLevelCallable

from PLPlumes.pio import imgio


def jit_filter_function(filter_function):
    jitted_function = numba.jit(filter_function, nopython=True)
    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
    def wrapped(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=float64)
        result[0] = jitted_function(values)
        return 1
    return LowLevelCallable(wrapped.ctypes)

@jit_filter_function
def f_std(a):
    return np.std(a)

    

def separation_alg(params):
    img,labeling_threshold, particle_threshold,min_size,particle_flare,window_size,f = params
    """
    separates tracers from larger particles in single frame
    """
    if particle_flare == 0:
        kernel=np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])
    elif particle_flare == 1:
        kernel=np.array([[0,0,0,1,0,0,0],[0,0,1,1,1,0,0],[0,1,1,1,1,1,0], [1,1,1,1,1,1,1],[0,1,1,1,1,1,0],[0,0,1,1,1,0,0],[0,0,0,1,0,0,0]])
    
    frame = img.read_frame2d(f)
    #image_bitdepth,iy,ix = frame.dtype,frame.shape[0],frame.shape[1]
    image_bitdepth = frame.dtype
    f1=copy.deepcopy(frame)
    # set all pixels below labeling threshold to 0 -- allows labeling function to find contiguous objects

    f1[f1<labeling_threshold] = 0
    f1lab, f1num = measurements.label(f1)
    f1slices = measurements.find_objects(f1lab)

    # initialization of particle and tracer images
    particles = np.zeros(f1.shape,dtype=image_bitdepth)
    #noise_particles = np.zeros(f1.shape,dtype=image_bitdepth)
    tracers = copy.deepcopy(frame)

    """# mean and std deviation for filling particles in
    #try:
    #    mu = np.mean(frame[frame<labeling_threshold])
    #    sigma = np.std(frame[frame<labeling_threshold])
    #    # fill array as large as image
    #    fillarray = np.abs(np.rint(np.random.normal(mu,.75*sigma,[iy,ix])))
    #    fillarray = fillarray.astype(tracers.dtype)
    #except:
    slices = []
    x = np.linspace(0,frame.shape[1],int(np.ceil(frame.shape[1]/fill_kernel)+1))
    y = np.linspace(0,frame.shape[0],int(np.ceil(frame.shape[0]/fill_kernel)+1))
    x = x.astype('int'); y = y.astype('int')
    fillarray = np.zeros(frame.shape)
    for xi in range(0,len(x)-1):
        for yi in range(0,len(y)-1):
            slices.append((slice(y[yi],y[yi]+fill_kernel),slice(x[xi],x[xi+1])))
    for s in slices:
t        mu = np.mean(frame[s])
        sigma = np.std(frame[s])
        fillarray[s] = np.abs(np.rint(np.random.normal(mu,sigma,frame[s].shape)))"""
   
    fill_array_avg = uniform_filter(frame,size=window_size,mode='nearest').flatten()
    fill_array_std = generic_filter(frame,f_std,size=window_size,mode='nearest').flatten()
    fillarray = np.zeros(frame.shape).flatten()
    for i in range(0,len(fillarray)):
        mu = fill_array_avg[i]
        sigma = fill_array_std[i]
        fillarray[i] = np.abs(np.rint(np.random.normal(mu,.25*sigma)))
    fillarray = fillarray.reshape(frame.shape[0],frame.shape[1])
    fillarray[fillarray>4096]=4096  
    fillarray[fillarray<0]=0
        
    for s, f1slice in enumerate(f1slices):
        # remove overlapping label objects from current slice
        img_object = f1[f1slice]*(f1lab[f1slice]==(s+1))
        obj_size = np.count_nonzero(img_object)
        # threshold objects based on size and then mean intensity
        if obj_size >= min_size:
            
            #obj_int = np.sum((img_object))/obj_size
            #if obj_int < particle_threshold:
                #noise_particles[f1slice] = img_object #TODO, should noise_particles always be removed??
            #elif obj_int > particle_threshold:
            particles[f1slice] = img_object
    # dilate paticles to remove any residual particle noise in the form of halos
    particles_mask = morph.binary_dilation(particles,structure = kernel)
    del particles
    tracers_mask = ~particles_mask
    # use mask on raw image to generate tracers only
    tracers = tracers*tracers_mask
    particles_fill = fillarray*particles_mask
    tracers = tracers + particles_fill
                
    return tracers




