# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import division
import numpy as np
import numpy.ma as nma

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
rc('text', usetex=True)
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
from scipy.signal import correlate2d
from scipy.spatial.distance import cdist
from copy import deepcopy
import cv2
import imutils
import scipy.ndimage.measurements as measurements
import scipy.ndimage as ndimage
from copy import deepcopy
#%%
cam1_cal = cv2.imread('/media/cluster/lustre2/fcoletti/ajp25/Data_2019/2019_6_20_plume/calibration/cam1_cal.tif')
cam2_cal = cv2.imread('/media/cluster/lustre2/fcoletti/ajp25/Data_2019/2019_6_20_plume/calibration/cam2_cal.tif')
img1 = cv2.cvtColor(cam1_cal,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cam2_cal,cv2.COLOR_BGR2GRAY)
#%%
# indicate region of interest
# Select ROI
im=deepcopy(img2)
fromCenter = False
r = cv2.selectROI('Image',im, fromCenter)

# Crop image
imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# Display cropped image
#cv2.imshow("Image", imCrop)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
threshold=95
imCrop[imCrop<threshold] = 0
f1lab,f1num = measurements.label(imCrop)
f1slices = measurements.find_objects(f1lab)
min_size=9
cal_points = []
for s, f1slice in enumerate(f1slices):
    imCopy = np.zeros(imCrop.shape)
    img_object = imCrop[f1slice]*(f1lab[f1slice]==(s+1))
    obj_size = np.count_nonzero(img_object)
    #threshold objects based on size
    if obj_size > min_size: 
        imCopy[f1slice] = img_object
        centroid = measurements.center_of_mass(imCopy)
        cal_points.append(np.array([s,f1slice,centroid[1]+0.5,centroid[0]+0.5]))
cal_points=np.asarray(cal_points)

#%%
pts = 6
c = np.array([r[0],r[1]])
plt.figure(1)
plt.pcolormesh((imCrop),cmap=cm.gray)
plt.plot(cal_points[:,2],cal_points[:,3],'rx')
print('Choose shared points between images')
x=[]
for i in range(0,pts):
    x0 = plt.ginput(1)[0]
    dist = (cal_points[:,3]-x0[1])**2+(cal_points[:,2]-x0[0])**2
    (mindist, i0) = (dist.min(),np.where(dist==dist.min())[0])
    plt.plot(cal_points[i0,2],cal_points[i0,3],'bo')
    x.append(np.array([cal_points[i0,2][0],cal_points[i0,3][0]]))
    i+=1
x = np.asarray(x).astype('float32') + c




c = np.array([r[0],r[1]])
plt.figure()
plt.pcolormesh(imCrop,cmap=cm.gray)
plt.plot(cal_points[:,2],cal_points[:,3],'rx')
print('Choose shared points between images')
x_pr=[]
for i in range(0,pts):
    x0 = plt.ginput(1)[0]
    dist = (cal_points[:,3]-x0[1])**2+(cal_points[:,2]-x0[0])**2
    (mindist, i0) = (dist.min(),np.where(dist==dist.min())[0])
    plt.plot(cal_points[i0,2],cal_points[i0,3],'bo')
    x_pr.append(np.array([cal_points[i0,2][0],cal_points[i0,3][0]]))
    i+=1

x_pr = np.asarray(x_pr).astype('float32') + c
#%%

#%%
T = np.zeros((3,3))

T[0,0]=1;T[1,1]=1;T[2,2]=1
T[0,2] = np.mean([x[i][0]+img1.shape[1] - x_pr[i][0] for i in range(0,pts)])
T[1,2] = np.mean([x[i][1]- x_pr[i][1] for i in range(0,pts)])
#%%
border = int(200)
height_panorama=img1.shape[0]+border
width_panorama = img1.shape[1]*2
panorama1 = np.zeros((height_panorama, width_panorama))
panorama1[int(border/2):height_panorama-int(border/2), img1.shape[1]:] = img1
ii2 = np.zeros((img2.shape[0]+border,img2.shape[1]))
ii2[100:ii2.shape[0]-int(border/2),:] = img2
panorama2 = cv2.warpPerspective(ii2, T, (width_panorama, height_panorama))
result = panorama1+panorama2

#%%
lr = img1.shape[1]+T[0,2]
ll = img2.shape[1]

alpha1 = np.zeros(panorama1.shape)
for c in range(ll,int(lr+1)):
    alpha1[:,c] = np.abs(c-lr)/(np.abs(c-lr)+np.abs(c-ll))

if T[1,2]<0:
    alpha1[0:int(border/2),:] = 1
    alpha1[int(np.ceil(border/2-T[1,2]+img1.shape[0])):,:] = 0
elif T[1,2]>0:
    alpha1[0:int(border/2),:] = 0
    alpha1[int(np.ceil(border/2-T[1,2]+img1.shape[0])):,:] = 1
alpha1[:,0:ll]=1

alpha2 = np.ones(panorama1.shape)
for c in range(ll,int(lr+1)):
    alpha2[:,c] = np.abs(c-lr)/(np.abs(c-lr)+np.abs(c-ll))
alpha2[:,int(lr+1):]=0
if T[1,2]<0:
    alpha2[0:int(border/2),:] = 1
    alpha2[int(np.ceil(border/2-T[1,2]+img1.shape[0])):,:] = 0
elif T[1,2]>0:
    alpha2[0:int(border/2),:] = 0
    alpha2[int(np.ceil(border/2-T[1,2]+img1.shape[0])):,:] = 1

#%%
panorama = alpha2*panorama2+(1-alpha1)*panorama1

#%%load in lightsheet images
from pio import imgio
cam1_lightsheet = np.zeros(img1.shape)
cam2_lightsheet = np.zeros(img2.shape)

for f in range(0,1000):
    cam1_lightsheet += imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2019/2019_06_25_plume/lightmap/cam1_lightmap.img').read_frame2d(f)
    cam2_lightsheet += imgio.imgio('/media/cluster/msi/pet00105/Coletti/Data_2019/2019_06_25_plume/lightmap/cam2_lightmap.img').read_frame2d(f)
cam1_lightsheet = cam1_lightsheet/(f+1)
cam2_lightsheet = cam2_lightsheet/(f+1)
#%%
border = int(200)
height_panorama=cam1_lightsheet.shape[0]+border
width_panorama = cam1_lightsheet.shape[1]*2
panorama1 = np.zeros((height_panorama, width_panorama))
panorama1[int(border/2):height_panorama-int(border/2), img1.shape[1]:] = cam1_lightsheet
ii2 = np.zeros((img2.shape[0]+border,img2.shape[1]))
ii2[100:ii2.shape[0]-int(border/2),:] = cam2_lightsheet
panorama2 = cv2.warpPerspective(ii2, T, (width_panorama, height_panorama))
lightmap = ndimage.filters.gaussian_filter(alpha2*panorama2+(1-alpha1)*panorama1,20)

lightmap = (alpha2*panorama2+(1-alpha1)*panorama1)

