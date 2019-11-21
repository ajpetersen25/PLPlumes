import numpy as np
from pio import imgio
import multiprocessing
from itertools import repeat
import os


def make_cmap_frame(params):
    img,slices,size,dx,frame = params
    c_map_frame = np.zeros(size)
    for s in slices:
        c_map_frame[int(s[0].start/dx-1),int(s[1].start/dx-1)] = np.mean(img.read_frame2d(frame)[s])

    return c_map_frame



def mp_handler():
    img = imgio.imgio('/home/colettif/pet00105/Coletti/Data_2019/Plumes/whole_plume/plume_30um_dn32_D019/pano/Analysis/pano.img')
    dx = 24
    ppn = 24
    x = np.arange(0+dx, img.ix-dx,dx)
    y = np.arange(0+dx,img.iy-dx,dx)
    slices = []
    for i in x:
        for j in y:
            slices.append((slice(j,j+dx),slice(i,i+dx)))


    param1 = img
    param2 = slices
    param3 = (44,212)
    param4 = dx
    param5 = range(0,1)
    f_tot = len(param3)
    objList = zip(repeat(param1,times=f_tot),
                  repeat(param2,times=f_tot),
                  repeat(param3,times=f_tot),
                  repeat(param4,times=f_tot),
                  param5)
    pool = multiprocessing.Pool(processes=ppn)

    c_map = pool.map(make_cmap_frame,objList)
    results = np.array(c_map)
    np.savez_compressed('/home/colettif/pet00105/Coletti/Data_2019/Plumes/whole_plume/plume_30um_dn32_D019/pano/Analysis/'+'.c_map_0_1000.npz',results)
    
mp_handler()