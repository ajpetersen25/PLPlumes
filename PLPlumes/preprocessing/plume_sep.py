#!/usr/bin/env python3
"""
Created on Wed Feb 19 14:55:54 2020

@author: alec
"""

import os
import copy
import getpass
import argparse
from datetime import datetime
from . import separation_alg
import numpy as np
from PLPlumes.pio import imgio
from clint.textui import progress
import multiprocessing
from itertools import repeat



####################################################################
"""
sep -- separation class
"""
####################################################################

class sep_obj:
    
    def __init__(self, img_file, labeling_threshold, particle_threshold, min_size,particle_flare, window_size, start_frame, end_frame,cores):

        """
        Initializes instance of a separation object coupled to standard separation algorithm
        and the raw IMG file
        
        Requires:
            img_file           - str -  name of raw IMG file (with extention)
            labeling_threshold - int -  noise threshold; pixels below get set to 0, pixels above get labeled
            particle_threshold - int -  threshold for mean intensity of an inertial particle 
            min_size           - int -  minimum area (in pixels) for an inertial particle
            particle_flare     -bool -  default 0 (little flare), choose 1 for larger dilation kernel to deal with brighter/bigger particles
            window_size        - int -  size of window for calculating local std on for fill in values
            start_frame        - int -  first frame number (numbering starts at 0)
            end_frames         - int -  termination frame (1000 stops at frame 999. therefore start_frame = 0 to end_frame = 1000 covers 1000 frames)
            
        """
        # 1. initialize the image instance
        self._init_img(img_file)
        
        # 2. assign the variables
        self._init_vars(labeling_threshold, particle_threshold, min_size, particle_flare, window_size, start_frame, end_frame, cores)
        
        # 3. initialize empty separated particle and tracer IMG files
        self._init_output_imgs()
        
        
    def _init_vars(self, labeling_threshold, particle_threshold, min_size, particle_flare, window_size, start_frame, end_frame, cores):
        """
        Initialize instance variables for separation by size and mean intensity and assigns
        to instance variabels
        
        Arguments:
            same as class
            
        Returns:
            None
        """
        
        self.labeling_threshold = labeling_threshold
        self.particle_threshold = particle_threshold
        self.min_size = min_size
        self.particle_flare = particle_flare
        self.window_size = window_size
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.cores = cores
        
        
    def _init_img(self, img_file):
        """
        Create and assign an imgio object based on img filename to class instance
        Store file root as class instance variable
        
        Arguments:
            img_filename -- str, valid relative path to IMG file
            
        Returns:
            None
        """
        
        #initialize img object
        self.img = imgio.imgio(img_file)
        
        # separate out IMG file and extension
        self.file_root, self.ext = os.path.splitext(self.img.file_name)
 

    def _init_output_imgs(self):
        """
        Initialize output tracers and particles IMG files as copies of original IMG
        
        Arguments:
            None
        Returns:
            None
        """
        #self.pimg = copy.deepcopy(self.img) 
        #self.pimg.file_name = '%s.particles.img' % self.file_root
        #self.pimg.it = (self.end_frame - self.start_frame)
        
        self.timg = copy.deepcopy(self.img)
        self.timg.file_name = '%s.tracers.img' % self.file_root
        self.timg.it = (self.end_frame - self.start_frame)

              
    def separate_frames(self):
        """
        main separation function, uses separation.py tools from ptvtools repository
        """
        print('Total frames to discriminate : %d' % (self.end_frame - self.start_frame))
    
        d=datetime.now()
        self.timg.comment = "%s\n%s %s \npypiv git sha: @SHA@\n%s\n\n" % (getpass.getuser(), os.path.basename(__file__), self.img.file_name, d.strftime("%a %b %d %H:%M:%S %Y")) + str(self.img.comment,'utf-8')

        #self.pimg.write_header()
        self.timg.write_header()
        
        #fp = open('%s' % self.pimg.file_name, 'ab')
        #fp.seek(self.pimg.header_length)
        
        ft = open('%s' % self.timg.file_name, 'ab')
        ft.seek(self.timg.header_length)
        param2 = self.labeling_threshold
        param3 = self.particle_threshold
        param4 = self.min_size
        param5 = self.particle_flare
        param6 = self.window_size
        param7 = range(self.start_frame,self.end_frame)
        f_tot = len(param7)
        objList = list(zip(repeat(self.img,times=f_tot),
                      repeat(param2,times=f_tot),
                      repeat(param3,times=f_tot),
                      repeat(param4,times=f_tot),
                      repeat(param5,times=f_tot),
                      repeat(param6,times=f_tot),
                      param7))

        pool = multiprocessing.Pool(processes=self.cores)
        
        # process
        
        #all_frame_masks,all_U,all_W = pool.map(mask_vframe,objList)
        tracer_images = pool.map(separation_alg.separation_alg,objList)
        tracer_images = np.array(tracer_images)
 
        for f in range(0,f_tot):
            self.timg.write_frame(np.flipud(tracer_images[f,:,:]))

    def gen_comment(self):
        """
        generate separation comments
        """
        d = datetime.now()
        
        return "%s\n%s %s %d %d %d %d \fmlib git sha: @VERSION@ git sha: @VERSION\n%s\n\n" % (getpass.getuser(),
                                                                                              inspect.stack()[-1][1],
                                                                                              self.img.file_name,
                                                                                              self.start_frame,
                                                                                              self.end_frame,
                                                                                              self.labeling_threshold,
                                                                                              self.particle_threshold,
                                                                                              self.min_size,
                                                                                              d.strftime("%a %b %d %H:%M:%S %Y"))


def sep_parse():
    
    parser= argparse.ArgumentParser(description='Program to separate .img file into particle and tracer img files', formatter_class=argparse.RawTextHelpFormatter)    
    parser.add_argument('img_file', type=str, help='Input image file')    
    parser.add_argument('labeling_threshold', type=int, help='Noise intensity threshold')
    parser.add_argument('particle_threshold', type=int, help='Particle intensity threshold' )    
    parser.add_argument('min_size', type=int, help='Particle radius threshold (pixels)')
    parser.add_argument('particle_flare',nargs='?',default=0,type=bool, help='Set to 1 if particles flare/increase local background intensity. (uses a larger kernel for dilation')
    parser.add_argument('window_size',nargs='?',default=50,type=int,help='window size for calculating local mean & std for noise filling')
    parser.add_argument('start_frame',nargs='?',default=0,type=int, help='Frame to start separation from')
    parser.add_argument('end_frame',nargs='?', default=-1,type=int, help='Number of frames to separate')
    parser.add_argument('cores',type=int,nargs='?',default=1,help='Optional - Force number of cores for node_separate')
    #TODO
    #parser.add_argument('auto_threshold',nargs='?', default='False',help='If true, labeling threshold chosen automatically based on histogram')
    args = parser.parse_args()
    print(args.img_file)
    img = imgio.imgio(args.img_file)
        
    if args.end_frame < 0:
        args.end_frame = img.it   
    
    # check is IMG file exists

    fail = False
    if os.path.exists(args.img_file) == 0:
        print('[ERROR] IMG file does not exist')
        fail = True

    if fail:
        print('Exiting...')
        os.sys.exit(1)
    return args                
                       
