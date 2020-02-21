#!/usr/bin/env python3
"""
Created on Wed Feb 19 14:59:50 2020

@author: alec
"""

"""
separate.py
Multi-phase image processing --- separates tracers and inertial particles in
.img files
Created on Wed May 25 15:21:53 2016
@author: alec petersen
"""
import time

from PLPlumes.preprocessing import plume_sep

def main():
    # parse inputs
    args = plume_sep.sep_parse()
    t1 = time.time()
    
    sep_frames=plume_sep.sep_obj(args.img_file, args.labeling_threshold, args.particle_threshold, args.min_size, args.particle_flare,args.window_size,args.start_frame, args.end_frame,args.cores)
                   
    sep_frames.separate_frames()
    
    print('%f ms elapsed' % ((time.time()-t1)*1000))
    
    
if __name__ == "__main__":
    main()