#!/bin/bash -l

source /home/colettif/pet00105/.bashrc

python /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/preprocessing/separate.py $img_file $labeling_threshold $particle_threshold $min_size $particle_flare $window_size $start_frame $end_frame $cores
