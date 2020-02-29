#!/bin/bash -l

source /home/colettif/pet00105/.bashrc

python /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/preprocessing/mask_velocity.py $img_file $piv_file $threshold $window_threshold $start_frame $end_frame $cores
