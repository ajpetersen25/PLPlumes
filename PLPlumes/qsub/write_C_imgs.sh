#!/bin/bash -l

source /home/colettif/pet00105/.bashrc

python /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/preprocessing/write_C_imgs.py $img_file $p_lin $start_height $start_frame $end_frame $cores $orientation 
