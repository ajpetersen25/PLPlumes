#!/bin/bash -l

source /home/colettif/pet00105/.bashrc

python /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/preprocessing/write_C_imgs.py $img_file $p_quad $p_lin $start_frame $end_frame $cores 
