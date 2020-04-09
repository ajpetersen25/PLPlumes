#!/bin/bash -l

source /home/colettif/pet00105/.bashrc

python /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/plume_processing/fluct_masked_vel.py $piv_file $start_frame $end_frame $cores
