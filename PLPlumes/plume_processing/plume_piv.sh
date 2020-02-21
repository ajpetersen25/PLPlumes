#!/bin/bash -l

source /home/colettif/pet00105/.zshrc

python /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/plume_processing/plume_piv.py $img_file $start_frame $end_frame $piv_increment $cores
