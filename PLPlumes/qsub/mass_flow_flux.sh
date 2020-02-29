#!/bin/bash -l

source /home/colettif/pet00105/.bashrc

python /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/plume_processing/mass_flow_flux.py $img_file $piv_file $start_frame $end_frame $cores
