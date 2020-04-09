#!/bin/bash -l

source /home/colettif/pet00105/.bashrc

python /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/tracer_processing/interpolate_at_boundary.py $img_file $piv_file $dilation_kernel $dilation_iter $threshold $med_filt_size $start_frame $end_frame $orientation $cores
