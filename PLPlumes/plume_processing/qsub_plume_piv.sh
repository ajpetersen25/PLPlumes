#!/bin/bash
# PIV cross-correlation code 
if [[ $1 == '-h' ]]; then
echo -e "\nusage: python plume_piv.py [-h]
                     save_path cores start_frame end_frame file_path
                     queue walltime pmem n_jobs working_dir

positional arguments: (all are required)
  save_path             path to save directory
  cores                 number of cores to use per job
  start_frame           start frame for piv
  end_frame             end frame for piv
  files


  queue 		MSI queue name (recommended: small)
  walltime 		walltime req'd (hh:mm:ss)
  pmem 			memory req'd per core (recommended: 2580mb)
  n_jobs 		number of jobs to submit


optional arguments:
  -h, --help            show this help message and exit

Note: this script sends each parallel part of the IMG file to the MSI scheduler as a separate job. 
To cancel the processing, you must cancel each job individually. To cancel all jobs under your 
username, run 'qselect -u \$USER | xargs qdel'.\n"
    exit 0

#elif ! [ -f "$1" ]; then
#    echo "[ERROR] $1 not found"
#    exit 0
fi
ulimit -s 100000
working_dir=`pwd`

# --------------- submit parallel PIV jobs --------------------
#echo {${5}'*.tif'}
declare -a img_list
for i in {'ls' ${5}'*.tif'; do
    img_list+=($i)
done
# find the number of image frames per core
declare -i pairs_per_job
declare -i pairs_last_job
declare -i pairs
pairs=$((${4} - ${3} - 1))
pairs_per_job=$(($pairs/${9}))
pairs_last_job=$(($pairs - $pairs_per_job * (${9} - 1)))
# submit jobs
#echo "${img_list[@]}"
for ((i=0; i<${9}; i++)); do
    start_frame=$(($i * ${pairs_per_job} * 1 + ${3}))
    id[$i]='qsub -q ${6} -l walltime={7},nodes=1:ppn=${2},pmem=${8} -v save_path=${1},cores=${2},start_frame=${3},end_frame=${4},files="${img_list[@]}" /home/alec/Desktop/plume_piv.sh'
done
