#!/bin/bash
# PIV cross-correlation code 
if [[ $1 == '-h' ]]; then
echo -e "\nusage: python plume_piv.py [-h]
                     img_file labeling_threshold particle_threshold min_size particle_flare window_size
                     start_frame end_frame cores queue walltime pmem n_jobs
                     
positional arguments: (all are required)
  img_file           - str -  name of raw IMG file (with extention)
  labeling_threshold - int -  noise threshold; pixels below get set to 0, pixels above get labeled
  particle_threshold - int -  threshold for mean intensity of an inertial particle 
  min_size           - int -  minimum area (in pixels) for an inertial particle
  particle_flare     -bool -  default 0 (little flare), choose 1 for larger dilation kernel to deal with brighter/bigger particles
  window_size        - int -  size of window for calculating local std on for fill in values
  start_frame        - int -  first frame number (numbering starts at 0)
  end_frames         - int -  termination frame (1000 stops at frame 999. therefore start_frame = 0 to end_frame = 1000 covers 1000 frames)
  cores              - int -  number of cores to use per node
  
  queue         MSI queue name (recommended: small)
  walltime      walltime req'd (hh:mm:ss)
  pmem          memory req'd per core (recommended: 2580mb)
  n_jobs        number of jobs to submit


optional arguments:
  -h, --help            show this help message and exit

Note: this script sends each parallel part of the IMG file to the MSI scheduler as a separate job. 
To cancel the processing, you must cancel each job individually. To cancel all jobs under your 
username, run 'qselect -u \$USER | xargs qdel'.\n"
    exit 0

elif ! [ -f "$1" ]; then
    echo "[ERROR] file $1 not found"
    exit 0
fi

working_dir=`pwd`

# find the number of image frames per core
declare -i pairs_per_job
declare -i pairs_last_job
declare -i pairs
pairs=$((${8} - ${7}))
pairs_per_job=$(($pairs / ${13}))
pairs_last_job=$(($pairs - $pairs_per_job * (${13}-1)))
# submit part of the img file to each core as a separate job for PIV processing
fname=$1
flen=${#fname}-4
fname=${fname[@]:0:$flen}

for ((i=0; i<${13}; i++)); do
	# create symlinks for img file for each job to use
	fname_i[$i]=$(printf '%s.c%04d.img' "$fname" "$i")
	fname_i_tracers[$i]=$(printf '%s.c%04d.tracers.img' "$fname" "$i")
	#if [ -f ${fname_i[$i]} ]; then
		#rm ${fname_i[$i]}
		#echo ${fname_i[$i]}
	#fi
	ln -s $1 ${fname_i[$i]} 
	
	# specify start frame and end frame for each job
    start=$(($i * ${pairs_per_job} + ${7}))
    if [[ $i == $((${13}-1)) ]]; then
        end=$((${start}+${pairs_last_job}))
    else
        end=$((${start}+${pairs_per_job}))
    fi
    #echo ${i} ${start} ${end} ${fname_i_tracers[$i]} 
    id[$i]=`qsub -q ${10} -l walltime=${11},nodes=1:ppn=${9},pmem=${12} -v img_file=${fname_i[$i]},labeling_threshold=${2},particle_threshold=${3},min_size=${4},particle_flare=${5},window_size=${6},start_frame=${start},end_frame=${end},cores=${9} /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/qsub/separate.sh`
done

# ----------------- wait for jobs to finish --------------------

shopt -s expand_aliases
sleep_time=30 # seconds
me=`whoami`
alias myqstat='qstat | grep $me'

# count number of jobs complete
no_complete=0
for ((i=0; i<${13}; i++)); do
	jobstate=`myqstat | grep ${id[$i]}` # check job status
	status=`echo $jobstate | awk -F' ' '{print $5}'`
	if [ "$status" == "C" ]; then
                ((no_complete++))
        fi
done
counter=0
aniwait=("|" "/" "-" "\\")
echo -n ${aniwait[$counter]}

while [ $no_complete -lt ${13} ]; do  # while not all jobs are complete
	no_complete=0
	for ((i=0; i<${13}; i++)); do
        	jobstate=`myqstat | grep ${id[$i]}` # check job status
		status=`echo $jobstate | awk -F' ' '{print $5}'`
		if [ "$status" == "C" ]; then
                	((no_complete++))
        	fi
	done

	sleep $sleep_time
	((counter++))
	echo -n -e "\r${aniwait[$counter%4]}"
done
echo -e "\rFINISHED in $(($counter*$sleep_time)) seconds"

# ----------------------- clean up ------------------------

# join img files
/home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/pio/merge_imgs.py $(printf '%s.tracers.img' "$fname") ${fname_i_tracers[*]}
