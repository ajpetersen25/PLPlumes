#!/bin/bash
# PIV cross-correlation code 
if [[ $1 == '-h' ]]; then
echo -e "\nusage: python plume_piv.py [-h]
                     img_file p_quad p_lin start_frame end_frame cores
                     queue walltime pmem n_jobs working_dir

positional arguments: (all are required)
  img_file              IMG filename for converted to bulk density
  p_quad                txt file of fits for quadratic part of conversion
  p_lin                 txt file of fits for linear part of conversion
  start_frame           start frame for piv
  end_frame             end frame for piv
  cores                 number of cores to use per job


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
pairs=$((${5} - ${4}-1))
pairs_per_job=$(($pairs/${9}))
pairs_last_job=$(($pairs - $pairs_per_job * (${9}-1)))
# submit part of the img file to each core as a separate job for PIV processing
fname=$1
flen=${#fname}-4
fname=${fname[@]:0:$flen}

for ((i=0; i<${9}; i++)); do
	# create symlinks for img file for each job to use
	fname_i[$i]=$(printf '%s.c%04d.img' "$fname" "$i")
	#if [ -f ${fname_i[$i]} ]; then
		#rm ${fname_i[$i]}
		#echo ${fname_i[$i]}
	#fi
	ln -s $1 ${fname_i[$i]} 
	
	# specify start frame and end frame for each job
    start=$(($i * ${pairs_per_job} + ${3}))
    if [[ $i == $((${9}-1)) ]]; then
        end=$((${start}+${pairs_last_job}))
    else
        end=$((${start}+${pairs_per_job}))
    fi
    echo ${i} ${start} ${end}
    id[$i]=`qsub -q ${7} -l walltime=${8},nodes=1:ppn=${6},pmem=${8} -v img_file=${fname_i[$i]},start_frame=${start},end_frame=${end},cores=${6} /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/qsub/write_C_imgs.sh`
done

# ----------------- wait for jobs to finish --------------------

shopt -s expand_aliases
sleep_time=30 # seconds
me=`whoami`
alias myqstat='qstat | grep $me'

# count number of jobs complete
no_complete=0
for ((i=0; i<${9}; i++)); do
	jobstate=`myqstat | grep ${id[$i]}` # check job status
	status=`echo $jobstate | awk -F' ' '{print $5}'`
	if [ "$status" == "C" ]; then
                ((no_complete++))
        fi
done
counter=0
aniwait=("|" "/" "-" "\\")
echo -n ${aniwait[$counter]}

while [ $no_complete -lt ${9} ]; do  # while not all jobs are complete
	no_complete=0
	for ((i=0; i<${9}; i++)); do
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

# join PIV files
/home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/pio/merge_images.py $(printf '%s.rho_b.img' "$fname") ${fname_i[*]}
