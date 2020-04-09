#!/bin/bash
# PIV cross-correlation code 
if [[ $1 == '-h' ]]; then
echo -e "\nusage: python plume_piv.py [-h]
                     piv_file start_frame end_frame cores
                     queue walltime pmem n_jobs

positional arguments: (all are required)
  piv_file              PIV filename
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

# --------------- submit parallel PIV jobs --------------------
#echo {${5}'*.tif'}

# find the number of image frames per core
declare -i pairs_per_job
declare -i pairs_last_job
declare -i pairs
pairs=$((${3} - ${2}-1))
pairs_per_job=$(($pairs/${8}))
pairs_last_job=$(($pairs - $pairs_per_job * (${8}-1)))


# submit part of the piv file to each core as a separate job for PIV processing
pname=$1
plen=${#pname}-4
pname=${pname[@]:0:$plen}

#echo ${pairs} ${pairs_per_job} ${pairs_last_job}
for ((i=0; i<${8}; i++)); do
	# create symlinks for img file for each job to use
	pname_i[$i]=$(printf '%s.c%04d.piv' "$pname" "$i")
	pname_i_flct[$i]=$(printf '%s.c%04d.flct.piv' "$pname" "$i")

    if [ -f ${pname_i[$i]} ]; then
		rm ${pname_i[$i]}
	fi
	ln -s $1 ${pname_i[$i]} 

	# specify start frame and end frame for each job
    start=$(($i * ${pairs_per_job} + ${2}))
    if [[ $i == $((${8}-1)) ]]; then
        end=$((${start}+${pairs_last_job}))
    else
        end=$((${start}+${pairs_per_job}))
    fi
    #echo ${i} ${start} ${end} ${fname_i[$i]} ${pname_i[$i]}
    id[$i]=`qsub -q ${5} -l walltime=${6},nodes=1:ppn=${4},pmem=${7} -v piv_file=${pname_i[$i]},start_frame=${start},end_frame=${end},cores=${4} /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/qsub/fluct_vel.sh`
done

# ----------------- wait for jobs to finish --------------------

shopt -s expand_aliases
sleep_time=30 # seconds
me=`whoami`
alias myqstat='qstat | grep $me'

# count number of jobs complete
no_complete=0
for ((i=0; i<${8}; i++)); do
	jobstate=`myqstat | grep ${id[$i]}` # check job status
	status=`echo $jobstate | awk -F' ' '{print $5}'`
	if [ "$status" == "C" ]; then
                ((no_complete++))
        fi
done
counter=0
aniwait=("|" "/" "-" "\\")
echo -n ${aniwait[$counter]}

while [ $no_complete -lt ${8} ]; do  # while not all jobs are complete
	no_complete=0
	for ((i=0; i<${8}; i++)); do
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
/home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/pio/join_piv.py $(printf '%s.flct.piv' "$pname") ${pname_i_flct[*]}

# delete job files and symlinks
#rm `echo "$fname.c*"` 
#rm `echo "*.sh.o* *.sh.e*"`
 
