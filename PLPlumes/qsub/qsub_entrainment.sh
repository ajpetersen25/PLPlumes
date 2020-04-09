#!/bin/bash
# PIV cross-correlation code 
if [[ $1 == '-h' ]]; then
echo -e "\nusage: python plume_piv.py [-h]
                     img_file piv_file dilation_kernel dilation_iter threshold med_filt_size
                     start_frame end_frame orientation cores queue walltime pmem n_jobs

positional arguments: (all are required)
  img_file              IMG filename containing bulk density values
  piv_file              PIV filename
  dilation_kernel
  dilation_iter
  threshold             image intensity threshold
  med_filt_size
  start_frame           start frame for piv
  end_frame             end frame for piv
  orientation           camera frame orientation
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
elif ! [ -f "$2" ]; then
    echo "[ERROR] file $2 not found"
    exit 0
fi
working_dir=`pwd`

# --------------- submit parallel PIV jobs --------------------
#echo {${5}'*.tif'}

# find the number of image frames per core
declare -i pairs_per_job
declare -i pairs_last_job
declare -i pairs
pairs=$((${8} - ${7}))
pairs_per_job=$(($pairs/${14}))
pairs_last_job=$(($pairs - $pairs_per_job * (${14}-1)))

# submit part of the img file to each core as a separate job for PIV processing
fname=$1
flen=${#fname}-4
fname=${fname[@]:0:$flen}

# submit part of the piv file to each core as a separate job for PIV processing
pname=$2
plen=${#pname}-4
pname=${pname[@]:0:$plen}

#echo ${pairs} ${pairs_per_job} ${pairs_last_job}
for ((i=0; i<${14}; i++)); do
	# create symlinks for img file for each job to use
	#fname_i[$i]=$(printf '%s.c%04d.img' "$fname" "$i")
	pname_i[$i]=$(printf '%s.c%04d.piv' "$pname" "$i")
	pname_i_hdf5[$i]=$(printf '%s.c%04d.u_e.hdf5' "$pname" "$i")
    if [ -f ${pname_i[$i]} ]; then
		rm ${pname_i[$i]}
	fi
	ln -s $2 ${pname_i[$i]} 

	# specify start frame and end frame for each job
    start=$(($i * ${pairs_per_job} + ${7}))
    if [[ $i == $((${14}-1)) ]]; then
        end=$((${start}+${pairs_last_job}))
    else
        end=$((${start}+${pairs_per_job}))
    fi
    #echo ${i} ${start} ${end} ${pname_i[$i]}
    id[$i]=`qsub -q ${11} -l walltime=${12},nodes=1:ppn=${10},pmem=${13} -v img_file=${1},piv_file=${pname_i[$i]},dilation_kernel=${3},dilation_iter=${4},threshold=${5},med_filt_size=${6},start_frame=${start},end_frame=${end},orientation=${9},cores=${10} /home/colettif/pet00105/Coletti/PLPlumes/PLPlumes/qsub/entrainment.sh`
done


 
