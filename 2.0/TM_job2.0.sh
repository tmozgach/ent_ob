#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-emodata
#SBATCH --mem=50000M
if [[ $1 -eq 0 ]] ; then
   echo 'You did not specify number of topics. Example: sbatch TM_job.sh 30'
   exit 1
fi
source /home/tmozgach/virtualenvironment/bin/activate
echo 'Number of topics:' $1
echo 'File name' $2
python ./TopicModeling2.0.py $1 $2
echo 'It finished'
