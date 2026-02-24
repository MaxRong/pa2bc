#!/bin/bash  
# Next line shows the job name you can find when querying the job status
#SBATCH --job-name="blasmm"
# Next line is the output file name of the execution log
#SBATCH --output="job.blasmm.%j.out"
# Next line shows where to ask for machine nodes
#SBATCH --partition=shared
#Next line asks for 1 machine node and  1 core per node 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
# Next line limits the job execution time at most 3 minute.
#SBATCH -t 00:03:00
#SBATCH --account=csb175

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
./blasmm
