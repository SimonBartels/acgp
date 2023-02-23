#!/bin/bash
#SBATCH --job-name=Compressor
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=8
#SBATCH --time=1-01:00:00
cd ./output/results/
tar --exclude=parameters* -czf ~/opt_results.tar.gz .

