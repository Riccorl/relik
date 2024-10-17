#!/bin/bash
#SBATCH --job-name=move-data           # Job name
#SBATCH -o /leonardo_scratch/fast/IscrC_TIGeR/relik-ml/relik-ric/logs/move-data.out
#SBATCH -e /leonardo_scratch/fast/IscrC_TIGeR/relik-ml/relik-ric/logs/move-data.err

#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS

#SBATCH -A IscrC_MEL
#SBATCH -p lrd_all_serial #boost_usr_prod #lrd_all_serial

rsync -ah --info=progress2 /leonardo_work/IscrC_MEL/relik-ml /leonardo_scratch/fast/IscrC_TIGeR
