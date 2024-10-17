#!/bin/bash
#SBATCH --job-name=sync-wand           # Job name
#SBATCH -o /leonardo_scratch/fast/IscrC_TIGeR/relik-ml/relik-ric/logs/sync-wand.out
#SBATCH -e /leonardo_scratch/fast/IscrC_TIGeR/relik-ml/relik-ric/logs/sync-wand.err

#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS

#SBATCH -A IscrC_MEL
#SBATCH -p lrd_all_serial #boost_usr_prod #lrd_all_serial


# folder to sync
# FOLDER=$1
FOLDER=/leonardo_scratch/fast/IscrC_TIGeR/relik-ml/relik-ric/wandb

if [ -z "$FOLDER" ]; then
    echo "Please provide a folder to sync"
    exit 1
fi

# check if it is a wandb folder or a run folder
if [[ $FOLDER == *"run"* ]]; then
    # sync run folder
    echo "Syncing run folder $FOLDER"
    wandb sync $FOLDER
else
    # iterate over all runs in the folder
    echo "Syncing all runs in folder $FOLDER"
    for run in $(ls $FOLDER); do
        # sync run folder
        echo "Syncing run folder $FOLDER/$run"
        wandb sync -p minerva-7b-bin $FOLDER/$run
    done
fi
