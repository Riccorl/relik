#!/bin/bash

#!/bin/bash
#SBATCH --job-name=relik-aida_me5_small_from_blink_01102024            # Job name
#SBATCH -o /leonardo_work/IscrC_MEL/relik-ml/relik-ric/logs/aida_me5_small_from_blink_05102024/aida_me5_small_from_blink_01102024.out
#SBATCH -e /leonardo_work/IscrC_MEL/relik-ml/relik-ric/logs/aida_me5_small_from_blink_05102024/aida_me5_small_from_blink_01102024.err

#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time 24:00:00          # format: HH:MM:SS

#SBATCH -A IscrC_MEL
#SBATCH -p boost_usr_prod #lrd_all_serial


# MODULES="cuda/12.1"

source /leonardo_work/IscrC_MEL/python-envs/relik-thesis-venv/bin/activate

export HF_HOME=/leonardo_work/IscrC_MEL/hf_cache
export HF_DATASETS_CACHE=/leonardo_work/IscrC_MEL/hf_cache
export HUGGINGFACE_HUB_CACHE=/leonardo_work/IscrC_MEL/hf_cache
export WANDB_MODE=offline
# get Huggingface token from python
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

# export NCCL_IB_SL=1
# export UCX_IB_SL=1
# export NVSHMEM_IB_SL=1
# export NVSHMEM_DISABLE_NCCL=1

# HYDRA_FULL_ERROR=1 relik retriever train /leonardo_work/IscrC_MEL/relik-ml/relik-ric/relik/retriever/conf/blink_me5_small.yaml
HYDRA_FULL_ERROR=1 relik retriever train /leonardo_work/IscrC_MEL/relik-ml/relik-ric/relik/retriever/conf/aida_me5_small_from_blink.yaml
