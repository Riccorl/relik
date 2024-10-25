#!/bin/bash

#!/bin/bash
#SBATCH --job-name=me5-base-dual-wikipedia-inorder-nohn-mewsli3Meval            # Job name
#SBATCH -o /leonardo_scratch/fast/IscrC_TIGeR/relik-ml/relik-ric/logs/me5-large-wikipedia-inorder-nohn-mewsli3Meval-23102024/train.out
#SBATCH -e /leonardo_scratch/fast/IscrC_TIGeR/relik-ml/relik-ric/logs/me5-large-wikipedia-inorder-nohn-mewsli3Meval-23102024/train.err

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

HYDRA_FULL_ERROR=1 relik retriever train /leonardo_scratch/fast/IscrC_TIGeR/relik-ml/relik-ric/relik/retriever/conf/wikipedia_me5_large.yaml
