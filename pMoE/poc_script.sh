#!/bin/bash

config_dir="/shared/workspace/shan/pMoE/config"
model_config="$config_dir/model"
data_config="$config_dir/data"

export OMP_NUM_THREADS=1       # Limit OpenMP threads to 1
export MKL_NUM_THREADS=1       # Limit MKL threads to 1
export NUMEXPR_NUM_THREADS=1   # Limit NumExpr threads to 1
export TORCH_NUM_THREADS=1     # Limit PyTorch threads to 1
export TRANSFORMERS_VERBOSITY=warning
export HF_DATASETS_VERBOSITY=warning

# Activate the conda environment
# source $conda_path activate $conda_env

nproc_per_node=2 # GPUs
nnodes=1 # # nodes
hostname=$(hostname)

model="transformer-xl"
dataset="enwiki8"


python -m torch.distributed.run --nproc_per_node=$nproc_per_node --nnodes=$nnodes --master_addr="system" --master_port=12366 poc_1.py --model_config $model_config/$model.json --data_config $data_config/$dataset.json
# python main.py --model_config $model_config/$model.json --data_config $data_config/$dataset.json

# OMP_NUM_THREADS=1 nsys profile --export sqlite --trace=cuda,nvtx,osrt -o "logs/profile-$hostname" python -m torch.distributed.run --nproc_per_node=$nproc_per_node --nnodes=$nnodes --master_addr="system" --master_port=12355 main.py

# python $json_py -f $log_dir/profile-$hostname.sqlite -o $log_dir/$hostname.json