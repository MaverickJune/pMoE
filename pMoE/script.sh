#!/bin/bash

conda_path="/shared/conda/etc/profile.d/conda.sh"
conda_env="shan_cuda12.1"

nodes=("nxc-node0" "nxc-node1" "nxc-node2" "nxc-node3")
# nodes=("nxc-node0" "nxc-node1")
# nodes=("nxc-node0")

log_dir=logs
config_dir="/shared/workspace/shan/pMoE/config"
model_config="$config_dir/model"
data_config="$config_dir/data"

# NCCL Related
export NCCL_BUFFSIZE=16777216  # 16MB buffer
export NCCL_NTHREADS=256       # More threads
export NCCL_MIN_NCHANNELS=4    # Minimum channels
export NCCL_MAX_NCHANNELS=8    # Maximum channels
export NCCL_ALGO=Ring          # Ring algorithm
export NCCL_PROTO=Simple       # Simple protocol for large data

export OMP_NUM_THREADS=16       # Limit OpenMP threads to 1
export MKL_NUM_THREADS=16      # Limit MKL threads to 1
export NUMEXPR_NUM_THREADS=32   # Limit NumExpr threads to 1
export TORCH_NUM_THREADS=32     # Limit PyTorch threads to 1
export TRANSFORMERS_VERBOSITY=warning
export HF_DATASETS_VERBOSITY=warning

# Activate the conda environment
# source $conda_path activate $conda_env

nproc_per_node=2 # GPUs
nnodes=${#nodes[@]}  # # nodes
hostname=$(hostname)

model="transformer-xl"
dataset="enwiki8"

# Determine node_rank dynamically based on hostname
node_rank=-1
for i in "${!nodes[@]}"; do
  echo "Checking hostname $hostname against ${nodes[$i]} with total nodes: $nnodes"
  if [[ $hostname == *"${nodes[$i]}"* ]]; then
    node_rank=$i
    break
  fi
done

# Error handling if node_rank is not set
if [[ $node_rank -lt 0 ]]; then
  echo "Error: Hostname $hostname not recognized $node_rank. Please add it to the nodes array."
  exit 1
fi

echo "Running on node: $hostname with node_rank: $node_rank"

NCCL_SOCKET_IFNAME=ens4f0np0 python -m torch.distributed.run \
  --nproc_per_node=$nproc_per_node \
  --nnodes=$nnodes \
  --node_rank=$node_rank \
  --master_addr=10.10.10.10 \
  --master_port=12357 \
  main.py \
  --model_config $model_config/$model.json \
  --data_config $data_config/$dataset.json
