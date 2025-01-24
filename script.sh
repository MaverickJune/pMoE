#!/bin/bash
# 
# export LD_LIBRARY_PATH=/shared/workspace/ScheMoE/zfp/build/lib:$LD_LIBRARY_PATH
# export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISABLE_ADDR2LINE=1
expoert CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_BUFFSIZE=134217728
# export NCCL_NTHREADS=256       # More threads
# export NCCL_MIN_NCHANNELS=4    # Minimum channels
# export NCCL_MAX_NCHANNELS=8    # Maximum channels
# export NCCL_ALGO=Ring          # Ring algorithm
# export NCCL_PROTO=Simple       # Simple protocol for large data

# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$(($(nproc --all) / $(nvidia-smi -L | wc -l)))
log_dir=logs_nsys
output_dir=~/$log_dir
mkdir -p $output_dir
nsys_output="$output_dir/profile_ScheMoE${node_rank}-r1.nsys-rep"
# python3 -m torch.distributed.run --nproc_per_node=2 -m schemoe.examples.pre_test --batch_size=16
# python3 -m torch.distributed.run --nproc_per_node=2 -m schemoe.examples.layer_test --batch_size=16



# nodes=("nxc-node0" "nxc-node1" "nxc-node2" "nxc-node3")
# nodes=("nxc-node0" "nxc-node1")
nodes=("nxc-node0")


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

# NCCL_SOCKET_IFNAME=ens4f0np0 python -m torch.distributed.run \
#   --nproc_per_node=$nproc_per_node \
#   --nnodes=$nnodes \
#   --node_rank=$node_rank \
#   --master_addr=10.10.10.10 \
#   --master_port=12357 \
#   -m schemoe.examples.layer_test

# env NCCL_SOCKET_IFNAME=ens4f0np0 

nsys profile --output=$nsys_output \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --force-overwrite true \
  --gpu-metrics-device all \
  python -m torch.distributed.run \
  --nproc_per_node=$nproc_per_node \
  --nnodes=$nnodes \
  --node_rank=$node_rank \
  --master_addr='localhost' \
  --master_port=12357 \
  -m schemoe.examples.layer_test

