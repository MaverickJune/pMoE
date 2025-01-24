#!/bin/bash

export LD_LIBRARY_PATH=/shared/workspace/ScheMoE/zfp/build/lib:$LD_LIBRARY_PATH
# export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISABLE_ADDR2LINE=1

# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BUFFSIZE=134217728
export NCCL_NTHREADS=1       # More threads
export NCCL_MIN_NCHANNELS=4    # Minimum channels
export NCCL_MAX_NCHANNELS=8    # Maximum channels
# export NCCL_ALGO=Ring          # Ring algorithm
# export NCCL_PROTO=Simple       # Simple protocol for large data

# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$(($(nproc --all) / $(nvidia-smi -L | wc -l)))
# echo $OMP_NUM_THREADS
# python3 -m torch.distributed.run --nproc_per_node=2 -m schemoe.examples.pre_test --batch_size=16
python3 -m torch.distributed.run --nproc_per_node=2 -m schemoe.examples.layer_test --gate_path "/shared/workspace/ScheMoE/p_count_selected.csv"


