#!/bin/bash

conda_path="/shared/conda/etc/profile.d/conda.sh"
conda_env="shan_cuda12.1"

# Node list
nodes=("nxc-node0" "nxc-node1" "nxc-node2" "nxc-node3")
log_dir=logs
config_dir="/shared/workspace/shan/pMoE/config"
model_config="$config_dir/model"
data_config="$config_dir/data"

# Export environment variables for performance tuning
export OMP_NUM_THREADS=1       # Limit OpenMP threads to 1
export MKL_NUM_THREADS=1       # Limit MKL threads to 1
export NUMEXPR_NUM_THREADS=1   # Limit NumExpr threads to 1
export TORCH_NUM_THREADS=1     # Limit PyTorch threads to 1
export TRANSFORMERS_VERBOSITY=warning
export HF_DATASETS_VERBOSITY=warning

# Distributed configuration
nproc_per_node=2  # GPUs per node
nnodes=${#nodes[@]} # Total number of nodes
master_addr="${nodes[0]}" # Set the master node as the first node in the list
master_port=12355
hostname=$(hostname)

model="transformer-xl"
dataset="enwiki8"

# Activate the conda environment
source $conda_path activate $conda_env

# Start torchrun via SSH on the master node
if [[ "$hostname" != "${nodes[0]}" ]]; then
    echo "This is not the master node. Connecting to the master node: ${nodes[0]}"
    ssh -A ${nodes[0]} "bash -s" << EOF
        source $conda_path activate $conda_env
        echo "Starting distributed inference from master node (${nodes[0]})"
        for node in "${nodes[@]}"; do
            if [[ "\$node" == "${nodes[0]}" ]]; then
                echo "Starting torchrun locally on \$node"
                python -m torch.distributed.run \
                    --nproc_per_node=$nproc_per_node \
                    --nnodes=$nnodes \
                    --master_addr=$master_addr \
                    --master_port=$master_port \
                    main.py \
                    --model_config $model_config/$model.json \
                    --data_config $data_config/$dataset.json &
            else
                echo "Starting torchrun on \$node via SSH"
                ssh -o ForwardAgent=yes -A \$node "source $conda_path activate $conda_env && python -m torch.distributed.run \
                    --nproc_per_node=$nproc_per_node \
                    --nnodes=$nnodes \
                    --master_addr=$master_addr \
                    --master_port=$master_port \
                    main.py \
                    --model_config $model_config/$model.json \
                    --data_config $data_config/$dataset.json" &
            fi
        done
        wait
EOF
else
    echo "This script is running on the master node: ${nodes[0]}"
    echo "Starting distributed inference from master node ($hostname)"
    for node in "${nodes[@]}"; do
        if [[ "$node" == "$hostname" ]]; then
            echo "Starting torchrun locally on $node"
            python -m torch.distributed.run \
                --nproc_per_node=$nproc_per_node \
                --nnodes=$nnodes \
                --master_addr=$master_addr \
                --master_port=$master_port \
                main.py \
                --model_config $model_config/$model.json \
                --data_config $data_config/$dataset.json &
        else
            echo "Starting torchrun on $node via SSH"
            ssh -o ForwardAgent=yes -A $node "source $conda_path activate $conda_env && python -m torch.distributed.run \
                --nproc_per_node=$nproc_per_node \
                --nnodes=$nnodes \
                --master_addr=$master_addr \
                --master_port=$master_port \
                main.py \
                --model_config $model_config/$model.json \
                --data_config $data_config/$dataset.json" &
        fi
    done
    wait
fi
# Optional: Profiling command (uncomment to use)
# OMP_NUM_THREADS=1 nsys profile --export sqlite --trace=cuda,nvtx,osrt -o "$log_dir/profile-$hostname" python -m torch.distributed.run \
#     --nproc_per_node=$nproc_per_node \
#     --nnodes=$nnodes \
#     --master_addr=$master_addr \
#     --master_port=$master_port \
#     main.py

# Optional: Convert profiling logs to JSON (uncomment to use)
# python $json_py -f $log_dir/profile-$hostname.sqlite -o $log_dir/$hostname.json
