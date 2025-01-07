import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn import Transformer
from lib.utils import init_distributed, ContextManager, generate_dummy_tokens

from torch.distributed.device_mesh import init_device_mesh

from fmoe.transformer import pMoE, pMoETransformerMLP, FMoETransformerMLP, FMoE
import argparse
import json
import os
import torch

import logging
logging.basicConfig(level=logging.DEBUG)

# To debug
from torch.profiler import profile, record_function, ProfilerActivity

def setup(rank, world_size):
    """
    Initialize the process group for the NCCL backend.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group(
        backend='nccl', rank=rank, world_size=world_size, init_method='env://'
    )

    # Assign each rank to a GPU
    torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup():
    """
    Destroy the process group.
    """
    dist.destroy_process_group()

def main(rank, world_size, mesh_shape, mesh_dims, input):
    top_k = 1
    """
    Function to initialize distributed environment and set up device mesh.
    """
    assert len(mesh_shape) == len(mesh_dims), "should be same mesh shape and mesh dim"
    
    # setup GPUs
    gpu_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_idx)
    
    # Set environment variables for the process
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Use localhost for single-node training
    os.environ["MASTER_PORT"] = "29500"      # Default port for torch.distributed
    
    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                  
    try:
        # Initialize the process group
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        if rank == 0:
            print(f"initialized the distributed DNN")
        
        # Initialize device mesh
        ctx = ContextManager(
            rank=rank, world_size=world_size,
            mesh_shape=mesh_shape, mesh_dim_names=mesh_dims,
            backend='nccl'
        )
            
        total_experts = 32 # the number of total experts
        d_model = 1024 # the embedding size
        d_hidden = 4096 # the hidden dimension size
        
        # Define MoE Model
        ffn = pMoETransformerMLP(total_experts, d_model, d_hidden, top_k=top_k,ctx=ctx).to(gpu_idx)
        ffn.eval()
        # dummy = generate_dummy_tokens(100, d_model).to(gpu_idx)
        input = input.to(gpu_idx)
    
        # warm up
        with torch.no_grad():
            for _ in range(10):
                output = ffn(input)
        
        # time evaluation
        torch.cuda.synchronize()
        start_event.record()
        with profile(activities=activities, record_shapes=True, with_modules= True,with_flops=True, profile_memory=True) as prof:
            with torch.no_grad():
                for _ in range(20):
                    output = ffn(input)
        end_event.record()
        torch.cuda.synchronize()
        
        
        # time eval
        elapsed_time_ms = start_event.elapsed_time(end_event) / 10
        print(f"PMOE: [Rank {rank}] Elapsed Time: {elapsed_time_ms} ms \n")
        print(f"PMOE: [Rank {rank}] \n", prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace("logs/trace_" + str(rank) + ".json")
    
    except Exception as e:
        raise e
    
    finally:
        cleanup()

def baseline(rank, world_size, mesh_shape, mesh_dims, input):
    """
    Function of baseline model.
    """
    top_k = 1
    # setup GPUs
    gpu_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_idx)
    
    # Set environment variables for the process
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Use localhost for single-node training
    os.environ["MASTER_PORT"] = "29500"      # Default port for torch.distributed
    
    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    try:
        # Initialize the process group
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        if rank == 0:
            print(f"initialized the distributed DNN")
        
        # Initialize device mesh
        ctx = ContextManager(
            rank=rank, world_size=world_size,
            mesh_shape=mesh_shape, mesh_dim_names=mesh_dims,
            backend='nccl'
        )
        
        group = ctx.get_group('tp')
        
        num_experts = 32 // world_size # the number of total experts
        d_model = 1024 # the embedding size
        d_hidden = 4096 # the hidden dimension size
        
        # Define MoE Model
        ffn = FMoETransformerMLP(num_expert=num_experts, d_model=d_model, d_hidden=d_hidden, top_k=top_k, world_size=world_size, moe_group=group).to(gpu_idx)
        ffn.eval()
        input = input.to(gpu_idx)
    
    
        # warm up
        with torch.no_grad():
            for _ in range(10):
                output = ffn(input)
        
        # time evaluation
        torch.cuda.synchronize()
        start_event.record()
        with profile(activities=activities, record_shapes=True, with_modules= True, with_flops=True, profile_memory=True) as prof:
            with torch.no_grad():
                for _ in range(20):
                    output = ffn(input)
        end_event.record()
        torch.cuda.synchronize()
        
        # time eval
        elapsed_time_ms = start_event.elapsed_time(end_event) / 10
        print(f"FMOE: [Rank {rank}] Elapsed Time: {elapsed_time_ms} ms \n")
        print(f"FMOE: [Rank {rank}] \n", prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace("logs/fmoe_trace_" + str(rank) + ".json")
        
        
    
    except Exception as e:
        raise e
    
    finally:
        cleanup()
        
if __name__ == "__main__":
    import os

    # Set environment variables for multi-node setup
    os.environ["MASTER_ADDR"] = "nxc-node0"  # Replace with the master node's IP
    os.environ["MASTER_PORT"] = "12355"  # Use a free port

    world_size = 2  # Total number of ranks (across nodes and GPUs)
    num_gpus = torch.cuda.device_count()

    mesh_shape = (1, 2)
    mesh_dims = ("dp", "tp")

    print(f"world_size: {world_size}, num_gpus: {num_gpus}")

    if num_gpus < world_size:
        print("This example requires at least 2 GPUs.")
        exit(1)

    # Generate dummy input
    args = argparse.Namespace(batch=100, d_model=512)
    input_data = generate_dummy_tokens(args.batch, args.d_model)

    # Spawn processes for distributed inference
    mp.spawn(main, args=(world_size, mesh_shape, mesh_dims, input_data), nprocs=world_size, join=True)
    mp.spawn(baseline, args=(world_size, mesh_shape, mesh_dims, input_data), nprocs=world_size, join=True)
