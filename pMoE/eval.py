import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn import Transformer
from lib.utils import pMOEdataset, ContextManager, generate_dummy_tokens
from lib.model import MoETransformer, pMoETransformer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from fmoe.transformer import pMoE, pMoETransformerMLP, FMoETransformerMLP, FMoE
import argparse
import json
import os
import torch
import sys

import logging
logging.basicConfig(level=logging.DEBUG)

# To debug
from torch.profiler import profile, record_function, ProfilerActivity

def cleanup():
    """
    Destroy the process group.
    """
    dist.destroy_process_group()
    
def setup():
    """
    Setup the distributed environment using environment variables set by torchrun.
    """
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for GPU communication
        init_method="env://"  # Torchrun sets environment variables automatically
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def parse(parameters):
    """
    Parse command-line and configuration file arguments.
    Args:
        parameters (list): List of command-line arguments passed to the function.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Define argument help descriptions
    argument_help = {
        "data_config": "Path to the data configuration file",
        "model_config": "Path to the model configuration file",
        "dataset": "Dataset name",
        "n_layer": "Number of total layers",
        "n_head": "Number of attention heads",
        "d_model": "Model dimension",
        "d_hidden": "Hidden dimension size",
        "lr": "Initial learning rate",
        "batch_size": "Batch size for training"
    }

    # Define command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--model_config', type=str, default='model/llama-7b.json', help=argument_help["model_config"])
    parser.add_argument('--data_config', type=str, default='data/wikitext.json', help=argument_help["data_config"])

    # Parse arguments
    args, unknown = parser.parse_known_args()

    # Parse model configuration file
    with open(args.model_config, 'r') as model_file:
        model_config_args = json.load(model_file)

    # Parse data configuration file
    with open(args.data_config, 'r') as data_file:
        data_config_args = json.load(data_file)

    # Combine configurations (optional: prioritize model configs over data configs if keys overlap)
    combined_config_args = {**data_config_args, **model_config_args}

    # Add arguments from the configuration files to the parser with proper help descriptions
    for key, value in combined_config_args.items():
        help_text = argument_help.get(key, f'No description provided for {key}')
        parser.add_argument(f'--{key}', default=value, help=help_text)

    # Final parsed arguments
    final_args = parser.parse_args()
    print(final_args)
    
    return final_args

def main(args, mesh_shape, mesh_dims, input):
    """
    Function to initialize distributed environment and set up device mesh.
    """
    assert len(mesh_shape) == len(mesh_dims), "should be same mesh shape and mesh dim"
    
    top_k = 1
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # setup GPUs
    gpu_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_idx)
    
    assert gpu_idx == local_rank, "gpu_idx should be equal to local_rank"
    
    print(f"gpuidx: {gpu_idx} with rank: {rank}")
    
    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    print(f"rank: {rank}, start_event: {start_event.device}, end_event: {end_event.device}")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                  
    try:
        if rank == 0:
            print(f"initialized the distributed DNN")
        
        # Initialize device mesh
        ctx = ContextManager(
            rank=rank, world_size=world_size,
            mesh_shape=mesh_shape, mesh_dim_names=mesh_dims,
            backend='nccl'
        )
        if rank == 0:
            print(f"ctx manager on")
            
        total_experts = 32 # the number of total experts
        d_model = 1024 # the embedding size
        d_hidden = 4096 # the hidden dimension size
        
        # Define MoE Model
        model = pMoETransformer(num_heads=args.n_head, d_model=args.d_model, d_hidden=args.d_hidden, num_layers=args.n_layers, mlp=pMoETransformerMLP, total_experts=total_experts, top_k=top_k, ctx=ctx)
        ffn = pMoETransformerMLP(total_experts, d_model, d_hidden, top_k=top_k,ctx=ctx).to(gpu_idx)
        ffn.eval()
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
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        
        # time eval
        elapsed_time_ms = start_event.elapsed_time(end_event) / 10
        print(f"PMOE: [Rank {rank}] Elapsed Time: {elapsed_time_ms} ms \n")
        print(f"PMOE: [Rank {rank}] \n", prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace("logs/trace_" + str(rank) + ".json")
    
    except Exception as e:
        raise e
    
    finally:
        cleanup()

def baseline(mesh_shape, mesh_dims, input):
    """
    Function of baseline model.
    """
    assert len(mesh_shape) == len(mesh_dims), "should be same mesh shape and mesh dim"
    
    top_k = 1
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # setup GPUs
    gpu_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_idx)
    
    assert gpu_idx == local_rank, "gpu_idx should be equal to local_rank"
    
    print(f"gpuidx: {gpu_idx} with rank: {rank}")
    
    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    try:
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
        num_heads, d_model, d_hidden, num_layers, mlp, num_experts, top_k, world_size, group
        model = MoETransformer(num_heads=args.n_head, d_model=args.d_model, d_hidden=args.d_hidden, num_layers=args.n_layers, mlp=FMoETransformerMLP, total_experts=num_experts, top_k=top_k, ctx=ctx)
        ffn = FMoETransformerMLP(num_expert=num_experts, d_model=d_model, d_hidden=d_hidden, top_k=top_k, world_size=world_size, moe_group=group).to(gpu_idx)
        ffn.eval()
        model = M
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
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        # time eval
        elapsed_time_ms = start_event.elapsed_time(end_event) / 10
        print(f"FMOE: [Rank {rank}] Elapsed Time: {elapsed_time_ms} ms \n")
        print(f"FMOE: [Rank {rank}] \n", prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace("logs/fmoe_trace_" + str(rank) + ".json")
        
        
    
    except Exception as e:
        raise e
    
    finally:
        cleanup()
    
if __name__ == "__main__":
    args = parse(sys.argv[1:])
    
    # Generate dummy input
    # args = argparse.Namespace(batch=100, d_model=1024)
    input_data = generate_dummy_tokens(args.batch_size, args.d_model)
      
    # Distributed setup
    local_rank = setup()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    num_gpus = torch.cuda.device_count()

    tp_size = world_size 
    dp_size = world_size // num_gpus # number of nodes
    
    mesh_shape = (dp_size, tp_size)
    mesh_dims = ("dp", "tp")    
    
    # Define Dataset and DataLoader
    # Set up the distributed DataLoader
    dataset = pMOEdataset(dataset_name=args.dataset, model_name=args.model_name)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    # Spawn processes for distributed inference
    
    print(f"[Rank {rank}] World size: {world_size}, Local rank: {local_rank}, shape: {mesh_shape}")
    
    # batch_idx, (input_ids, attention_masks) = enumerate(dataloader)
    
    
    # for batch_idx, texts in enumerate(dataloader):
    #     print(f"[Rank {rank}] Batch {batch_idx}: with Text len: {len(texts)}")
    # Iterate over the DataLoader
    for batch_idx, (input_ids, attention_masks) in enumerate(dataloader):
        print(f"[Rank {rank}] Batch {batch_idx} Input IDs shape: {input_ids.shape}, Attention Mask shape: {attention_masks.shape}")

    # Convert DataLoader to a list for analysis (debugging only; avoid in distributed scenarios)
    dataloader_list = list(dataloader)
    print(f"[Rank {rank}] Data loader shape: {len(dataloader_list)} with batch size: {len(dataloader_list[0])}")
        
    # main(args, mesh_shape=mesh_shape, mesh_dims=mesh_dims, input=input_data)
    # baseline(args, mesh_shape=mesh_shape, mesh_dims=mesh_dims, input=input_data)

