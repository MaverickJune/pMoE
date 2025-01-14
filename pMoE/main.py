import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

import numpy as np
import gc

from torch.nn import Transformer
from lib.utils import pMOEdataset, ContextManager, generate_dummy_tokens, collate_fn_batching
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from fmoe.transformer import pMoE, pMoETransformerMLP, FMoETransformerMLP, FMoE
from lib.moe_utils import llama_wrapper, get_model_from_hf, txl_wrapper, tinymix_wrapper, load_tinymix 
import argparse
import json
import os
import torch
import sys

import logging
logging.basicConfig(level=logging.DEBUG)

# To debug
from torch.profiler import profile, record_function, ProfilerActivity
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
# torch.set_num_threads(32)  # Replace 32 with the desired number of threads
# torch.set_num_interop_threads(8)  # Adjust interop threads if required

import nvtx

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

    # # Parse model configuration file
    # with open(args.model_config, 'r') as model_file:
    #     model_config_args = json.load(model_file)

    # # Parse data configuration file
    # with open(args.data_config, 'r') as data_file:
    #     data_config_args = json.load(data_file)

    # # Combine configurations (optional: prioritize model configs over data configs if keys overlap)
    # combined_config_args = {**data_config_args, **model_config_args}

    # # Add arguments from the configuration files to the parser with proper help descriptions
    # for key, value in combined_config_args.items():
    #     help_text = argument_help.get(key, f'No description provided for {key}')
    #     parser.add_argument(f'--{key}', default=value, help=help_text)

    # Final parsed arguments
    final_args = parser.parse_args()
    # print(final_args)
    
    return final_args

def test():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id)

    text = "Hello my name is"
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def main(args, mesh_shape, mesh_dims, dataloader, iteration=10):
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
    
    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    # llama_wrapper(model, moe_name, moe_config: Dict, ctx, gpu_idx):
    # _model = get_model_from_hf("meta-llama/Llama-3.1-70B-Instruct", partial=0.1)
    
    try:
        if rank == 0:
            print(f"initialized the distributed DNN")
        
        # Initialize device mesh
        ctx = ContextManager(
            rank=rank, world_size=world_size,
            mesh_shape=mesh_shape, mesh_dim_names=mesh_dims,
            backend='nccl'
        )
        # if rank == 0:
        #     print(f"ctx manager on")
        # group = ctx.get_group('tp')
        gpu_rank = ctx.get_rank('tp') # node 0 gpu 0 -> o node 1 gpu 0 -> 2 
        
        total_experts = 8 # the number of total experts
        d_model = 2048  # the embedding size
        d_hidden = 5632 # the hidden dimension size
        
        # Define MoE Model
        # model = llama_wrapper(_model, "pMoE", {"total_experts": total_experts, "d_model": 4096, "d_hidden": 14336, "top_k": 2}, ctx, gpu_idx).to(gpu_idx)
        model = load_tinymix(gpu_idx)
        model = tinymix_wrapper(model, "pMoE", {"total_experts": total_experts, "d_model": 2048, "d_hidden": 5632, "top_k": 2}, ctx, gpu_rank, gpu_idx).to(gpu_idx)
        model.eval()
        
        # ffn = pMoETransformerMLP(total_experts, args.d_model, args.d_hidden, top_k=top_k, ctx=ctx).to(gpu_idx)
        # embedding = embedding.to(gpu_idx)
        
        # ffn.eval()
        # embedding.eval()
        
        warmup = 10        
        with torch.no_grad():
            i = 0
            for d in dataloader:
                if i >= warmup:
                    break
                _tokens = d["input_ids"].to(gpu_idx)
                attention_mask = d["attention_mask"].to(gpu_idx)
                # embs = embedding(_tokens)
                output = model(_tokens)
                i += 1
        
        ffn_elapsed_times=[]
        # cuda.cudaProfilerStart()
        with torch.no_grad():
            i = 0
            # with profile(activities=activities, record_shapes=True, with_modules=True, with_flops=True, profile_memory=True) as prof:
            # @nvtx.annotate("pMoE")
            # with nvtx.annotate("pMoE", color="green"):
            for d in dataloader:
                if i % 10 == 0:
                    if rank == 0:
                        print(f"processing {i}th data")
                
                if iteration != -1 and i >= iteration:
                    break
                
                # embedding generate
                _tokens = d["input_ids"].to(gpu_idx)
                attention_mask = d["attention_mask"].to(gpu_idx)
                # embs = embedding(_tokens)
                # if rank ==0:
                #     print(f"tokens {_tokens.shape}")
                # record only ffn
                torch.cuda.synchronize()
                start_event.record()
                output = model(_tokens)
                end_event.record()
                torch.cuda.synchronize()
                
                # save and iterate
                ffn_elapsed_times.append(start_event.elapsed_time(end_event))
                i += 1
        
        # cuda.cudaProfilerStop()
        # time eval
        average_elapsed_time = sum(ffn_elapsed_times) / len(ffn_elapsed_times)
        print(f"PMOE: [Rank {rank}] Average Elapsed Time: {average_elapsed_time} ms \n")
        # print(f"PMOE: [Rank {rank}] \n", prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # prof.export_chrome_trace("logs/trace_" + str(rank) + ".json")
        return ffn_elapsed_times
    
    except Exception as e:
        raise e
    
    finally:
        del model
        torch.cuda.empty_cache()
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
        cleanup()

def baseline(args, mesh_shape, mesh_dims, dataloader, iteration=10):
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
    
    # Variable to save gate data
    gate_topk_and_latency = []
    
    assert gpu_idx == local_rank, "gpu_idx should be equal to local_rank"
    
    print(f"gpuidx: {gpu_idx} with rank: {rank} with world size: {world_size}")
    
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
        gpu_rank = ctx.get_rank('tp') # node 0 gpu 0 -> o node 1 gpu 0 -> 2
        
        num_experts = 8 // world_size # the number of total experts
        d_model = 2048  # the embedding size
        d_hidden = 5632 # the hidden dimension size
        
        # Define MoE Model
        model = load_tinymix(gpu_idx)
        model = tinymix_wrapper(model, "fMoE", {"total_experts": num_experts, "d_model": 2048, "d_hidden": 5632, "top_k": 2, "world_size": world_size, "moe_group": group}, ctx, gpu_rank, gpu_idx).to(gpu_idx)
        model.eval()
        
        # model = llama_wrapper(_model, "pMoE", {"total_experts": num_experts, "d_model": 8192, "d_hidden": 28672, "top_k": 2}, ctx, gpu_idx)
        
        # ffn = FMoETransformerMLP(num_expert=num_experts, d_model=args.d_model, d_hidden=args.d_hidden, top_k=top_k, world_size=world_size, moe_group=group).to(gpu_idx)
        # embedding = embedding.to(gpu_idx)
        
        # model.eval()
        # embedding.eval()
        
        warmup = 10        
        with torch.no_grad():
            i = 0
            for d in dataloader:
                if i >= warmup:
                    break
                _tokens = d["input_ids"].to(gpu_idx)
                attention_mask = d["attention_mask"].to(gpu_idx)
                # embs = embedding(_tokens)
                output = model(_tokens)
                i += 1
        
        ffn_elapsed_times=[]
        # cuda.cudaProfilerStart()
        GATE_DATA_SAVE_PATH = "~/fMoE/gate_data"
        os.makedirs(GATE_DATA_SAVE_PATH, exist_ok=True)
        
        with torch.no_grad():
            i = 0
            # with profile(activities=activities, record_shapes=True, with_modules=True, with_flops=True, profile_memory=True) as prof:
            # with nvtx.annotate("FMoE", color="red"):
            for d in dataloader:
                if i % 10 == 0:
                    if rank == 0:
                        print(f"processing {i}th data")
                        
                if iteration != -1 and i >= iteration:
                    break
                
                # embedding generate
                _tokens = d["input_ids"].to(gpu_idx)
                attention_mask = d["attention_mask"].to(gpu_idx)
                # if rank==0:
                #     print(f"tokens {_tokens.shape}")
                # embs = embedding(_tokens)
                
                # record only ffn
                torch.cuda.synchronize()
                start_event.record()
                output = model(_tokens)
                end_event.record()
                torch.cuda.synchronize()
                
                # save and iterate
                ffn_elapsed_times.append(start_event.elapsed_time(end_event))
                
                # save gate data
                if rank == 0:
                    save_dict = {}
                    save_dict[f"iter_{i}"] = i
                    for idx in range(len(model.model.layers)):
                        gate_data = model.model.layers[idx].block_sparse_moe.save_gate_data
                        save_dict[f"layer_{idx}_gate"] = gate_data
                    save_dict[f"latency"] = ffn_elapsed_times[-1]
                    gate_topk_and_latency.append(save_dict)
                
                # Increase iter count
                i += 1
                
        # Save gate data to .pt file
        if rank == 0:
            final_save_path = os.path.join(GATE_DATA_SAVE_PATH, f"{args.d_name}_{iteration}.pt")
            torch.save(gate_topk_and_latency, final_save_path)
        
        # cuda.cudaProfilerStop()
        # time eval
        average_elapsed_time = sum(ffn_elapsed_times) / len(ffn_elapsed_times)
        print(f"FMOE: [Rank {rank}] Average Elapsed Time: {average_elapsed_time}ms \n")
        # print(f"FMOE: [Rank {rank}] \n", prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # prof.export_chrome_trace("logs/fmoe_trace_" + str(rank) + ".json")
        return ffn_elapsed_times
        
        
    
    except Exception as e:
        raise e
    
    finally:
        cleanup()
    
if __name__ == "__main__":
    args = parse(sys.argv[1:])
    
    # Distributed setup
    local_rank = setup()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    num_gpus = torch.cuda.device_count()
    
    tp_size = world_size 
    dp_size = world_size // tp_size # number of nodes
    # print(f"debugging: rank: {rank}, world_size: {world_size}, num_gpus: {num_gpus}, tp_size: {tp_size}, dp_size: {dp_size}")

    mesh_shape = (dp_size, tp_size)
    mesh_dims = ("dp", "tp")  
    # Define Dataset and DataLoader
    # Set up the distributed DataLoader
    d_name ="enwik8" # wikitext-103, enwik8, wikitext-2
    args.d_name = d_name
    
    dataset = pMOEdataset(dataset_name=d_name, model_name="eastwind/tinymix-8x1b-chat")
    dataset.prune_dataset(1024) # prune items that are longer than 1024 tokens
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: collate_fn_batching(batch, dataset.tokenizer))
    
    
    # emb = dataset.emb
    # embedding = torch.load("adaptive_embeddings.pt") # 1024 tokenizer + embedding t

    # pmoe_time = main(args, mesh_shape=mesh_shape, mesh_dims=mesh_dims, dataloader=dataloader, iteration=10000)
    fmoe_time = baseline(args, mesh_shape=mesh_shape, mesh_dims=mesh_dims, dataloader=dataloader, iteration=10000)
    # _pmoe = torch.tensor(pmoe_time) 
    _fmoe = torch.tensor(fmoe_time)
    # comp = _pmoe / _fmoe
    
    # if rank == 0:
    #     print(f"pmoe vs. fmoe {torch.mean(comp)}")
    #     print(f"pmoe vs. fmoe value {comp}")
    #     print(f"pmoe time {torch.mean(_pmoe)}")
    
    # # mp.spawn(main, args=(world_size, mesh_shape, mesh_dims, input_data), nprocs=world_size, join=True)
    # # mp.spawn(baseline, args=(world_size, mesh_shape, mesh_dims, input_data), nprocs=world_size, join=True)


