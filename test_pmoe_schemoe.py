import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import torch

from torch.utils.data import DataLoader

import argparse, os, random
import gc
import nvtx
import csv

from moe_lib.utils import pMOEdataset, ContextManager, generate_dummy_tokens, collate_fn_batching
from moe_lib.moe_utils import get_model_from_hf, model_wrapper_spmoe

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def custom_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_tokens", type=int, default=512)
    parser.add_argument("--model_dim", type=int, default=8192) # 4096 
    parser.add_argument("--hidden_size", type=int, default=14336) # 14336
    parser.add_argument("--num_local_experts", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--fp32_gate", default=False, action="store_true")
    parser.add_argument("--top", type=int, default=2)
    parser.add_argument("--a2a_ffn_overlap_degree", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--capacity_factor", type=float, default=1.0)
    parser.add_argument("--parallel_type", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_2dh", default=False, action="store_true")
    parser.add_argument("--record_shapes", default=False, action="store_true")
    parser.add_argument("--with_stack", default=False, action="store_true")
    parser.add_argument("--log", type=str, default="test.log")
    parser.add_argument("--encode", type=str, default="no")
    parser.add_argument("--moe_router_topk", type=int, default=1)
    parser.add_argument("--schemoe_compress_name", type=str, default="no")
    parser.add_argument("--schemoe_comm_name", type=str, default="pipe")
    parser.add_argument("--schemoe_overlap_degree", type=int, default=1)
    parser.add_argument("--ffn_hidden_size", type=int, default=4096)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--moe_expert_capacity_factor", type=int, default=1.0)
    parser.add_argument("--gate_path", type=str, default="/home")
    
    # Arguments added for input control
    parser.add_argument("--use_dataloader", default=False, action="store_true")
    parser.add_argument("--dataset_name", type=str, default="wikitext-2")
    parser.add_argument("--custom_input_size", type=int, default=256)
    
    # Arguments added for model control
    parser.add_argument("--partial", type=float, default=0.1)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    
    # Arguments added for iteration control
    parser.add_argument("--iterations", type=int, default=100)
    
    args = parser.parse_args()
    
    return args


def main():
    # Define the logger
    def log(msg):
        if dist_rank == 0:
            print(f"{msg}")
            
    # Set the multi-gpu inference environment
    args = custom_argparser()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dist.init_process_group("nccl")

    dist_rank, dist_world_size = dist.get_rank(), dist.get_world_size()
    log(f"Total # of GPUs(processes): {dist_world_size}")
    args.local_rank = os.environ.get("LOCAL_RANK", 0)
            
    device = torch.device("cuda:%s" % args.local_rank)
    # print(device)
    gpu_idx = dist_rank % torch.cuda.device_count()
    # print(gpu_idx)
    
    torch.cuda.set_device(device)
    torch.set_printoptions(sci_mode=False)
    
    if args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float16":
        torch.set_default_dtype(torch.float16)
    elif args.dtype == "bfloat16":
        torch.set_default_dtype(torch.bfloat16)
    else:
        raise Exception("Unrecognized data type specified: %s" % args.dtype)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    
    # Get the model from the configuration
    model_name = args.model_name
    model_dict = {
        "d_model": args.model_dim,
        "d_hidden": args.hidden_size
    }
    
    log(f"Configuring model with the following parameters: {model_dict}")
    
    model = get_model_from_hf(model_name, partial=args.partial, gpu_idx=gpu_idx, model_dict=model_dict)
    model = model_wrapper_spmoe(model, moe_name="pmoe", world_size=dist_world_size, args=args)
    model.eval()
    
    # Run multiple forward passes for evaluation
    ffn_elapsed_times = []
    iterations = args.iterations
    warmup = 10
    
    if args.use_dataloader:
        dataset = pMOEdataset(dataset_name=args.dataset_name, model_name=args.model_name)
        dataset.prune_dataset(1024)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: collate_fn_batching(batch, dataset.tokenizer))
        
        with torch.no_grad():
            i = 0
            for d in dataloader:
                if i >= warmup:
                    break
                _tokens = d["input_ids"].to(gpu_idx)
                attention_mask = d["attention_mask"].to(gpu_idx)
                _ = model(_tokens)
                i += 1
        
        with torch.no_grad():
            i = 0
            for d in dataloader:
                if i % 10 == 0:
                    if dist_rank == 0:
                        log(f"processing {i}th data")
                        
                if iterations != -1 and i >= iterations:
                    break
                
                # embedding generate
                _tokens = d["input_ids"].to(gpu_idx)
                attention_mask = d["attention_mask"].to(gpu_idx)
                torch.cuda.synchronize()
                start_event.record()
                _ = model(_tokens)
                end_event.record()
                torch.cuda.synchronize()
                
                # save and iterate
                ffn_elapsed_times.append(start_event.elapsed_time(end_event))
                
                # Increase iter count
                i += 1
    else:
        custom_input_size = args.custom_input_size
        random_input = torch.randint(10, 50, (1, custom_input_size)).to(gpu_idx)
        
        with torch.no_grad():
            for i in range(warmup):
                _ = model(random_input)
        
        with torch.no_grad():
            for i in range(iterations):
                if i % 10 == 0:
                    if dist_rank == 0:
                        log(f"processing {i}th data")
                
                torch.cuda.synchronize()
                start_event.record()
                _ = model(random_input)
                end_event.record()
                torch.cuda.synchronize()
                
                ffn_elapsed_times.append(start_event.elapsed_time(end_event))
    
    
    # Calculate the average time taken for the forward pass
    average_elapsed_time = sum(ffn_elapsed_times) / len(ffn_elapsed_times)
    log(f"Average time [pMOE] with {args.iterations}th iterations: {average_elapsed_time} ms")
    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()