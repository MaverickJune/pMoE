import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import torch

import statistics

from torch.utils.data import DataLoader

import argparse, os, random
import gc
import nvtx
import csv
import json
from datetime import datetime

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
    
    # Arguments added for logging the results
    parser.add_argument("--log_results", default=False, action="store_true")
    
    # Arguments for decoding phase test
    parser.add_argument("--decode", type=int, default=-1)
    
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
    
    if args.use_dataloader:
        dataset = pMOEdataset(dataset_name=args.dataset_name, model_name=args.model_name)
        dataset.prune_dataset(1024)
        pad_token_id = dataset.tokenizer.pad_token_id
    
    log(f"Configuring model with the following parameters: {model_dict}")
    
    model = get_model_from_hf(model_name, partial=args.partial, gpu_idx=gpu_idx, model_dict=model_dict, enable_cache=False, pad_token_id=pad_token_id)
    model = model_wrapper_spmoe(model, moe_name="pmoe", world_size=dist_world_size, args=args)
    model.eval()
    
    # Run multiple forward passes for evaluation
    ffn_handled_tokens = []
    ffn_elapsed_times = []
    iterations = args.iterations
    warmup = 10
    
    if args.use_dataloader:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn_batching(batch, dataset.tokenizer))
        
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
                if iterations != -1 and i >= iterations:
                    break
                
                if i % 10 == 0:
                    if dist_rank == 0:
                        log(f"processing {i}th batch")
                        
                # embedding generate
                _tokens = d["input_ids"].to(gpu_idx)
                assert _tokens.dim() == 2, "The input tensor should have a dimension of 2"
                ffn_handled_tokens.append(_tokens.size(0) * _tokens.size(1)) # batch_size * seq_len
                attention_mask = d["attention_mask"].to(gpu_idx)
                torch.cuda.synchronize()
                start_event.record()
                if args.decode == -1:
                    _ = model(_tokens)
                else:
                    # perform decoding
                    decoding_step = args.decode
                    raise NotImplementedError("Not implemented decoding yet")
                end_event.record()
                torch.cuda.synchronize()
                
                # save and iterate
                ffn_elapsed_times.append(start_event.elapsed_time(end_event) * 0.001) # ms -> s
                
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
                
                ffn_elapsed_times.append(start_event.elapsed_time(end_event) * 0.001) # ms -> s
                
    # Calculate the throughput
    ffn_throughput = []
    for i in range(len(ffn_elapsed_times)):
        ffn_throughput.append(ffn_handled_tokens[i] / ffn_elapsed_times[i])
    
    # Log the results if specified
    if args.log_results:
        os.makedirs("results", exist_ok=True)
        curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_name = f"results/pmoe_schemoe_{curr_datetime}.json"
        # Make a new dictionary to store the results
        final_list = []
        n_items = len(ffn_elapsed_times)
        for i in range(n_items):
            result_dict = {}
            item_list = [ffn_elapsed_times[i], ffn_handled_tokens[i], ffn_throughput[i]] # tokens per second
            result_dict[f"item_{i}"] = item_list
            final_list.append(result_dict)
        final_list.append({"batch_size": args.batch_size, "pipeline_stage": args.schemoe_overlap_degree, "avg_tp": statistics.mean(ffn_throughput), "std_tp": statistics.stdev(ffn_throughput)})
            
        with open(result_name, "w") as f:
            json.dump(final_list, f, indent=4)
            
    # Calculate the average time taken for the forward pass
    average_elapsed_time = sum(ffn_elapsed_times) / len(ffn_elapsed_times)
    log(f"Average time [pMOE] with {args.iterations}th iterations: {average_elapsed_time} s")
    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()