import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import torch

import argparse, os, random
import gc
import nvtx
import csv

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
    
    args = parser.parse_args()
    
    return args

def main():
    args = custom_argparser()
    dist.init_process_group("nccl")

    dist_rank, dist_world_size = dist.get_rank(), dist.get_world_size()
    args.local_rank = os.environ.get("LOCAL_RANK", 0)

    def log(msg):
        if dist_rank == 0:
            print(f"{msg}\n")
            
    device = torch.device("cuda:%s" % args.local_rank)
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
        
    

if __name__ == "__main__":
    main()