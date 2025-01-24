# from megatron.training import get_args
import torch.nn.functional as F
from schemoe.moe import moe_layer, pmoe_layer
import torch.distributed as dist

import torch, argparse, os, random
import gc

import nvtx

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss

    @property
    def has_loss(self):
        return self.loss is not None
    
class MimicGate(BaseGate):
    r"""
    Get the probability distribution data from the csv file,
    and mimic the distribution of the expert selection.
    """
    
    def __init__(self, d_model, num_expert, world_size, top_k=1,  gpu_idx=-1, path="/home/wjbang/workspace/pMoE/pMoE/p_count_selected.csv"):
        super().__init__(num_expert, world_size)
        assert top_k == 1, "Pshave gate only supports top_k = 1"
        
        self.num_expert = self.tot_expert
        self.world_size = world_size
        self.d_model = d_model
        self.gpu_idx = gpu_idx
        
        self.loaded_distribution = self.load_distribution(path)
        self.loaded_distribution = self.loaded_distribution.to(torch.float32).to(self.gpu_idx)
        
    def load_distribution(self, path):
        with open(path, "r", newline="") as f:
            tmp = []
            reader = csv.reader(f)
            data_all = list(reader)
            for data in data_all:
                tmp.append(list(map(float, data)))
            distribution = torch.tensor(tmp)
            
        return distribution
    
    def forward(self, x):
        r"""
        The forward function of the pshave gate.
        """
        n_tokens = x.size(0)
        n_samples = self.loaded_distribution.size(0)
        
        # Create dummy gate_top_k_val
        gate_top_k_val = torch.zeros(n_tokens, 1, dtype=torch.bfloat16).to(x.device)
        
        # Randomly select an index from the loaded distribution
        idx = torch.randint(n_samples, (1,)).to(x.device)
        selected_prob = self.loaded_distribution[idx].squeeze(0)
        
        # Set the token_board for each expert
        expected_tokens = n_tokens * selected_prob
        token_board = torch.floor(expected_tokens).to(torch.long).to(x.device)
        remainder = n_tokens - token_board.sum().item()
        
        # Find the bin with the maximum allocation
        max_idx = torch.argmax(token_board)
        token_board[max_idx] += remainder
        return token_board, gate_top_k_val
    
def split_gate(tensor, n):
    if n <= 0:
        raise ValueError("Number of sub-tensors (n) must be greater than 0.")
    
    # list saving subtensors
    sub_tensors = [torch.zeros_like(tensor) for _ in range(n)]
    
    # 텐서의 모든 위치를 순회하며 값을 랜덤하게 분할
    total_elements = tensor.numel()  # 전체 요소 수
    for linear_idx in range(total_elements):
        # torch.unravel_index는 torch.Tensor를 필요로 하므로 linear_idx를 텐서로 변환
        multi_index = torch.unravel_index(torch.tensor(linear_idx), tensor.shape)
        
        # 현재 위치의 값
        remaining_value = tensor[multi_index].item()
        
        for j in range(n - 1):
            # 각 서브 텐서의 값을 랜덤하게 설정
            sub_value = torch.randint(0, remaining_value + 1, (1,)).item()
            sub_tensors[j][multi_index] = sub_value
            remaining_value -= sub_value
        
        # 마지막 서브 텐서에 남은 값을 할당
        sub_tensors[-1][multi_index] = remaining_value
    
    return sub_tensors

class HardCodedGate:
    def __init__(self, gate):
        self.gate = gate # 2D Tensor

    def src(self, x): # x -> destinations
        return self.gate[x][:]
    
    def dst(self, x): # sources -> x
        return self.gate[:][x]
    
    def all(self): # all
        return self.gate.sum(dim=1)
    
    def input(self, rank): # all
        return int(self.gate.sum(dim=1)[rank].item())
    
    def output(self, rank):
        return self.gate.sum(dim=0)[rank].item()
    
    def batch(self): 
        return self.gate.sum().item()
    
def generate_random_tensor(n):
    """
    Generate an n x n tensor where each element is 2^k (k > 1).
    
    Args:
        n (int): Size of the tensor (n x n).
    
    Returns:
        torch.Tensor: Random n x n tensor with values as powers of 2.
    """
    # Define the range for k (k > 1)
    k_min = 2  # Minimum value for k
    k_max = 6  # Maximum value for k (you can adjust this)
    
    # Generate random powers of 2 for each element
    values = [[2 ** random.randint(k_min, k_max) for _ in range(n)] for _ in range(n)]
    
    # Convert to a PyTorch tensor
    tensor = torch.tensor(values, dtype=torch.long, device='cuda')
    
    return tensor


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
# parser.add_argument("--schemoe_comm_name", type=str, default="pipe")





args = parser.parse_args()

dist.init_process_group("nccl")

dist_rank, dist_world_size = dist.get_rank(), dist.get_world_size()

args.local_rank = os.environ.get("LOCAL_RANK", 0)

def log(msg):
    if dist_rank == 0:
        print(f"{msg}\n")
        
device = torch.device("cuda:%s" % args.local_rank)
# print(f"{device}")
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

# hard_gate = generate_random_tensor(dist_world_size)
hard_gate = MimicGate(args.model_dim, 1, dist_world_size, top_k=1, gpu_idx=device, path=args.gate_path)
_hard_gate = torch.tensor([[128, 128], [128, 128]],  dtype=torch.long)
# ours -> 128 
# imb -> 64
# ours -> 96
# imb -> 64
# ours -> 80
# imb -> 64
# hard_gate
# ours -> 112 1/N 
# imb -> 128 remote max(64, 128) -> 128 
# _2 = torch.tensor([[32, 32], [16, 64]],  dtype=torch.long)
# hard_gate = [hard_gate_1, hard_gate_2]

# print(f"Hard gate: {hard_gate}")

from schemoe.impls import communicate as C
def schmoe_moe(args, idx):
    hidden_size = args.hidden_size
    ffn_hidden_size = args.hidden_size
    world_size = dist.get_world_size()
    num_experts = args.num_experts
    if args.moe_expert_capacity_factor is not None and args.moe_expert_capacity_factor > 0:
        capacity_factor = args.moe_expert_capacity_factor
    else:
        capacity_factor = 0.0
    # log(f"ScheMoE capacity factor: {capacity_factor}")
    expert_per_node = num_experts // world_size
    top_k = args.moe_router_topk
    activation = F.gelu
    compress_name = args.schemoe_compress_name
    comm_name = args.schemoe_comm_name
    overlap_degree = args.schemoe_overlap_degree
    moe_ffn = moe_layer(
        gate_type={
            'type' : 'top', 'k' : top_k, 'capacity_factor': capacity_factor,
            'fp32_gate': True, 'gate_noise': 1.0
        },
        model_dim=args.model_dim,
        hardcode_gate=hard_gate,
        experts={
            'count_per_node': expert_per_node,'type': 'ffn', 
            'hidden_size_per_expert': ffn_hidden_size,
            'activation_fn' : lambda x: activation(x)
        },
        a2a_ffn_overlap_degree = overlap_degree,
        compress_name = compress_name,
        comm_name = comm_name,
        scan_expert_func = lambda name, param: setattr(param, 'allreduce', False),
    )
    return moe_ffn

def balance_moe(args, idx):
    hidden_size = args.hidden_size
    ffn_hidden_size = args.hidden_size
    world_size = dist.get_world_size()
    num_experts = args.num_experts
    if args.moe_expert_capacity_factor is not None and args.moe_expert_capacity_factor > 0:
        capacity_factor = args.moe_expert_capacity_factor
    else:
        capacity_factor = 0.0
    log(f"ScheMoE capacity factor: {capacity_factor}")
    expert_per_node = num_experts // world_size
    top_k = args.moe_router_topk
    activation = F.gelu
    compress_name = args.schemoe_compress_name
    comm_name = args.schemoe_comm_name
    overlap_degree = args.schemoe_overlap_degree
    pmoe_ffn = pmoe_layer(
        gate_type={
            'type' : 'top', 'k' : top_k, 'capacity_factor': capacity_factor,
            'fp32_gate': True, 'gate_noise': 1.0
        },
        model_dim=args.model_dim,
        hardcode_gate=hard_gate,
        experts={
            'count_per_node': expert_per_node,'type': 'ffn', 
            'hidden_size_per_expert': ffn_hidden_size,
            'activation_fn' : lambda x: activation(x)
        },
        a2a_ffn_overlap_degree = overlap_degree,
        compress_name = compress_name,
        comm_name = comm_name,
        scan_expert_func = lambda name, param: setattr(param, 'allreduce', False),
    )
    return pmoe_ffn

if __name__ == "__main__":
    #self, d_model, num_expert, world_size, top_k=1,  gpu_idx=-1, path="/home/wjbang/workspace/pMoE/pMoE/p_count_selected.csv"
    pmoe_layer=balance_moe(args, 0)
    moe_layer=schmoe_moe(args, 0)
    
    
    # g = HardCodedGate(hard_gate)
    # input = torch.randn(g.input(dist_rank), args.model_dim)
    # global_gate = hard_gate[0] + hard_gate[1]
    g = HardCodedGate(_hard_gate)
    input = torch.randn(g.input(dist_rank), args.model_dim, device=device)
    # print(hard_gate.data_ptr())
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    pstart_event = torch.cuda.Event(enable_timing=True)
    pend_event = torch.cuda.Event(enable_timing=True)
    
    # Warm up
    for i in range(10):
        with torch.no_grad():
            _ = moe_layer(input)
            _ = pmoe_layer(input)
    # pend_event.record()
    
    # print(f"output: {z}\n")
    torch.cuda.synchronize()  
    # _elapsed_time_moe = pstart_event.elapsed_time(pend_event) 
    # log(f"pmoe layer execution time: {_elapsed_time_moe:.3f} ms\n")
    
    pstart_event.record()
    with nvtx.annotate("ScheMoE", color="green"):
        for i in range(1):
            with torch.no_grad():
                _ = moe_layer(input)
    pend_event.record()
    torch.cuda.synchronize()  
    _elapsed_time_moe = pstart_event.elapsed_time(pend_event) 
    log(f"moe layer execution time: {_elapsed_time_moe:.3f} ms\n")
    
    torch.cuda.synchronize() 
    
    start_event.record()
    with nvtx.annotate("pMoE", color="green"):
        for i in range(1):
            with torch.no_grad():
                _ = pmoe_layer(input)
    end_event.record()
    torch.cuda.synchronize() 
    elapsed_time_moe = start_event.elapsed_time(end_event) 
    log(f"pmoe layer execution time: {elapsed_time_moe:.3f} ms\n")
    
    
    dist.destroy_process_group()