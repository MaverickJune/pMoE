# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import schemoe_custom_kernel
from time import time
from torch.distributed import get_rank
import torch.distributed as dist

from ..impls import communicate as C

def get_world_size(group=None):
    try:
        return dist.get_world_size(group)
    except:
        return 1


def get_world_rank(group=None):
    try:
        return dist.get_rank(group)
    except:
        return 0

# def log(msg):
#     if dist_rank == 0:
#         print(f"{msg}\n")
        
def split_gate(tensor, n):
    if n <= 0:
        raise ValueError("Number of sub-tensors (n) must be greater than 0.")
    
    # list saving subtensors
    sub_tensors = [torch.zeros_like(tensor) for _ in range(n)]
    
    sub_tensors[0] = tensor // n + tensor % n
    sub_tensors[0] = sub_tensors[0].contiguous()
    
    for i in range(n):
        if i == 0:
            continue
        sub_tensors[i] = tensor // n
        sub_tensors[i] = sub_tensors[i].contiguous()
    # # 텐서의 모든 위치를 순회하며 값을 랜덤하게 분할
    # total_elements = tensor.numel()  # 전체 요소 수
    # for linear_idx in range(total_elements):
    #     # torch.unravel_index는 torch.Tensor를 필요로 하므로 linear_idx를 텐서로 변환
    #     multi_index = torch.unravel_index(torch.tensor(linear_idx), tensor.shape)
        
    #     # 현재 위치의 값
    #     remaining_value = tensor[multi_index].item()
        
    #     for j in range(n - 1):
    #         # 각 서브 텐서의 값을 랜덤하게 설정
    #         sub_value = torch.randint(0, remaining_value + 1, (1,)).item()
    #         sub_tensors[j][multi_index] = sub_value
    #         remaining_value -= sub_value
        
    #     # 마지막 서브 텐서에 남은 값을 할당
    #     sub_tensors[-1][multi_index] = remaining_value
    
    return sub_tensors

# global_gate =     ???
# gates = pipe_gate(global_gate)
# gate = [HardcodedGate(gate) for gate in gates]


class Buffer_imbalance:
    def __init__(self, group, gate, model_dim, hidden_dim):
        self.gate = gate
        self.size = get_world_size(group)
        self.rank = get_world_rank(group)
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.input = torch.randn((gate.input(self.rank), model_dim), device='cuda')
        self.buffer_1 = torch.zeros((gate.output(self.rank), model_dim), device='cuda')
        self.buffer_2 = torch.zeros((gate.input(self.rank), model_dim), device='cuda')

        self._dst = self.gate.dst(self.rank).contiguous()
        self._src = self.gate.src(self.rank).contiguous()

    def dst(self):
        return self._dst
    
    def src(self):
        return self._src
        
class Buffer_balance:
    def __init__(self, group, gate, model_dim, hidden_dim):
        self.gate = gate
        self.size = get_world_size(group)
        self.rank = get_world_rank(group)
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        # print(type(self.size), type(model_dim), type(gate.input(self.rank)))
        self.input = torch.randn((gate.input(self.rank), model_dim), device='cuda')
        self.buffer_1 = torch.zeros((gate.batch(), int(model_dim // self.size)), device='cuda')
        self.buffer_2 = torch.zeros((gate.batch(), hidden_dim), device='cuda')
        self.buffer_3 = torch.zeros((int(gate.input(self.rank) * self.size), model_dim), device='cuda')
        
        self._src = self.gate.src(self.rank).contiguous()
        self._dst = self.gate.dst(self.rank).contiguous()
        self._global_dst = self.gate.all().contiguous()
        # print(f"RANK: {self.rank}\n")
        self._reshape_dst = self.gate.reshape_dst(self.rank).contiguous()
        
    def buffer(self):
        return self.buffer_1, self.buffer_2, self.buffer_3
    
    def dst(self):
        # self._dst = self.gate.dst(self.rank).contiguous()
        return self._dst
    
    def src(self):
        # self._src = self.gate.src(self.rank).contiguous()
        return self._src
    
    def global_dst(self):
        # self._global_dst = self.gate.all().contiguous()
        return self._global_dst
    
    def reshape_dst(self):
        return self._reshape_dst
    
    def reduce(self, buff):
        if buff.shape[1] == self.hidden_dim:
            reshaped_tensor = buff.view(self.gate.batch(), self.size, self.hidden_dim//self.size)
            return reshaped_tensor.mean(dim=1).contiguous()
        elif buff.shape[1] == self.model_dim:
            reshaped_tensor = buff.view(self._global_dst[self.rank].item(), self.size, self.model_dim)
            return reshaped_tensor.mean(dim=1).contiguous()
        else: 
            ValueError(f"Invalid Shape. Please check the shape of the tensor shape: {buff.shape}") 
            return
        
class HardCodedGate:
    def __init__(self, gate):
        self.gate = gate # 2D Tensor in CPU
        if self.gate.device.type != 'cpu':
            print(f"Error: Buffer is not on CPU. Current device: {self.gate.device}")
        
    def src(self, rank): # x -> destinations
        return self.gate[rank,:]
    
    def dst(self, rank): # sources -> x
        return self.gate[:,rank]
    
    def all(self): # all
        return self.gate.sum(dim=1)
    
    def input(self, rank): # all
        return int(self.gate.sum(dim=1)[rank].item())
    
    def output(self, rank):
        return int(self.gate.sum(dim=0)[rank].item())
    
    def batch(self): 
        return int(self.gate.sum().item())
    
    def reshape_dst(self, i):
        return torch.full((self.gate.shape[0],), self.gate.sum(dim=1)[i])
    
class Compress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output, compress_name, comm_name, src, dst):
        input = schemoe_custom_kernel.compress_operation(input, output, compress_name, comm_name, src, dst)
        return input

    @staticmethod
    def backward(ctx, grad):
        grad = schemoe_custom_kernel.decompress_operation(grad)
        return grad, None, None


class Decompress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, compress_name, comm_name):
        ctx.compress_name = compress_name
        ctx.comm_name = comm_name
        input = schemoe_custom_kernel.decompress_operation(input)
        return input

    @staticmethod
    def backward(ctx, grad):
        return schemoe_custom_kernel.compress_operation(grad, ctx.compress_name, ctx.comm_name), None, None


class Comm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, version=0):
        return schemoe_custom_kernel.comm_operation(input, version)

    @staticmethod
    def backward(ctx, grad, version=0):
        return schemoe_custom_kernel.comm_operation(grad, version)


def a2a_ffn_overlap_balance(_input, gate, model_dim, hidden_dim, expert_fn, a2a_ffn_overlap_degree, use_2dh, group, compress_name, comm_name):
    split_dim = 1
    
    assert a2a_ffn_overlap_degree <= C.AllToAllStatus.max_num_split, "Excepting a2a_ffn_overlap_degree (%d) <= AllToAllStatus.max_num_split (%d)." % (
        a2a_ffn_overlap_degree, C.AllToAllStatus.max_num_split)
    
    C.AllToAllStatus.init(group, a2a_ffn_overlap_degree, split_dim)
    
    
    # Gate Dim: (1 x N)
    idx, _ = gate(_input) # 1 x 8
    size = get_world_size(group)
    assert idx.sum() == _input.shape[0]
    idx = idx.view(size, -1).sum(dim=1).contiguous()
    gidx = torch.zeros((size, size), dtype=torch.long, device='cuda')
    dist.all_gather(list(gidx.chunk(size, dim=0)), idx)
    # CPU 에 올리기
    gidx_cpu = gidx.cpu()
    # inserted
    # print(f"sum: {idx.sum}\n, {idx}")
    # print(gidx_cpu)
    
    # idx = gate(input) -> (B x 1)
    # selection = idx.count(E) -> (1 x N)
    # selection.sum() == B (Check)
    # gselect = dist.all2all(selection) -> (N x N)
    # gates = split_gate(gate, a2a_ffn_overlap_degree)
    
    # buffer = [Buffer_balance(group, HardCodedGate(_gate), model_dim, hidden_dim) for _gate in gates]
    
    # Gate를 GPU로 집어넣고 cpp 에서 수정해야함.
    
    gates = split_gate(gidx_cpu, a2a_ffn_overlap_degree)
    
    # gates = split_gate(gate, a2a_ffn_overlap_degree)
    
    
    # buffer initialization
    buffer = [Buffer_balance(group, HardCodedGate(_gate), model_dim, hidden_dim) for _gate in gates]
    
    input = [None] * a2a_ffn_overlap_degree
    schemoe_custom_kernel.clear_ptr_lst()

    for i in range(a2a_ffn_overlap_degree):
        # print(f"Pipeline Stage: {i}\n")
        input[i] = buffer[i].input #.clone().contiguous()
        # print(f"input data ptr: {hex(input[i].data_ptr())}, device: {input[i].device} \n")
        input[i] = Compress.apply(input[i], buffer[i].buffer_1, compress_name, comm_name, buffer[i].src(), buffer[i].global_dst())
        input[i] = Comm.apply(input[i], 1)
        # print(f"Debug 1: {i}\n")
        # print(f"RANK: {buffer[i].rank} buffer src: {hex(buffer[i].src().data_ptr())}, buffer dst: {hex(buffer[i].global_dst().data_ptr())}\n")
    for i in range(a2a_ffn_overlap_degree):
        input[i] = Decompress.apply(input[i], compress_name, comm_name)
        input[i] = expert_fn(input[i], 0)        
        # print(f"RANK: {buffer[i].rank} buffer src: {hex(buffer[i].src().data_ptr())}, buffer dst: {hex(buffer[i].dst().data_ptr())}\n")
        input[i] = Compress.apply(input[i], buffer[i].buffer_2, compress_name, comm_name, buffer[i].src(), buffer[i].dst())
        input[i] = Comm.apply(input[i], 2)
        # print(f"Debug 2: {i} \n")
    for i in range(a2a_ffn_overlap_degree):
        input[i] = Decompress.apply(input[i], compress_name, comm_name)
        input[i] = buffer[i].reduce(input[i])
        input[i] = expert_fn(input[i], 1)
        # print(f"RANK: {buffer[i].rank} Debug reduced input: {input[i].shape}, global_dst(): {hex(buffer[i].global_dst().data_ptr())} reshape_dst(): {hex(buffer[i].reshape_dst().data_ptr())}\n")
        input[i] = Compress.apply(input[i], buffer[i].buffer_3, compress_name, comm_name, buffer[i].global_dst(), buffer[i].reshape_dst())
        # print(f"Debug 3-1: {i}\n")
        input[i] = Comm.apply(input[i], 3)
        # print(f"\n [{buffer[i].rank}] Debug 3-2: {i}\n")
    for i in range(a2a_ffn_overlap_degree):
        input[i] = Decompress.apply(input[i], compress_name, comm_name)
        # print(f"Debug 4-1: {i}\n")
        input[i] = buffer[i].reduce(input[i])
        # print(f"Debug 4-2: {i}\n")
    output = [input[i] for i in range(a2a_ffn_overlap_degree)]
    output = torch.cat(output, dim=0).contiguous()
    # print(f"Debug 5: output shape: {output.shape}\n")
    # for i in range(a2a_ffn_overlap_degree):
    #     _ = Decompress.apply(buffer[i].input, compress_name, comm_name)
    #     _ = Decompress.apply(buffer[i].input, compress_name, comm_name)
    #     _ = Decompress.apply(buffer[i].input, compress_name, comm_name)
        
    return output

def a2a_ffn_overlap_forward(_input, gate, model_dim, hidden_dim, expert_fn, a2a_ffn_overlap_degree, use_2dh, group, compress_name, comm_name):
    split_dim = 1
    
    assert a2a_ffn_overlap_degree <= C.AllToAllStatus.max_num_split, "Excepting a2a_ffn_overlap_degree (%d) <= AllToAllStatus.max_num_split (%d)." % (
        a2a_ffn_overlap_degree, C.AllToAllStatus.max_num_split)
    C.AllToAllStatus.init(group, a2a_ffn_overlap_degree, split_dim)
    
    # Gate Dim: (1 x N)
    idx, _ = gate(_input) # 1 x 8
    size = get_world_size(group)
    assert idx.sum() == _input.shape[0]
    idx = idx.view(size, -1).sum(dim=1).contiguous()
    gidx = torch.zeros((size, size), dtype=torch.long, device='cuda')
    dist.all_gather(list(gidx.chunk(size, dim=0)), idx)
    # CPU 에 올리기
    gidx_cpu = gidx.cpu()
    # inserted
    # print(f"sum: {idx.sum}\n, {idx}")
    # print(gidx_cpu)
    
    gates = split_gate(gidx_cpu, a2a_ffn_overlap_degree)
    
    # gates = split_gate(gate, a2a_ffn_overlap_degree)
    buffer = [Buffer_imbalance(group, HardCodedGate(g.contiguous()), model_dim, hidden_dim) for g in gates]
    
    input = [None] * a2a_ffn_overlap_degree
    schemoe_custom_kernel.clear_ptr_lst()

    
    for i in range(a2a_ffn_overlap_degree):
        input[i] = buffer[i].input # .clone().contiguous()
        input[i] = Compress.apply(input[i], buffer[i].buffer_1, compress_name, comm_name, buffer[i].src(), buffer[i].dst())
        input[i] = Comm.apply(input[i], 0)
        
    for i in range(a2a_ffn_overlap_degree):
        input[i] = Decompress.apply(input[i], compress_name, comm_name)
        input[i] = expert_fn(input[i])
        input[i] = Compress.apply(input[i], buffer[i].buffer_2, compress_name, comm_name, buffer[i].dst(), buffer[i].src())
        input[i] = Comm.apply(input[i], 0)
        
    for i in range(a2a_ffn_overlap_degree):
        input[i] = Decompress.apply(input[i], compress_name, comm_name)
    
    output = [input[i] for i in range(a2a_ffn_overlap_degree)]
    output = torch.cat(output, dim=0).contiguous()
    return output
   