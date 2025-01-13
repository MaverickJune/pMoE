r"""
FMoE core layer
"""
import tree
import os
import torch
import torch.nn as nn

from .functions import prepare_forward, ensure_comm, prepare_balance_forward
from .functions import MOEScatter, MOEGather, MoEReshape
from .functions import AllGather, Slice
from .gates import NaiveGate

from .fastermoe.config import switch_from_env

import torch.distributed as dist
import nvtx
# from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage
from concurrent.futures import ThreadPoolExecutor
import time

# Function to be executed in parallel
# def process_task(i, comp_stream, comm_stream, done_event, _x, expert_fn, fwd_expert_count):
#     with torch.cuda.stream(comp_stream[i]):
#         done_event[i].wait(comm_stream[i])  # Wait for communication stream
#         _x[i] = expert_fn(_x[i], fwd_expert_count[i])  # Perform computation

# def thread_setup(
def log(message):
    # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(f"{message}\n")
    
def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size, **kwargs):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    def scatter_func(tensor):
        return MOEScatter.apply(
            tensor,
            torch.div(pos, topk, rounding_mode='floor'),
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
        )
    
    # TODO: benchmark its speed
    with nvtx.annotate("FMOE Comm 1", color="red"):
        x = tree.map_structure(scatter_func, inp)
    with nvtx.annotate("FMoE Comp 1", color="red"):
        x = expert_fn(x, fwd_expert_count)

    out_batch_size = tree.flatten(inp)[0].shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    def gather_func(tensor):
        return MOEGather.apply(
            tensor,
            pos,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            world_size,
        )
        
    # TODO: benchmark its speed
    with nvtx.annotate("FMoE Comm 2", color="red"):
        outp = tree.map_structure(gather_func, x)
    return outp

# # TO DO: reshape the tensor.
# def _local_reshape(inp, local_expert_count, fwd_expert_count, pos):
#     lec_cum = torch.cumsum(local_expert_count, dim=0).int()
#     reshape_idx = torch.empty((pos.size(0),), device=inp.device, dtype=torch.long)
#     for i in reshape_idx:
        
#     # inp_buf = torch.index_select(inp, 0, pos)
     
#     return inp

def reduce_scatter_column(tensor, dim, ctx):
    """
    Reduce-scatter the tensor from shape [b x k x n, H] to [b x k, aH/n].

    Args:
        tensor (torch.Tensor): Input tensor of shape [b x k x n, aH].
        _shape (tuple): size of output (b x k, H).
        rank (int): Rank of the current process.

    Returns:
        torch.Tensor: Reduced-scattered tensor of shape [b x k, H] for this process.
    """
    # Ensure distributed process group is initialized
    assert dist.is_initialized()

    group = ctx.get_group("tp")
    world_size = ctx.get_size("tp")
    
    # Validate input tensor shape
    bkn, H = tensor.shape[0], tensor.shape[1]
    # print(f"shape of tensor is {tensor.shape}")
    assert H % world_size == 0, "Tensor must be divisible by world_size."
    
    # Compute chunk size
    chunk_size = H // world_size  # Each chunk corresponds to 4h

    # Split input tensor into chunks for reduce-scatter
    input_chunks = list(tensor.chunk(world_size, dim=dim))

    # Create output tensor for this rank
    output_tensor = torch.empty((bkn, chunk_size), dtype=tensor.dtype, device=tensor.device)
    # with nvtx.annotate("FFN sub comm", color="green"):
    # Perform reduce-scatter
    # print(f"shape of output tensor is {output_tensor.shape}")
    work = dist.reduce_scatter(output_tensor, input_chunks, op=dist.ReduceOp.SUM, group=group, async_op=True)

    return output_tensor,work

def _pmoe_general_global_forward(inp, gate, expert_fn, experts, total_experts, layer_num, pipeline, ctx, stream, idx, **kwargs):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    
    group: communication groups
    
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    # if pipeline > 1:
    #     # Initialize executors at a higher scope
    #     executor = [ThreadPoolExecutor(max_workers=1) for _ in range(pipeline)]
    #     futures_stage_gate = []
    #     futures_stage_reshape = []
    #     futures_stage_subcomp1 = []
    #     futures_stage_subcomm = []
    #     futures_stage_subcomp2 = []
    #     future_stage_allreduce = []
    #     def reshape_func(tensor, i):
    #         return MoEReshape.apply(
    #             tensor,
    #             torch.div(pos[i], topk, rounding_mode='floor'),
    #             local_expert_count[i],
    #             global_expert_count[i],
    #             fwd_batch_size[i],
    #             ctx,
    #             i
    #         )
    #     local_expert_count = [None]*pipeline
    #     global_expert_count = [None]*pipeline
    #     fwd_expert_count = [None]*pipeline
    #     fwd_batch_size = [None]*pipeline
    #     pos = [None]*pipeline
    #     _gate = torch.chunk(gate, pipeline, dim=0)
    #     chunk = torch.chunk(inp, pipeline, dim=0) # [None for _ in range(pipeline)]
    #     x = [None]*pipeline
    #     g = [None]*pipeline
    #     g[0] = _gate[0].clone()  # new memory buffer
    #     g[1] = _gate[1].clone()
    #     x[0] = chunk[0].clone()  # new memory buffer
    #     x[1] = chunk[1].clone()  # new memory buffer
    #     _x = [None]*pipeline
    #     work = [None]*pipeline
        
    #     print(f"[RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}]")
        
    #     # for i in range(pipeline):
    #     #     with torch.cuda.stream(stream[i]):
    #     #         log(f"[START] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} in prepare_balance_forward")
    #     #         (
    #     #             pos[i],
    #     #             local_expert_count[i], # [E]
    #     #             global_expert_count[i], # [E x n]
    #     #             fwd_expert_count[i], # [E], idx: expert idx, value: number of tokens
    #     #             fwd_batch_size[i], # b x k x n
    #     #         ) = prepare_balance_forward(g[i], total_experts, ctx, i) 
    #     #         log(f"[END] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} submitted prepare_balance_forward")
        
    #     # for i in range(pipeline):
    #     #     with torch.cuda.stream(stream[i]):
    #     #         log(f"[START] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} in reshape")
    #     #         _x[i] = tree.map_structure(reshape_func, x[i], i) 
    #     #         log(f"[END] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} submitted reshape_func")
        
    #     # for i in range(pipeline):
    #     #     with torch.cuda.stream(stream[i]):
    #     #         log(f"[START] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} sub comp 1 {_x[i].shape}")
    #     #         _x[i] = tree.map_structure(experts.htoh4,  _x[i], fwd_expert_count[i])
    #     #         log(f"[END] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} submitted experts.htoh4")
        
    #     # for i in range(pipeline):
    #     #     with torch.cuda.stream(stream[i]):
    #     #         log(f"[START] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} sub comm 1 {_x[i].shape}")
    #     #         _x[i], work[i] = tree.map_structure(lambda t: reduce_scatter_column(t, 1, ctx), _x[i])
        
    #     # for i in range(pipeline):
    #     #         log(f"[START] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} sub comp 2\n")
    #     #         _x[i] = experts.activation(_x[i])
    #     #         _x[i] = experts.h4toh(_x[i], fwd_expert_count[i])
                
    #     # torch.cuda.synchronize()
    #     for i in range(pipeline):
    #         with torch.cuda.stream(stream[i]):
    #             stream[i].synchronize()
    #             log(f"[START] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} in prepare_balance_forward")
    #             fut = executor[i].submit(prepare_balance_forward, _gate[i], total_experts, ctx, i)
    #             futures_stage_gate.append(fut)
    #             log(f"[END] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} submitted prepare_balance_forward")
                
    #     for i in range(pipeline):
    #         try:
    #             with torch.cuda.stream(stream[i]):
    #                 # Wait for prepare_balance_forward[i] to complete and get the result
    #                 pos[i], local_expert_count[i], global_expert_count[i], fwd_expert_count[i], fwd_batch_size[i] = futures_stage_gate[i].result()
    #                 log(f"[START] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} in reshape")
    #                 fut = executor[i].submit(reshape_func, x[i], i)
    #                 futures_stage_reshape.append(fut)
    #                 log(f"[END] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} submitted reshape_func")
    #         except Exception as e:
    #             log(f"Error in pipeline stage {i} in prepare_balance_forward {e}\n ")
        
    #     for i in range(pipeline):
    #         try:
    #             with torch.cuda.stream(stream[i]):
    #                 _x[i] = futures_stage_reshape[i].result()
    #                 log(f"[START] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} sub comp 1 {_x[i].shape}")
    #                 fut = executor[i].submit(experts.htoh4, _x[i], fwd_expert_count[i])
    #                 futures_stage_subcomp1.append(fut)
    #                 log(f"[END] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} submitted experts.htoh4")
    #         except Exception as e:
    #             print(f"Error in pipeline stage {i} in reshape {e}")
        
    #     for i in range(pipeline):
    #         try:
    #             with torch.cuda.stream(stream[i]): 
    #                 _x[i] = futures_stage_subcomp1[i].result()
    #                 log(f"[RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} sub comm 1 {_x[i].shape}")
    #                 fut = executor[i].submit(reduce_scatter_column, _x[i], 1, ctx)
    #                 futures_stage_subcomm.append(fut)
    #         except Exception as e:
    #             print(f"Error in pipeline stage {i} in subcomp1 {e}")
        
    #     for i in range(pipeline):
    #         try:
    #             _x[i], work = futures_stage_subcomm[i].result()
    #             work.wait()
    #             print(f"[RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{i} sub comp 2\n")
    #             # with torch.cuda.stream(stream[i]):
    #             fut = executor[i].submit(experts.h4toh, experts.activation(_x[i]), fwd_expert_count)
    #             futures_stage_subcomp2.append(fut)
    #         except Exception as e:
    #             print(f"Error in pipeline stage {i} in subcomm {e}")
                
    #     for i in range(pipeline):
    #         try:
    #             _x[i] = futures_stage_subcomp2[i].result()
    #             # with torch.cuda.stream(stream[i]):
    #             reshaped_tensor = _x[i].view(_x[i].shape[0]//_gate[i].shape[1], _gate[i].shape[1], _x[i].shape[1])
    #             _x[i] = reshaped_tensor.sum(dim=1) # [B, H]
    #             work = dist.all_reduce(_x[i], op=dist.ReduceOp.SUM, group=ctx.get_group("tp"), async_op=True)
    #             future_stage_allreduce.append(fut)
    #         except Exception as e:
    #             print(f"Error in pipeline stage {i} in subcomp2 {e}")
        
    #     for i in range(pipeline):
    #         try:
    #             work[i] = future_stage_allreduce[i].result()
    #             work[i].wait()
    #         except Exception as e:
    #             print(f"Error in pipeline stage {i} in allreduce {e}")
        
    #     out_batch_size = tree.flatten(inp)[0].shape[0]
    #     outp = torch.cat([_x[0], _x[1]], dim=0)
    #     outp = outp[:out_batch_size]
        
    #     # Shutdown
    #     for e in executor:
    #         e.shutdown(wait=True)
    
    #     return outp
        
    # else:
        #with torch.cuda.stream(stream[0]):
        with torch.cuda.stream(stream[idx]):
            (
                pos,
                local_expert_count, # [E]
                global_expert_count, # [E x n]
                fwd_expert_count, # [E], idx: expert idx, value: number of tokens
                fwd_batch_size, # b x k x n
            ) = prepare_balance_forward(gate, total_experts, ctx, idx, stream) # need only expert count
            
            # print(f"[RANK {ctx.get_rank('tp')} gate size and value: {gate.shape} {gate} local expert count: {local_expert_count.sum()} , global expert count: {global_expert_count.sum()}, fwd expert count: {fwd_expert_count.sum()}")
            def reshape_func(tensor, idx):
                return MoEReshape.apply(
                    tensor,
                    torch.div(pos, topk, rounding_mode='floor'),
                    local_expert_count,
                    global_expert_count,
                    fwd_batch_size,
                    ctx,
                    idx
                )
            # 1. all to all 1
            log(f"[START] [RANK {ctx.get_rank('tp')} / {ctx.get_size('tp')}] T{idx} after prepare_balance_forward, pos: {pos.shape}, local_expert_count: {local_expert_count.shape}, global_expert_count: {global_expert_count.shape}, fwd_expert_count: {fwd_expert_count}, fwd_batch_size: {fwd_batch_size}")
            x = tree.map_structure(reshape_func, inp, idx)   
            # if ctx.get_rank('tp') == 0:
            # print(f"[RANK {ctx.get_rank('tp')} L{layer_num} T{idx}] shape of original: {inp.shape}, input tensor is {x.shape} \n")
            # 2. sub comp + comm    
            # print(f"[RANK {ctx.get_rank('tp')} L{layer_num} T{idx}] After A2A \n")
            x = expert_fn(x, fwd_expert_count) # [B x k, H]
            # print(f"[RANK {ctx.get_rank('tp')} L{layer_num} T{idx}] After 2 sub and 1 comm \n")
            
            # 3. all to all 2
            if len(gate.shape) == 2:
                reshaped_tensor = x.view(x.shape[0]//gate.shape[1], gate.shape[1], x.shape[1])
                x = reshaped_tensor.sum(dim=1) # [B, H]
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=ctx.get_group("tp"))
            out_batch_size = tree.flatten(inp)[0].shape[0]
            # print(f"[RANK {ctx.get_rank('tp')} L{layer_num} T{idx}] After AR \n")
            # save input
            outp = x[:out_batch_size]
        
            return outp

    # if ctx.get_rank('tp') == 0:
    #     # layer num, expert idx
    #     print(f"{layer_num} {local_expert_count}")



fmoe_faster_schedule = False
if switch_from_env('FMOE_FASTER_SCHEDULE_ENABLE', False):
    fmoe_faster_schedule = True
    from .fastermoe.schedule import _fmoe_general_global_forward


class FMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `mp_group` is a deprecated alias of `slice_group`
    * `moe_group` stands for the group of process that performs expert
    parallelism. The default value `None` means all processes. See the
    parallelism document for more details of the groups.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    * `gate_bias` is only valid for naive_gate and its subclasses, it means
    whether to add bias to the gate module.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,  # being deprecated
        slice_group=None,
        moe_group=None,
        top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
        gate_bias=True,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size

        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()

        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True

        if issubclass(gate, NaiveGate):
            self.gate = gate(d_model, num_expert, world_size, top_k, gate_bias=gate_bias)
        else:
            self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count_cpu = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count_cpu[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice, torch.tensor([fwd_expert_count[i]])))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def expert_fn_single(self, inp, fwd_expert_count, idx):
        r"""
        forward single expert for smart scheduling.
        """
        assert not self.experts_fused, "should not use fused experts"
        output = self.experts[idx](inp, fwd_expert_count)
        return output

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """

        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        gate_top_k_idx, gate_score = self.gate(moe_inp)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]
        with nvtx.annotate("FMoE FFN", color="red"):
            fwd = _fmoe_general_global_forward(
                moe_inp, gate_top_k_idx, self.expert_fn_single if fmoe_faster_schedule else self.expert_fn,
                self.num_expert, self.world_size,
                experts=self.experts
            )

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp, gate_top_k_idx

class pMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `total_experts` stands for the number of sub experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `comm` is NCCL Manager
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    * `gate_bias` is only valid for naive_gate and its subclasses, it means
    whether to add bias to the gate module.
    """

    def __init__(
        self,
        total_experts=32,
        d_model=1024,
        world_size=1,
        ctx=None, # NCCLManager
        top_k=2,
        gate=NaiveGate,
        expert=None,
        layer_num=0,
        pipeline=1,
        executor = None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
        gate_bias=True,
    ):
        super().__init__()
        self.total_experts = total_experts
        self.d_model = d_model
        self.world_size = world_size
        self.num_expert = total_experts // world_size
        # self.pipeline = pipeline
        self.pipe_stream = pipeline # [torch.cuda.Stream() for _ in range(self.pipeline)]
        self.executor = executor # [ThreadPoolExecutor(max_workers=1) for _ in range(self.pipeline)]
        
        self.ctx = ctx
        self.layer_num = layer_num

        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = expert(total_experts)
            self.experts_fused = True
        else:
            self.experts_fused = True

        if issubclass(gate, NaiveGate):
            self.gate = gate(d_model, self.num_expert, world_size, top_k, gate_bias=gate_bias)
        else:
            self.gate = gate(d_model, self.num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
    
    def __del__(self):
        self.cleanup()
        
    def cleanup(self):
        for executor in self.executor:
            executor.shutdown(wait=True)
        print("Executors shut down successfully.")
            
    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        return self.experts(inp, fwd_expert_count)
        

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.ctx.get_size('tp') > 1:
            ensure_comm_futures = []
            for i in range(len(self.pipe_stream)):
                with torch.cuda.stream(self.pipe_stream[i]):
                # print(f"ensure_comm: device={torch.cuda.current_device()}, stream={torch.cuda.current_stream()}, device={moe_inp.device}")
                # ensure_comm(moe_inp, self.ctx.get_group('tp'), i)
                    ensure_comm_futures.append(
                        self.executor[i].submit(ensure_comm, moe_inp, self.ctx.get_group('tp'), i)
                    )
            #     # torch.distributed.barrier()
            
            for future in ensure_comm_futures:
                future.result()  # Block until complete
                
        gate_top_k_idx, gate_score = self.gate(moe_inp)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]
        
        
        g = torch.chunk(gate_top_k_idx, len(self.pipe_stream), dim=0)
        chunk = torch.chunk(moe_inp, len(self.pipe_stream), dim=0) # [None for _ in range(pipeline)]
        f = [None]*len(self.pipe_stream)
        _fwd = [None]*len(self.pipe_stream)
        
        with nvtx.annotate("pMoE FFN", color="green"):
            for i in range(len(self.pipe_stream)):
                with torch.cuda.stream(self.pipe_stream[i]):
                    f[i] = self.executor[i].submit(_pmoe_general_global_forward, chunk[i].clone(), g[i].clone(), self.expert_fn, self.experts, self.total_experts, self.layer_num, len(self.pipe_stream), self.ctx, self.pipe_stream, i)
        
        
        for i in range(len(self.pipe_stream)):
            _fwd[i] = f[i].result()
        fwd = torch.cat(_fwd, dim=0)
        
            # if self.pipeline > 1:
            #     pipe_stream = [torch.cuda.Stream() for _ in range(2)]
            #     g = torch.chunk(gate_top_k_idx, self.pipeline, dim=0)
            #     chunk = torch.chunk(moe_inp, self.pipeline, dim=0) # [None for _ in range(pipeline)]
            #     # # assert _chunk[0].shape[0] == _chunk[1].shape[0]
            #     x0 = chunk[0].clone()  # new memory buffer
            #     x1 = chunk[1].clone()  # new memory buffer
            #     with ThreadPoolExecutor(max_workers=self.pipeline) as executor:
            #         f0 = executor.submit(_pmoe_general_global_forward, x0, g[0], self.expert_fn, self.experts, self.total_experts, self.layer_num, self.pipeline, self.ctx, pipe_stream, 0)
            #         f1 = executor.submit(_pmoe_general_global_forward, x1, g[1], self.expert_fn, self.experts, self.total_experts, self.layer_num, self.pipeline, self.ctx, pipe_stream, 1)
            #         b0, work0 = f0.result()
            #         b1, work1 = f1.result()
            #         work0.wait()
            #         work1.wait()
            #         torch.cuda.synchronize()
            #         out_batch_size = tree.flatten(moe_inp)[0].shape[0]
            #         outp = torch.cat([b0, b1], dim=0)
            #         fwd = outp[:out_batch_size]
            # else:
            # fwd = _pmoe_general_global_forward(
            #     moe_inp, gate_top_k_idx, self.expert_fn, 
            #     self.experts,
            #     self.total_experts, layer_num=self.layer_num,
            #     pipeline=self.pipeline,
            #     ctx = self.ctx,
            #     stream = self.pipe_stream,
            #     idx=0
            # )
        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:
            moe_outp = fwd
            # def view_func(tensor):
            #     dim = tensor.shape[-1]
            #     tensor = tensor.view(-1, self.top_k, dim)
            #     return tensor

            # moe_outp = tree.map_structure(view_func, fwd)

        # gate_score = gate_score.view(-1, 1, self.top_k)

        # def bmm_func(tensor):
        #     dim = tensor.shape[-1]
        #     tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
        #     return tensor

        # moe_outp = tree.map_structure(bmm_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp, gate_top_k_idx
