r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from .layers import FMoE, pMoE
from .linear import FMoELinear, pMoELinear
from .fastermoe.config import switch_from_env

import torch.distributed as dist
import tree
import nvtx
class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=False, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=False, rank=rank)
        self.w3 = FMoELinear(num_expert, d_model, d_hidden, bias=False, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x) * self.w3(inp, fwd_expert_count)
        x = self.h4toh(x, fwd_expert_count)
        return x


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        **kwargs
    ):
        def one_expert(d_model):
            return _Expert(1, d_model, d_hidden, activation, rank=0)
        
        expert = one_expert
        super().__init__(num_expert=num_expert, d_model=d_model, expert=expert, **kwargs)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output, idx = super().forward(inp)
        return output.reshape(original_shape), idx

class _pExpert(nn.Module):
    r"""
    An expert using 2 pMoELinear modules to balance the stochasic computation of experts
    between multiple worker.
    """
    def __init__(self, total_experts, d_model, d_hidden, activation, ctx):
        super().__init__()
        self.htoh4 = pMoELinear(total_experts, d_model, d_hidden, bias=False, ctx = ctx)
        self.h4toh = pMoELinear(total_experts, d_hidden, d_model, bias=False, ctx = ctx)
        self.w3 = pMoELinear(total_experts, d_model, d_hidden, bias=False, ctx = ctx)
        self.activation = activation
        self.ctx = ctx # context

    def _comm_forward(self, inp, experts, fwd_expert_count, dim):
        # x: [B x k, h] expert: [E, h, 4H]
        x = experts(inp, fwd_expert_count) # [B x k, 4H] get partial input value with same dimension
        # comm = self.ctx.mesh.get_group("tp")
        def reduce_scatter_column(tensor, dim, ctx):
            """
            Reduce-scatter the tensor from shape [b x k x n, H] to [b x k, H].

            Args:
                tensor (torch.Tensor): Input tensor of shape [b x k x n, H].
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
            assert H % world_size == 0, "Tensor must be divisible by world_size."
            
            # Compute chunk size
            chunk_size = H // world_size  # Each chunk corresponds to 4h

            # Split input tensor into chunks for reduce-scatter
            input_chunks = list(tensor.chunk(world_size, dim=dim))

            # Create output tensor for this rank
            output_tensor = torch.empty((bkn, chunk_size), dtype=tensor.dtype, device=tensor.device)
            # if ctx.get_rank("tp") == 0:
            #     print(f"shape of input tensor is {tensor.shape}")
            #     print(f"shape of input chunks is {input_chunks[0].shape}")
            #     print(f"shape of output tensor is {output_tensor.shape}")
            # with nvtx.annotate("FFN sub comm", color="green"):
            # Perform reduce-scatter
            # work = dist.reduce_scatter(output_tensor, input_chunks, op=dist.ReduceOp.SUM, group=group, async_op=True)
            work = dist.reduce_scatter(output_tensor, input_chunks, op=dist.ReduceOp.SUM, group=group, async_op=True)

            return output_tensor, work
        
        if dim == 1:
            outp, work = tree.map_structure(lambda t: reduce_scatter_column(t, dim, self.ctx), x)
        
        return outp, work # [B x 704]
    
    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        # [B x 2048] -> [B x 5632]
        # print(f"shape of input is {inp.shape}")
        x, work = self._comm_forward(inp, self.htoh4, fwd_expert_count, dim=1) 
        # print(f"shape 1 of x is {x.shape}")
        # [B x 704] [B x 704] -> 
        work.wait()
        x = self.activation(x) # * self.w3(inp, fwd_expert_count) # [B x 704] x [B x 5632]
        # print(f"shape 2 of x is {x.shape}")
        
        x = self.h4toh(x, fwd_expert_count)
        return x


class pMoETransformerMLP(pMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        total_experts=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        layer_num=0,
        pipeline=1,
        executor="none",
        ctx="none",
        **kwargs
    ):
        
        def fused_sub_experts(total_experts):
            return _pExpert(total_experts, d_model, d_hidden, activation, ctx)
        
        expert = fused_sub_experts
        super().__init__(total_experts=total_experts, d_model=d_model, ctx = ctx, expert=expert, layer_num=layer_num, pipeline=pipeline, executor = executor,**kwargs)
        self.mark_parallel_comm(self.ctx.get_group('dp'))

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output, idx = super().forward(inp)
        return output.reshape(original_shape), idx
