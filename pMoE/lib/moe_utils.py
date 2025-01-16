from transformers import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.deprecated.transfo_xl import TransfoXLLMHeadModel

from fmoe.transformer import pMoETransformerMLP, FMoETransformerMLP
import torch
import glog
from typing import Dict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import gc

MOE_DICT = {
    'fMoE': FMoETransformerMLP,
    'pMoE': pMoETransformerMLP
}
MODEL_DICT = {
    "transfo-xl/transfo-xl-wt103": TransfoXLLMHeadModel, 
    "mistralai/Mistral-7B-v0.1": MistralForCausalLM, 
    "meta-llama/Llama-3.1-70B-Instruct": LlamaForCausalLM
    }

@torch.no_grad()
def txl_wrapper(model, moe_name, moe_config: Dict, ctx, gpu_idx):
    if moe_name not in MOE_DICT.keys():
        raise ValueError(f"Invalid MoE name. Choose from {MOE_DICT.keys()}")
    
    # Get the configuration
    total_experts = moe_config.get("total_experts", 16)
    d_model = moe_config.get("d_model", 1024)
    d_hidden = moe_config.get("d_hidden", 4096)
    top_k = moe_config.get("top_k", 2)
    world_size = moe_config.get("world_size", None)
    moe_group = moe_config.get("moe_group", None)
    
    # Convert ffn -> pMoE
    for i in range(len(model.transformer.layers)):
        glog.info(f"Wrapping layer {i} from mlp to {moe_name}")
        if moe_name == 'pMoE':
            model.transformer.layers[i].pos_ff = pMoETransformerMLP(total_experts, d_model, d_hidden, top_k=top_k, ctx=ctx, layer_num=i)
        else: # fMoE
            # num_expert=num_experts, d_model=args.d_model, d_hidden=args.d_hidden, top_k=top_k, world_size=world_size, moe_group=group
            model.transformer.layers[i].pos_ff = FMoETransformerMLP(num_expert=total_experts, d_model=d_model, d_hidden=d_hidden, top_k=top_k, world_size=world_size, moe_group=moe_group)
        
    # wrapped_model = model.to(gpu_idx)
    # wrapped_model.eval()
    
    return model

@torch.no_grad()
def llama_wrapper(model, moe_name, moe_config: Dict, ctx, gpu_idx):
    if moe_name not in MOE_DICT.keys():
        raise ValueError(f"Invalid MoE name. Choose from {MOE_DICT.keys()}")
    
    # Get the configuration
    total_experts = moe_config.get("total_experts", 16)
    d_model = moe_config.get("d_model", 1024)
    d_hidden = moe_config.get("d_hidden", 4096)
    top_k = moe_config.get("top_k", 2)
    world_size = moe_config.get("world_size", None)
    moe_group = moe_config.get("moe_group", None)
    
    # Convert ffn -> pMoE
    for i in range(len(model.model.layers)):
        glog.info(f"Wrapping layer {i} from mlp to {moe_name}")
        if moe_name == 'pMoE':
            model.model.layers[i].mlp = pMoETransformerMLP(total_experts, d_model, d_hidden, top_k=top_k, ctx=ctx)
        else: # fMoE
            # num_expert=num_experts, d_model=args.d_model, d_hidden=args.d_hidden, top_k=top_k, world_size=world_size, moe_group=group
            model.model.layers[i].mlp = FMoETransformerMLP(num_expert=total_experts, d_model=d_model, d_hidden=d_hidden, top_k=top_k, world_size=world_size, moe_group=moe_group)
        
    # wrapped_model = model.to(gpu_idx)
    # wrapped_model.eval()
    
    return model

def clean():
    gc.collect()
    # https://stackoverflow.com/questions/57858433/how-to-clear-gpu-memory-after-pytorch-model-training-without-restarting-kernel
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

@torch.no_grad()
def tinymix_wrapper(model, moe_name, moe_config: Dict, ctx, gpu_rank, gpu_idx, weight_path="/home/wjbang/workspace/pMoE/pMoE/models/models_weight/tinymix-8x1b-chat"):
    if moe_name not in MOE_DICT.keys():
        raise ValueError(f"Invalid MoE name. Choose from {MOE_DICT.keys()}")
    
    # Get the configuration
    total_experts = moe_config.get("total_experts", 16)
    d_model = moe_config.get("d_model", 1024)
    d_hidden = moe_config.get("d_hidden", 4096)
    top_k = moe_config.get("top_k", 2)
    world_size = moe_config.get("world_size", None)
    moe_group = moe_config.get("moe_group", None)
    # print(f"d_model : {d_model}, {d_hidden}")
    # Convert ffn -> pMoE
    # model = model.to('cpu')
    for i in range(len(model.model.layers)):
        # glog.info(f"Wrapping layer {i} from mlp to {moe_name}")
        # print(f"layer number: {i} / {len(model.model.layers)}")
        if moe_name == 'pMoE':
            if gpu_rank == 0:
                print(f"Wrapping layer {i} / {len(model.model.layers)} from mlp to {moe_name}")
            model.model.layers[i].block_sparse_moe = pMoETransformerMLP(total_experts, d_model, d_hidden, top_k=top_k, layer_num=i, ctx=ctx).to(torch.bfloat16)
            gate = torch.load(f"{weight_path}/layer_{i}_gate.pt").to(gpu_idx)
            model.model.layers[i].block_sparse_moe.gate.gate.weight.copy_(gate)
            del gate
            clean()
            
            
            # w_1 = torch.tensor() # weight 랑 똑같은 형태의 텐서 (E x H/8 x 4H)
            # for i in range(total_experts):
            #     _w1 = torch.load(f"{weight_path}/layer_{i}_expert_{expert_idx}_w1.pt") # 1 x H x 4H
            #     _w1_eff = torch.split(w1, 8, dim=0)[gpu_rank] # 1 x H/8 x 4H
            #     w1[0] = _w1_eff # E x H/8 x 4H <-1 1 x H/8 x 4H (하나하나씩 쌓기)
            # weight.copy(w1) # rank , size 값 갖고 -> 서로 똑같은 buffer 사이즈를 공유
            # # 동일한 사이즈의 버퍼가 서로 다른 값을 갖는거지 
            w1_savebuf = []
            w2_savebuf = []
            w3_savebuf = []
            # _expert_idx = [E]
            # expert_idx = _expert_idx[rank*]
            for expert_idx in range(total_experts):
                w1 = torch.load(f"{weight_path}/layer_{i}_expert_{expert_idx}_w1.pt")
                w2 = torch.load(f"{weight_path}/layer_{i}_expert_{expert_idx}_w2.pt")
                w3 = torch.load(f"{weight_path}/layer_{i}_expert_{expert_idx}_w3.pt")
                
                # print(f"[Per Expert]: w1 shape: {w1.shape}, w2 shape: {w2.shape}, w3 shape: {w3.shape}")
                # print()
                w1_eff = torch.split(w1, w1.shape[1] // ctx.get_size('tp'), dim=1)[gpu_rank]
                w2_eff = torch.split(w2, w2.shape[1] // ctx.get_size('tp'), dim=1)[gpu_rank]
                w3_eff = torch.split(w3, w3.shape[1] // ctx.get_size('tp'), dim=1)[gpu_rank]  
                
                # print(f"[Per Slice]: w1 shape: {w1_eff.shape}, w2 shape: {w2_eff.shape}, w3 shape: {w3_eff.shape}")
                
                w1_savebuf.append(w1_eff)
                w2_savebuf.append(w2_eff)
                w3_savebuf.append(w3_eff)
            
            # 8 x 2048 x 5632
            # 2 gpu (1 node) -> 8 x 1024 x 5632
            # 4 gpu (2 node) -> 8 x 512 x 5632
            # 8 gpu (2 node) -> 8 x 512 x 5632
            # [8, 2816, 2048]
            # [8, 5632, 1024] -> [8, 2816, 2048]
            w1_savebuf = torch.stack(w1_savebuf, dim=0) # E x H/N x 4H (real htoh4 E x 4H x H/N)
            w2_savebuf = torch.stack(w2_savebuf, dim=0) # E x 4H/N x H (real htoh4 E x H x 4H/N)
            w3_savebuf = torch.stack(w3_savebuf, dim=0) # E x H/N x 4H
            # print(f"w1 shape: {w1_savebuf.shape}, w2 shape: {w2_savebuf.shape}, w3 shape: {w3_savebuf.shape}")
            model.model.layers[i].block_sparse_moe.experts.htoh4.weight.copy_(w1_savebuf)
            model.model.layers[i].block_sparse_moe.experts.h4toh.weight.copy_(w2_savebuf)
            model.model.layers[i].block_sparse_moe.experts.w3.weight.copy_(w3_savebuf)
            
            del w1, w2, w3, w1_eff, w2_eff
            clean()
        elif moe_name == 'fMoE':
            # raise NotImplementedError("Also need to change fmoe to w1, w2, w3 structure")
            if gpu_rank == 0:
                print(f"Wrapping layer {i} / {len(model.model.layers)} from mlp to {moe_name}")
            model.model.layers[i].block_sparse_moe = FMoETransformerMLP(num_expert=total_experts, d_model=d_model, d_hidden=d_hidden, top_k=top_k, world_size=world_size, moe_group=moe_group).to(torch.bfloat16)
            gate = torch.load(f"{weight_path}/layer_{i}_gate.pt").to(gpu_idx)
            model.model.layers[i].block_sparse_moe.gate.gate.weight.copy_(gate)
            del gate
            clean()
            
            '''
            fMoE weight loading
            - 각각의 GPU가 # of expert / # of GPU 만큼의 expert를 가지고 있음: total_experts / world_size
            - moe_group: world_size
            '''
            # TODO: Implement the weight loading for fMoE
            num_expert_per_gpu = total_experts // world_size
            expert_start_idx = num_expert_per_gpu * gpu_rank
            
            for idx in range(expert_start_idx, expert_start_idx + num_expert_per_gpu):
                w1 = torch.load(f"{weight_path}/layer_{i}_expert_{idx}_w1.pt")
                w2 = torch.load(f"{weight_path}/layer_{i}_expert_{idx}_w2.pt")
                w3 = torch.load(f"{weight_path}/layer_{i}_expert_{idx}_w3.pt")
                
                model.model.layers[i].block_sparse_moe.experts[idx].htoh4.weight.copy_(w1)
                model.model.layers[i].block_sparse_moe.experts[idx].h4toh.weight.copy_(w2)
                model.model.layers[i].block_sparse_moe.experts[idx].w3.weight.copy_(w3)
                
                del w1, w2, w3
                clean()
        else:
            raise ValueError(f"Invalid MoE name. Choose from {MOE_DICT.keys()}")
        
    return model

@torch.no_grad()
def deepseek_wrapper(model, moe_name, moe_config: Dict, ctx, gpu_rank, gpu_idx, weight_path="/home/wjbang/workspace/pMoE/pMoE/models/models_weight/deepseek-moe-16b-chat"):
    if moe_name not in MOE_DICT.keys():
        raise ValueError(f"Invalid MoE name. Choose from {MOE_DICT.keys()}")
    
    # Get the configuration
    total_experts = moe_config.get("total_experts", 16)
    d_model = moe_config.get("d_model", 1024)
    d_hidden = moe_config.get("d_hidden", 4096)
    top_k = moe_config.get("top_k", 2)
    world_size = moe_config.get("world_size", None)
    moe_group = moe_config.get("moe_group", None)
    
    # Convert ffn -> fMoE
    for i in range(1, len(model.model.layers)):
        if moe_name != 'fMoE':
            raise ValueError(f"Deepseek model only supports fMoE. Choose fMoE")
        
        if gpu_rank == 0:
            print(f"Wrapping layer {i} / {len(model.model.layers)} from mlp to {moe_name}")
        model.model.layers[i].mlp = FMoETransformerMLP(num_expert=total_experts, d_model=d_model, d_hidden=d_hidden, top_k=top_k, world_size=world_size, moe_group=moe_group, is_deepseek=True).to(torch.bfloat16)
        
        # Copy gate weight
        gate_weight = torch.load(f"{weight_path}/layer_{i}_gate.pt").to(gpu_idx)
        model.model.layers[i].mlp.gate.gate.weight.copy_(gate_weight)
        del gate_weight
        clean()
        
        # Copy expert weights
        num_expert_per_gpu = total_experts // world_size
        expert_start_idx = num_expert_per_gpu * gpu_rank
        
        for idx in range(expert_start_idx, expert_start_idx + num_expert_per_gpu):
            up_weight = torch.load(f"{weight_path}/layer_{i}_expert_{idx}_up_proj.pt")
            down_weight = torch.load(f"{weight_path}/layer_{i}_expert_{idx}_down_proj.pt")
            gate_weight = torch.load(f"{weight_path}/layer_{i}_expert_{idx}_gate_proj.pt")
            
            model.model.layers[i].mlp.experts[idx].up_proj.weight.copy_(up_weight)
            model.model.layers[i].mlp.experts[idx].down_proj.weight.copy_(down_weight)
            model.model.layers[i].mlp.experts[idx].gate_proj.weight.copy_(gate_weight)
            
            del up_weight, down_weight, gate_weight
            clean()
            
        # Copy shared expert weights
        shared_expert_weight = torch.load(f"{weight_path}/layer_{i}_shared_experts.pt") # state_dict
        model.model.layers[i].mlp.shared_experts.load_state_dict(shared_expert_weight)
        
    return model
 
# from models.tinymix_8x_1b_chat import MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from transformers import AutoModelForCausalLM

@torch.no_grad()
def load_tinymix(gpu_idx):
    model_name = "eastwind/tinymix-8x1b-chat"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, torch_dtype=torch.bfloat16).to(gpu_idx)
    
    # print(f"Device map : {model.hf_device_map}")
    
    num_params = sum(p.numel() for p in model.parameters())
    model_size = num_params / 1024 / 1024 / 1024
    
    # print(f"Model size: {model_size} GB")
    
    return model

@torch.no_grad()
def load_deepseek(gpu_idx):
    model_name = "deepseek-ai/deepseek-moe-16b-base"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True).to(gpu_idx)
    
    return model
    
    
def get_model_from_hf(model_name, partial=0.4):
    if model_name not in MODEL_DICT.keys():
        raise ValueError(f"Unsupported model name. Choose from {MODEL_DICT.keys()}")
    
    # TODO: Delete this part after implemetation
    # if 'meta-llama' not in model_name:
    #     raise NotImplementedError(f"Only Llama models are supported for now")
    
    # Get the model configuration and modify some values
    config = PretrainedConfig.from_pretrained(model_name)
    if hasattr(config, 'use_cache'):
        config.use_cache = False
    if hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = int(config.num_hidden_layers * partial)
    if hasattr(config, 'n_layer'):
        config.n_layer = int(config.n_layer * partial)
    
    # Get the model
    model = MODEL_DICT[model_name](config)
    
    return model