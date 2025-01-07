from transformers import AutoModelForCausalLM
from accelerate import infer_auto_device_map
import torch
import os
import json
import glog

@torch.no_grad()
def get_model(model_name, save_model_info=False):
    # device_map = infer_auto_device_map(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
    print(model.hf_device_map)
    model_save_path = os.path.join("./models_info", model_name.split("/")[-1])
    os.makedirs(model_save_path, exist_ok=True)
    
    model_info_save_path = os.path.join(model_save_path, "model_info.txt")
    config_save_path = os.path.join(model_save_path, "config.json")
    
    if save_model_info:
        with open(model_info_save_path, "w") as f:
            print(model, file=f)
            
        with open(config_save_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=4)
        
    return model

@torch.no_grad()
def save_model(model, model_name):
    MODEL_LIST = ["ibm-granite/granite-3.1-1b-a400m-instruct", "eastwind/tinymix-8x1b-chat"]
    if model_name not in MODEL_LIST:
        raise ValueError(f"Model name {model_name} not in the list of available models: {MODEL_LIST}")
    
    # model = model.to('cpu')
    model_weight_save_path = os.path.join("./models_weight", model_name.split("/")[-1])
    os.makedirs(model_weight_save_path, exist_ok=True)
    
    '''
    moe-expert modules: store in pure weight format (experts + gate)
    others: store in state_dict format
    '''
    
    if model_name == "eastwind/tinymix-8x1b-chat":
        # Save lm_head_weight
        torch.save(model.lm_head.state_dict(), os.path.join(model_weight_save_path, "lm_head_weight.pt"))
        
        # Save norm weight
        torch.save(model.model.norm.state_dict(), os.path.join(model_weight_save_path, "norm_weight.pt"))
        
        for idx in range(len(model.model.layers)):
            glog.info(f"Saving layer {idx}")
            layer = model.model.layers[idx]
            
            # Save self_attn
            torch.save(layer.self_attn.state_dict(), os.path.join(model_weight_save_path, f"layer_{idx}_self_attn.pt"))
            
            # Save input_layernorm
            torch.save(layer.input_layernorm.state_dict(), os.path.join(model_weight_save_path, f"layer_{idx}_input_layernorm.pt"))
            
            # Save post_layer_norm
            torch.save(layer.post_attention_layernorm.state_dict(), os.path.join(model_weight_save_path, f"layer_{idx}_post_layer_norm.pt"))
            
            # Save block_sparse_moe
            torch.save(layer.block_sparse_moe.gate.weight, os.path.join(model_weight_save_path, f"layer_{idx}_gate.pt"))
            for expert_idx, expert in enumerate(layer.block_sparse_moe.experts):
                torch.save(expert.w1.weight, os.path.join(model_weight_save_path, f"layer_{idx}_expert_{expert_idx}_w1.pt"))
                torch.save(expert.w2.weight, os.path.join(model_weight_save_path, f"layer_{idx}_expert_{expert_idx}_w2.pt"))
                torch.save(expert.w3.weight, os.path.join(model_weight_save_path, f"layer_{idx}_expert_{expert_idx}_w3.pt"))
                
            glog.info(f"Layer {idx} saved")
                
    elif model_name == "ibm-granite/granite-3.1-1b-a400m-instruct":
        raise NotImplementedError("Model not implemented yet")
        

if __name__ == '__main__':
    # model_name = 'ibm-granite/granite-3.1-1b-a400m-instruct'
    # model_name = "Qwen/Qwen1.5-MoE-A2.7B-Chat" # Currently Unavailable
    model_name = "eastwind/tinymix-8x1b-chat"
    model = get_model(model_name, save_model_info=True)
    save_model(model, model_name)
    