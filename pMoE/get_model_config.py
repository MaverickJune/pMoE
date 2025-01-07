from transformers import PretrainedConfig
import json
import os
import glog

def get_and_save_model_config(model_name="meta-llama/Llama-3.1-70B-Instruct", save_path='./save_config', return_config=False, print_config=False):
    config = PretrainedConfig.from_pretrained(model_name)
    save_path = os.path.join(save_path, model_name.replace('/', '_') + '.json')
    
    if print_config:
        glog.info(f"Model Config")
        glog.info(f"{config}")
    
    with open(save_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
        
    glog.info(f"Model Config saved at {save_path}")
    
    if return_config:
        return config
    
    
if __name__ == '__main__':
    # get_and_save_model_config(model_name="meta-llama/Llama-3.1-70B-Instruct")
    # get_and_save_model_config(model_name="meta-llama/Llama-2-7b-hf")
    # get_and_save_model_config(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    # get_and_save_model_config(model_name="mistralai/Mistral-7B-v0.1")
    get_and_save_model_config(model_name="transfo-xl/transfo-xl-wt103")
    