from transformers import AutoModelForCausalLM
import torch

def main():
    model_name = "deepseek-ai/deepseek-moe-16b-chat"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
    print(model.hf_device_map)

if __name__ == '__main__':
    main()