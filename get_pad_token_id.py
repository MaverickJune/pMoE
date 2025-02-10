from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    pad = "<|pad|>"
    tokenizer.add_special_tokens({"pad_token": pad})
    tokenizer.padding_side = "left"
    
print(tokenizer.pad_token_id)