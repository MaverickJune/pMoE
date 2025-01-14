import torch

gate_data = torch.load("/home/wjbang/workspace/pMoE/pMoE/gate_data/enwik8_10000.pt")

for i, item in enumerate(gate_data):
    if i>=10:
        break
    print(item["layer_3_gate"])