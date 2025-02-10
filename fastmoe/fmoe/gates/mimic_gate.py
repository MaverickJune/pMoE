from .base_gate import BaseGate
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

class MimicGate(BaseGate):
    r"""
    Get the probability distribution data from the csv file,
    and mimic the distribution of the expert selection.
    """
    
    def __init__(self, d_model, num_expert, world_size, top_k=1,  gpu_idx=-1, path="/workspace/pMoE/p_count_selected.csv"):
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
        token_board = torch.floor(expected_tokens).to(torch.int).to(x.device)
        remainder = n_tokens - token_board.sum().item()
        
        # Find the bin with the maximum allocation
        max_idx = torch.argmax(token_board)
        token_board[max_idx] += remainder
        gate_top_k_idx = torch.arange(self.num_expert, device=x.device).repeat_interleave(token_board).to(torch.long).to(x.device)
        gate_top_k_idx = gate_top_k_idx.unsqueeze(1)
        
        self.set_loss(torch.zeros(1, requires_grad=True).to(x.device))
        
        return gate_top_k_idx, gate_top_k_val
    

if __name__ == '__main__':
    # use torch.device('cuda:3') for MimicGate
    gate = MimicGate(1024, 8, 1, gpu_idx=3)
    x = torch.randn(100, 1024).to(3)
    idx, val = gate(x)
    print(idx)
    
