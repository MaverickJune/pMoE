from .base_gate import BaseGate
import torch
import torch.nn as nn
import torch.nn.functional as F

class PshaveGate(BaseGate):
    r"""
    This is a gate that can control the imbalance distribution of the experts.
    pshave = probability shave
    """
    
    def __init__(self, d_model, num_expert, world_size, top_k=1, imbalance_level=0.125, gate_bias=False, gpu_idx=-1):
        super().__init__(num_expert, world_size)
        assert top_k == 1, "Pshave gate only supports top_k = 1"
        
        self.num_expert = self.tot_expert
        self.world_size = world_size
        self.d_model = d_model
        self.imbalance_level = imbalance_level
        self.gpu_idx = gpu_idx
        
    def shave_distribution(self, imbalance_level, num_expert):
        p_board = torch.zeros(num_expert, device=self.gpu_idx)
        max_idx = torch.randint(num_expert, (1,), device=self.gpu_idx).item()
        p_board[max_idx] = imbalance_level
        for i in range(0, num_expert):
            if i != max_idx:
                p_board[i] = (1 - imbalance_level) / (num_expert - 1)
        return p_board
    
    def forward(self, x):
        r"""
        The forward function of the pshave gate.
        """
        n_tokens = x.size(0)
        shaved_distribution = self.shave_distribution(self.imbalance_level, self.num_expert)
        
        expected_tokens = n_tokens * shaved_distribution
        token_board = torch.floor(expected_tokens).to(torch.long)
        remainder = n_tokens - token_board.sum().item()
        
        max_idx = torch.argmax(token_board)
        token_board[max_idx] += remainder
        
        # Generate the gate score (trash value)
        gate_top_k_val = torch.zeros(n_tokens, 1, dtype=torch.bfloat16, device=self.gpu_idx)
        
        return token_board, gate_top_k_val
    

if __name__ == '__main__':
    # Test pshave gate
    gate = PshaveGate(512, 8, 1)
    x = torch.randn(512, 512).to('cuda:0')
    idx, val = gate(x)
    print(idx)  