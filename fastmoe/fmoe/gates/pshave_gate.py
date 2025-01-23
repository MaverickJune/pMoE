from .base_gate import BaseGate
import torch
import torch.nn as nn
import torch.nn.functional as F

class PshaveGate(BaseGate):
    r"""
    This is a gate that can control the imbalance distribution of the experts.
    pshave = probability shave
    """
    
    def __init__(self, d_model, num_expert, world_size, top_k=1, imbalance_level=0.125, gate_bias=False):
        super().__init__(num_expert, world_size)
        assert top_k == 1, "Pshave gate only supports top_k = 1"
        
        self.num_expert = self.tot_expert
        self.world_size = world_size
        self.d_model = d_model
        self.imbalance_level = imbalance_level
        
        self.shaved_distribution = self.shave_distribution(imbalance_level, self.num_expert)
        
    def shave_distribution(self, imbalance_level, num_expert):
        p_board = torch.zeros(num_expert)
        p_board[0] = imbalance_level
        for i in range(1, num_expert):
            p_board[i] = (1 - imbalance_level) / (num_expert - 1)
        return p_board
    
    def forward(self, x):
        r"""
        The forward function of the pshave gate.
        """
        n_tokens = x.size(0)
        # print(f"x shape: {x.shape}")
        
        self.shaved_distribution = self.shaved_distribution.to(x.device)
        # print(f"Shaved distribution: {self.shaved_distribution}")
        
        # Set the token_board for each expert
        r_tokens = n_tokens
        zero_flag = False
        token_board = torch.zeros(self.num_expert, dtype=torch.int).to(x.device)
        for i in range(self.num_expert):
            if zero_flag:
                token_board[i] = 0
                continue
            curr_token = torch.ceil(n_tokens * self.shaved_distribution[i]).int().item()
            if r_tokens > curr_token:
                r_tokens -= curr_token
                token_board[i] = curr_token
            else:
                token_board[i] = r_tokens
                zero_flag = True
                
        # print(f"Token board: {token_board}")
                
        # Generate the gating result
        gate_top_k_idx = torch.arange(self.num_expert, device=x.device).repeat_interleave(token_board).to(torch.long).to(x.device)
        gate_top_k_idx = gate_top_k_idx.unsqueeze(1) # [n_tokens, 1]
        
        self.set_loss(torch.zeros(1, requires_grad=True).to(x.device))
        
        # Generate the gate score (trash value)
        gate_top_k_val = torch.zeros(n_tokens, 1, dtype=torch.bfloat16).to(x.device)
        
        return gate_top_k_idx, gate_top_k_val
    

if __name__ == '__main__':
    # Test pshave gate
    gate = PshaveGate(512, 8, 1)
    x = torch.randn(512, 512)
    idx, val = gate(x)
    print(idx)  