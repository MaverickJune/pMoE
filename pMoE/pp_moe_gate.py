import csv
import torch
import torch.nn as nn

class ppmoe_gate(nn.Module):
    def __init__(self, gate_path, n_experts, mode=0):
        super(ppmoe_gate, self).__init__()
        self.gate_imbalance = None
        self.n_experts = n_experts
        self.prob_board = None
        
        '''
        mode 0: read gate imbalance from csv file
        mode 1: read actual distribution from csv file
        '''
        
        if mode == 0:
            with open(gate_path, "r", newline="") as f:
                tmp = []
                reader = csv.reader(f)
                data_all = list(reader)
                for idx, data in enumerate(data_all):
                    if idx == 0:
                        continue # skip header
                    tmp.extend(data)
                tmp = list(map(float, tmp))
            self.gate_imbalance = torch.tensor(tmp)
            assert self.gate_imbalance.dim() == 1, "gate imbalance should be 1-d tensor"
            
            self.prob_board = self.generate_dist(self.gate_imbalance, n_experts)
        elif mode == 1:
            raise NotImplementedError("mode 1 is not implemented yet")
        else:
            raise ValueError("mode should be 0 or 1")
        
    def generate_dist(self, imbalance_scores, n_experts):
        prob_board = []
        
        for i in range(len(imbalance_scores)):
            selected_idx = i
            selected_idx = torch.randint(0, len(imbalance_scores), (1,)).item()
            max_prob = imbalance_scores[selected_idx]
            remaining_prob = 1 - max_prob
            n_remaining_experts = n_experts - 1
            random_probs = torch.distributions.Dirichlet(torch.ones(n_remaining_experts)).sample()
            scaled_probs = random_probs * remaining_prob
            
            selected_expert_idx = torch.randint(0, n_experts, (1,)).item()
            probabilities = torch.zeros(n_experts)
            probabilities[selected_expert_idx] = max_prob
            other_indices = [i for i in range(n_experts) if i != selected_expert_idx]
            probabilities[other_indices] = scaled_probs
            
            prob_board.append(probabilities)
        
        return torch.stack(prob_board)
        
    def forward(self, x):
        n_tokens = x.size(0)
        expert_idx_board = torch.zeros(n_tokens)
        
        # select probability distribution
        selected_idx = torch.randint(0, len(self.prob_board), (1,)).item()
        selected_prob = self.prob_board[selected_idx]
        
        for idx in range(n_tokens):
            expert_idx_board[idx] = torch.multinomial(selected_prob, 1).item()
        
        return expert_idx_board
    
    
def test():
    gate_path = "/home/wjbang/workspace/pMoE/pMoE/wikitext_squad_top1.csv"
    n_experts = 8
    gate = ppmoe_gate(gate_path, n_experts)
    x = torch.randn(10, 768)
    expert_idx_board = gate(x)
    print(expert_idx_board)
    
if __name__ == "__main__":
    test()
        
        