import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        # Self-attention
        _mask = (mask == 0) 
        attn_output, _ = self.self_attention(x, x, x, key_padding_mask=_mask)
        # Add & norm
        x = self.layer_norm(x + attn_output)
        return x

class pMoETransformer(nn.Module):
    def __init__(self, num_heads, d_model, d_hidden, num_layers, mlp, total_experts, top_k, ctx):
        super(pMoETransformer, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    AttentionLayer(d_model, num_heads),
                    mlp(total_experts, d_model, d_hidden, top_k=top_k, ctx=ctx)
                )
            )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer[0](x, mask)
            x = layer[1](x, mask)
        return x
    
class MoETransformer(nn.Module):
    def __init__(self, num_heads, d_model, d_hidden, num_layers, mlp, num_experts, top_k, world_size, group):
        super(MoETransformer, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    AttentionLayer(d_model, num_heads),
                    mlp(num_expert=num_experts, d_model=d_model, d_hidden=d_hidden, top_k=top_k, world_size=world_size, moe_group=group)
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
if __name__ == "__main__":
    # Parameters
    batch_size = 8
    seq_len = 16
    embed_dim = 64
    num_heads = 4
    hidden_dim = 128
    num_layers = 6

    # Input tensor
    x = torch.randn(seq_len, batch_size, embed_dim)

    # Transformer model
    model = pMoETransformer(embed_dim, num_heads, hidden_dim, num_layers)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
