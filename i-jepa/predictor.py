import torch.nn as nn
from layers import TransformerBlock
  
class VisionPredictor(nn.Module):
  def __init__(self, embed_dim, num_layers, num_heads, mlp_ratio=4., dropout=0.):
    super().__init__()
    self.transformer_blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
        for _ in range(num_layers)
    ])
    self.norm = nn.LayerNorm(embed_dim)

  def forward(self, x): # x shape: (batch_size, embed_dim)
    # Add a sequence dimension for the transformer blocks (batch_size, 1, embed_dim)
    x = x.unsqueeze(1)
    for block in self.transformer_blocks:
      x = block(x)
    x = self.norm(x)
    # Remove the sequence dimension to return to (batch_size, embed_dim)
    x = x.squeeze(1)
    return x
