from layers import PatchEmbedding,TransformerBlock
import torch.nn as nn

class VisionTransformerEncoder(nn.Module):
  def __init__(self, img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_ratio=4., dropout=0., include_pooling=True):
    super().__init__()
    self.include_pooling = include_pooling
    self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
    num_patches = (img_size // patch_size)**2
    self.transformer_blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
        for _ in range(num_layers)
    ])
    self.norm = nn.LayerNorm(embed_dim)
    if self.include_pooling:
      self.output_layer = nn.Linear(embed_dim, embed_dim)
  def forward(self, x):
    x = self.patch_embedding(x)
    for block in self.transformer_blocks:
      x = block(x)
    x = self.norm(x)
    if self.include_pooling:
      x = x.mean(dim=1)
      x = self.output_layer(x)
    return x
