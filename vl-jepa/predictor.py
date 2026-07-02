import torch.nn as nn
import torch

class MultiModalPredictor(nn.Module):
  def __init__(self,vision_embed_dim,language_embed_dim,hidden_dim,output_dim):
    super().__init__()
    input_dim=vision_embed_dim+language_embed_dim
    self.mlp=nn.Sequential(
        nn.Linear(input_dim,hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim,output_dim),
        nn.GELU(),
        nn.Linear(output_dim,output_dim)
    )
  def forward(self,vision_embedding,language_embedding):
    combined_embedding=torch.cat([vision_embedding,language_embedding],dim=-1)
    return self.mlp(combined_embedding)