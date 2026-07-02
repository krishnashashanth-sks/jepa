
import torch
import torch.nn as nn
from layers import TransformerBlock

class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, max_seq_len, embed_dim):
    super().__init__()
    self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
    self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

  def forward(self, x):
    # x is a tensor of token indices (batch_size, seq_len)
    token_embeds = self.token_embeddings(x)
    # Add positional embeddings to token embeddings
    # The position_embeddings tensor should broadcast correctly across batch dimension
    return token_embeds + self.position_embeddings[:, :x.size(1), :]

class LanguageTransformerEncoder(nn.Module):
  def __init__(self,vocab_size,max_seq_len,embed_dim,num_layers,num_heads,mlp_ratio,dropout=0.,include_pooling=True):
    super().__init__()
    self.include_pooling=include_pooling
    self.token_embedding=TokenEmbedding(vocab_size,max_seq_len,embed_dim)
    self.transformer_blocks=nn.ModuleList(
        TransformerBlock(embed_dim,num_heads,mlp_ratio,dropout)
        for _ in range(num_layers)
    )
    self.norm=nn.LayerNorm(embed_dim)
    if self.include_pooling:
      self.output_layer=nn.Linear(embed_dim,embed_dim)
  def forward(self,x):
    x=self.token_embedding(x)
    for block in self.transformer_blocks:
      x=block(x)
    x=self.norm(x)
    if self.include_pooling:
      x=x.mean(dim=1) 
      x=self.output_layer(x)
    return x