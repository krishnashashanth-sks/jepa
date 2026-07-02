import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
  def __init__(self,img_size,patch_size,in_channels,embed_dim):
    super().__init__()
    self.img_size=img_size
    self.patch_size=patch_size
    self.in_channels=in_channels
    self.embed_dim=embed_dim
    assert img_size % patch_size ==0,"Image dimension must be divisible by the patch size"
    num_patches=(img_size//patch_size)**2
    self.patcher=nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
    self.positions=nn.Parameter(torch.randn(1,num_patches,embed_dim))
  def forward(self,x):
    x=self.patcher(x)
    x=rearrange(x, 'b c h w -> b (h w) c') # Corrected line
    return x+self.positions

class MultiHeadSelfAttention(nn.Module):
  def __init__(self,embed_dim,num_heads):
    super().__init__()
    self.embed_dim=embed_dim
    self.num_heads=num_heads
    self.head_dim=embed_dim//num_heads
    assert self.head_dim* num_heads==self.embed_dim,"embed_dim must be divisible by the num_heads"
    self.qkv_proj=nn.Linear(embed_dim,embed_dim*3,bias=False)
    self.out_proj=nn.Linear(embed_dim,embed_dim)
  def forward(self, x):
    batch_size, num_patches, _ = x.shape
    qkv = self.qkv_proj(x).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
    attn_scores = torch.matmul(q, k.transpose(-1, -2) / (self.head_dim**0.5))
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
    return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.):
    super().__init__()
    self.norm1 = nn.LayerNorm(embed_dim)
    self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
    self.norm2 = nn.LayerNorm(embed_dim)
    mlp_hidden_dim = int(embed_dim * mlp_ratio)
    self.mlp = nn.Sequential(
        nn.Linear(embed_dim, mlp_hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden_dim, embed_dim),
        nn.Dropout(dropout)
    )
  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    return x + self.mlp(self.norm2(x))
