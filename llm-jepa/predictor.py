import torch.nn as nn
import torch

class Predictor(nn.Module):
  def __init__(self,d_model:int,hidden_dim:int,num_layers:int=2):
    super().__init__()
    layers=[]
    in_features=d_model
    for i in range(num_layers-1):
      layers.append(nn.Linear(in_features,hidden_dim))
      layers.append(nn.GELU())
      in_features=hidden_dim
    layers.append(nn.Linear(in_features,d_model))
    self.mlp=nn.Sequential(*layers)
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.mlp(x)