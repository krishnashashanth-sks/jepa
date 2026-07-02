import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False):
    super(ConvBlock,self).__init__()
    self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
    self.bn=nn.BatchNorm2d(out_channels)
    self.relu=nn.ReLU(inplace=True)
  def forward(self,x):
    return self.relu(self.bn(self.conv(x)))

class ContextEncoder(nn.Module):
  def __init__(self,in_channels=3,base_channels=64,output_dim=512):
    super(ContextEncoder,self).__init__()
    self.encoder=nn.Sequential(
        ConvBlock(in_channels,base_channels,kernel_size=7,stride=2,padding=3),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        ConvBlock(base_channels,base_channels*2),
        ConvBlock(base_channels*2,base_channels*4,stride=2),
        ConvBlock(base_channels*4,base_channels*8,stride=2),
        ConvBlock(base_channels*8,base_channels*16,stride=2)
    )
    self.avgpool=nn.AdaptiveAvgPool2d((1,1))
    self.fc=nn.Linear(base_channels*16,output_dim)
  def forward(self,x):
    x=self.avgpool(self.encoder(x))
    x=torch.flatten(x,1)
    return self.fc(x)
  
class Predictor(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(), # Using GELU for consistency with TransformerBlock
        nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, x):
    return self.mlp(x)
