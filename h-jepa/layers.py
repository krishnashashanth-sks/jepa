import random
import torch.nn as nn
import torch

class ContextEncoder(nn.Module):
  def __init__(self,input_channels,hidden_dims,output_dim,hierarchical_levels=1):
    super(ContextEncoder,self).__init__()
    self.heiarchical_levels=hierarchical_levels
    self.encoders=nn.ModuleList()
    if isinstance(input_channels,list):
      assert len(input_channels)==hierarchical_levels,"Input channels list must match hierarchical_levels"
      _input_channels=input_channels
      print("Note: ContextEncoder is designed to handle hierarchical inputs, but currently processes the first level for demonstration.")
    else:
      _input_channels=input_channels
    layers=[
        nn.Conv2d(_input_channels,hidden_dims[0],kernel_size=3,stride=2,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_dims[0])
    ]
    for i in range(len(hidden_dims)-1):
      layers.extend([
          nn.Conv2d(hidden_dims[i],hidden_dims[i+1],kernel_size=3,stride=1,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(hidden_dims[i+1])
      ])
    self.conv_layers=nn.Sequential(*layers)
    self.fc_input_dim=hidden_dims[-1]*(4*4)
    self.fc=nn.Linear(self.fc_input_dim,output_dim)
  def forward(self,x):
    if isinstance(x,list):
      x_processed=self.conv_layers(x[0])
    else:
      x_processed=self.conv_layers(x)
    x_processed=torch.flatten(x_processed,1)
    context_embedding=self.fc(x_processed)
    return context_embedding

class TargetEncoder(nn.Module):
  def __init__(self,input_channels,hidden_dims,output_dim,hierarchical_levels=1):
    super(TargetEncoder,self).__init__()
    self.hierarchical_levels=hierarchical_levels
    self.encocer=nn.ModuleList()
    if isinstance(input_channels,list):
      assert len(input_channels)==hierarchical_levels,""
      _input_channels=input_channels[0]
      print("Note: TargetEncoder is designed to handle hierarchical inputs, but currently processes the first level for demonstration.")
    else:
      _input_channels=input_channels
    layers=[
        nn.Conv2d(_input_channels,hidden_dims[0],kernel_size=3,stride=2,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_dims[0])
    ]
    for i in range(len(hidden_dims)-1):
      layers.extend([
          nn.Conv2d(hidden_dims[i],hidden_dims[i+1],kernel_size=3,stride=2,padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(hidden_dims[i+1])
      ])
    self.conv_layers=nn.Sequential(*layers)
    self.fc_input_dim=hidden_dims[-1]*(4*4)
    self.fc=nn.Linear(self.fc_input_dim,output_dim)
  def forward(self,x):
    if isinstance(x,list):
      x_processed=self.conv_layers(x[0])
    else:
      x_processed=self.conv_layers(x)
    x_processed = torch.flatten(x_processed, 1) # Flatten for the fully connected layer
    target_embedding = self.fc(x_processed)
    return target_embedding
  
class RandomPatchMasking(nn.Module):
    def __init__(self, num_patches, min_patch_size_ratio, max_patch_size_ratio, mask_value=0.0):
        super(RandomPatchMasking, self).__init__()
        self.num_patches = num_patches
        self.min_patch_size_ratio = min_patch_size_ratio
        self.max_patch_size_ratio = max_patch_size_ratio
        self.mask_value = mask_value

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a PyTorch Tensor, but got {type(x)}")

        is_single_image = False
        if x.dim() == 3: # (C, H, W) for a single image, add batch dimension
            x = x.unsqueeze(0)
            is_single_image = True
        elif x.dim() != 4: # Assuming NCHW format for batches
            raise ValueError(f"Expected input tensor to be 3-dimensional (C, H, W) or 4-dimensional (N, C, H, W), but got {x.dim()} dimensions.")

        batch_size, channels, H, W = x.shape
        masked_x = x.clone()

        for _ in range(self.num_patches):
            # Randomly determine patch size
            patch_H = int(random.uniform(self.min_patch_size_ratio, self.max_patch_size_ratio) * H)
            patch_W = int(random.uniform(self.min_patch_size_ratio, self.max_patch_size_ratio) * W)

            # Ensure patch size is at least 1
            patch_H = max(1, patch_H)
            patch_W = max(1, patch_W)

            # Randomly determine patch top-left corner
            start_H = random.randint(0, H - patch_H)
            start_W = random.randint(0, W - patch_W)

            # Apply mask to all channels for all items in the batch
            masked_x[:, :, start_H : start_H + patch_H, start_W : start_W + patch_W] = self.mask_value

        if is_single_image:
            masked_x = masked_x.squeeze(0) # Remove batch dimension if it was added

        return masked_x

class PredictorNetwork(nn.Module):
  def __init__(self,context_embedding_dim,target_embedding_dim,hidden_dims):
    super(PredictorNetwork,self).__init__()
    layers=[
        nn.Linear(context_embedding_dim,hidden_dims[0]),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dims[0]),
    ]
    for i in range(len(hidden_dims)-1):
      layers.extend([
          nn.Linear(hidden_dims[0],hidden_dims[i+1]),
          nn.ReLU(),
          nn.BatchNorm1d(hidden_dims[i+1]),
      ])
      layers.append(nn.Linear(hidden_dims[-1],target_embedding_dim))
      self.predictor=nn.Sequential(*layers)
    def forward(self,context_embedding):
      predicted_target_embedding=self.predictor(context_embedding)
      return predicted_target_embedding