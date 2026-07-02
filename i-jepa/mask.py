import random
import torch

class IJEPA_Masking(object):
  def __init__(self,img_size,patch_size,num_context_blocks,num_target_blocks):
    self.img_size=img_size
    self.patch_size=patch_size
    self.num_context_blocks=num_context_blocks
    self.num_target_blocks=num_target_blocks
    assert img_size % patch_size ==0 ,"Image dimension must be divisible by the patch size"
    self.num_patches_per_dim=img_size//patch_size
    self.total_patches=self.num_patches_per_dim
    self.all_patch_indices=[
        (r,c) for r in range(self.num_patches_per_dim) for c in range(self.num_patches_per_dim)
    ]
  def __call__(self,img_tensor):
    C,H,W=img_tensor.shape
    assert H==self.img_size and W==self.img_size,"input image tensor size mismatch"
    context_block_indices=random.sample(self.all_patch_indices,self.num_target_blocks)
    remaining_patch_indices=list(set(self.all_patch_indices)-set(context_block_indices))
    if len(remaining_patch_indices)<self.num_target_blocks:
      target_block_indices=random.sample(self.all_patch_indices,self.num_target_blocks)
    else:
      target_block_indices=random.sample(remaining_patch_indices,self.num_target_blocks)
    context_view_iamge=torch.zeros_like(img_tensor)
    target_view_image=torch.zeros_like(img_tensor)
    def copy_patches(source_img,dest_img,indices):
      for r,c in indices:
        r_start,r_end=r*self.patch_size,(r+1)*self.patch_size
        c_start,c_end=c*self.patch_size,(c+1)*self.patch_size
        dest_img[:, r_start:r_end, c_start:c_end] = source_img[:, r_start:r_end, c_start:c_end]
    copy_patches(img_tensor,context_view_iamge,context_block_indices)
    copy_patches(img_tensor,target_view_image,target_block_indices)
    return context_view_iamge,target_view_image