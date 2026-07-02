import torch
import random

class TextMasking(object):
  def __init__(self, max_seq_len, num_context_tokens, num_target_tokens, mask_token_id=0):
    self.max_seq_len = max_seq_len
    self.num_context_tokens = num_context_tokens
    self.num_target_tokens = num_target_tokens
    self.mask_token_id = mask_token_id
    assert (num_context_tokens + num_target_tokens) <= max_seq_len, "Sum of context and target tokens cannot exceed max_seq_len"

  def __call__(self, token_ids):
    assert token_ids.shape[0] == self.max_seq_len, "Token IDs sequence length mismatch"

    all_indices = list(range(self.max_seq_len))

    available_indices = [i for i, token in enumerate(token_ids) if token != self.mask_token_id]
    num_actual_context = min(self.num_context_tokens, len(available_indices))
    context_indices = random.sample(available_indices, num_actual_context)

    remaining_indices = list(set(all_indices) - set(context_indices))
    num_actual_target = min(self.num_target_tokens, len(remaining_indices))
    target_indices = random.sample(remaining_indices, num_actual_target)

    context_text_tokens = torch.full_like(token_ids, self.mask_token_id)
    for idx in context_indices:
      context_text_tokens[idx] = token_ids[idx]

    target_text_tokens = torch.full_like(token_ids, self.mask_token_id)
    for idx in target_indices:
      target_text_tokens[idx] = token_ids[idx]

    return context_text_tokens, target_text_tokens

class IJEPA_Masking(object):
  def __init__(self, img_size, patch_size, num_context_blocks, num_target_blocks):
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_context_blocks = num_context_blocks
    self.num_target_blocks = num_target_blocks
    assert img_size % patch_size == 0, "Image dimension must be divisible by the patch size"
    self.num_patches_per_dim = img_size // patch_size
    self.total_patches = self.num_patches_per_dim * self.num_patches_per_dim
    self.all_patch_indices = [
        (r, c) for r in range(self.num_patches_per_dim) for c in range(self.num_patches_per_dim)
    ]

  def __call__(self, img_tensor):
    C, H, W = img_tensor.shape
    assert H == self.img_size and W == self.img_size, "Input image tensor size mismatch"

    # Select context blocks
    context_block_indices = random.sample(self.all_patch_indices, self.num_context_blocks)

    # Select target blocks (ideally disjoint from context)
    remaining_patch_indices = list(set(self.all_patch_indices) - set(context_block_indices))
    if len(remaining_patch_indices) < self.num_target_blocks:
        # If not enough disjoint patches, sample from all patches
        target_block_indices = random.sample(self.all_patch_indices, self.num_target_blocks)
    else:
        target_block_indices = random.sample(remaining_patch_indices, self.num_target_blocks)

    context_view_image = torch.zeros_like(img_tensor)
    target_view_image = torch.zeros_like(img_tensor)

    def copy_patches(source_img, dest_img, indices):
      for r, c in indices:
        r_start, r_end = r * self.patch_size, (r + 1) * self.patch_size
        c_start, c_end = c * self.patch_size, (c + 1) * self.patch_size
        dest_img[:, r_start:r_end, c_start:c_end] = source_img[:, r_start:r_end, c_start:c_end]

    copy_patches(img_tensor, context_view_image, context_block_indices)
    copy_patches(img_tensor, target_view_image, target_block_indices)

    return context_view_image, target_view_image