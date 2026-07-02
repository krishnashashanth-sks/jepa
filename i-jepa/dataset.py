import torch

class IJEPA_Dataset(torch.utils.data.Dataset):
  def __init__(self, base_dataset, transform, ijepa_masking_strategy):
    self.base_dataset = base_dataset
    self.transform = transform
    self.ijepa_masking_strategy = ijepa_masking_strategy

  def __len__(self):
    return len(self.base_dataset)

  def __getitem__(self, idx):
    img, _ = self.base_dataset[idx] # We don't need labels for self-supervised learning; img is now a PIL Image

    # Apply the initial image transformations (ToTensor, Normalize) on the PIL Image
    augmented_img = self.transform(img)

    # Apply I-JEPA masking strategy to get context and target views
    context_view, target_view = self.ijepa_masking_strategy(augmented_img)

    return context_view, target_view