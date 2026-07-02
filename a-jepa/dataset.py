import torch
from torch.utils.data import Dataset

class DummyAudioDataset(Dataset):
  def __init__(self,num_samples,sample_rate,duration,transform=None):
    self.num_samples=num_samples
    self.sample_rate=sample_rate
    self.duration=duration
    self.transform=transform
    self.waveform_length=int(sample_rate*duration)
  def __len__(self):
    return self.num_samples
  def __getitem__(self,idx):
    waveform=torch.randn(self.waveform_length)
    if self.transform:
      context_mel,target_mel=self.transform(waveform)
      return context_mel,target_mel
    return waveform