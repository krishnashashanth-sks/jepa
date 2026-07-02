import torch
import torchvision.transforms as transforms
from decord import VideoReader,cpu
import numpy as np
import random

class VideoTransform:
  def __init__(self,target_size=224,num_frames=16,frame_rate=30,temporal_sample_type='uniform',mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]):
    self.target_size=target_size
    self.num_frames=num_frames
    self.frame_rate=frame_rate
    self.temporal_sample_type=temporal_sample_type
    self.mean=mean
    self.std=std
    self.spatial_transform=transforms.Compose([
        transforms.RandomResizedCrop(target_size,scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
    ])
    self.normalize=transforms.Normalize(mean=self.mean,std=self.std)

  def _sample_frames(self,vr):
    total_frames=len(vr)
    if total_frames<self.num_frames:
      indices=np.arange(total_frames).tolist()
      repeat_factor=int(np.ceil(self.num_frames/total_frames))
      indices=indices*repeat_factor
      indices=indices[:self.num_frames]
      return sorted(list(set(indices)))

    if self.temporal_sample_type=='uniform':
      interval=total_frames/self.num_frames
      indices=np.arange(0,total_frames,interval).astype(int)
      indices=indices[:self.num_frames]
    elif self.temporal_sample_type=="random_start":
      max_start_frame=total_frames-self.num_frames
      start_frame=random.randint(0,max_start_frame)if max_start_frame>0 else 0
      indices=np.arange(start_frame,start_frame+self.num_frames).astype(int)
    elif self.temporal_sample_type=="random_subclip":
      if total_frames<self.num_frames:
        indices=np.arange(total_frames).tolist()
        indices+=[indices[-1]]*(self.num_frames-total_frames)
      else:
        start_idx=random.randint(0,total_frames-self.num_frames)
        indices=list(range(start_idx,start_idx+self.num_frames))
    else:
      raise ValueError(f"Unknown temporal_sample_type: {self.temporal_sample_type}")

    if len(indices)<self.num_frames:
      indices=np.pad(indices,(0,self.num_frames-len(indices)),'edge').tolist()
    elif len(indices)> self.num_frames:
      indices=indices[:self.num_frames]

    return sorted(list(set(indices)))

  def __call__(self,video_path):
    try:
      vr=VideoReader(video_path,ctx=cpu(0))
      sampled_indices=self._sample_frames(vr)
      valid_indices=[idx for idx in sampled_indices if idx<len(vr)]

      if not valid_indices:
        print(f"Warning: No valid frames sampled for {video_path}. Returning dummy data.")
        return torch.zeros(3,self.num_frames,self.target_size,self.target_size)

      frames=vr.get_batch(valid_indices).asnumpy()
      frames=torch.from_numpy(frames).permute(0,3,1,2) # (N_frames, C, H, W)

      transformed_frames=[]
      seed=np.random.randint(2147483647)
      for i in range(frames.shape[0]):
        img=transforms.ToPILImage()(frames[i])
        random.seed(seed)
        torch.manual_seed(seed)
        transformed_img = self.spatial_transform(img)
        transformed_frames.append(transforms.ToTensor()(transformed_img)) # Convert PIL to Tensor [0,1]

      transformed_frames=torch.stack(transformed_frames) # (N_frames, C, H, W), values now [0,1]
      video_tensor=self.normalize(transformed_frames)
      video_tensor=video_tensor.permute(1,0,2,3) # (C, N_frames, H, W)

      return video_tensor
    except Exception as e:
      print(f"Error processing video {video_path}: {e}")
      return torch.zeros(3,self.num_frames,self.target_size,self.target_size)