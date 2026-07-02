import torch
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

class DistributedTrainer:
  def __init__(self, model, optimizer, data_loader, rank, world_size, amp_enabled=False):
    self.rank = rank
    self.world_size = world_size

    # Determine the device
    if torch.cuda.is_available():
        self.device = torch.device(f"cuda:{rank}")
    else:
        self.device = torch.device("cpu")
    print(f"Rank {self.rank} is using device: {self.device}")

    self.model = model.to(self.device)

    # Conditionally enable AMP
    self.amp_enabled = amp_enabled and (self.device.type == 'cuda')
    self.scaler = GradScaler() if self.amp_enabled else None

    # Conditionally wrap with DDP
    if self.world_size > 1 and self.device.type == 'cuda':
        self.ddp_model = DistributedDataParallel(self.model, device_ids=[self.rank])
    else:
        self.ddp_model = self.model # Use the unwrapped model if not distributed or not on CUDA
        if self.world_size > 1 and self.device.type == 'cpu':
            print(f"Warning: Rank {self.rank} is configured for distributed training (world_size > 1) but running on CPU. DDP will not be used.")

    self.optimizer = optimizer
    self.data_loader = data_loader


  def train_one_epoch(self, epoch):
    if self.world_size > 1 and hasattr(self.data_loader.sampler, 'set_epoch'):
        self.data_loader.sampler.set_epoch(epoch)

    self.ddp_model.train()
    total_loss = 0.0
    for batch_idx, videos in enumerate(self.data_loader):
      videos = videos.to(self.device, non_blocking=True)
      self.optimizer.zero_grad()
      if self.amp_enabled:
        with autocast():
          loss, _, _ = self.ddp_model(videos)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
      else:
        loss, _, _ = self.ddp_model(videos)
        loss.backward()
        self.optimizer.step()
      total_loss += loss.item()
      if self.rank == 0 and batch_idx % 100 == 0:
        print(f"Rank {self.rank}, Epoch {epoch}, Batch {batch_idx}/{len(self.data_loader)}, Loss: {loss.item():.4f}")

    # All-reduce average loss across processes
    avg_loss = torch.tensor(total_loss / len(self.data_loader)).to(self.device)
    if self.world_size > 1:
      dist.reduce(avg_loss, dst=0, op=dist.ReduceOp.SUM)
      if self.rank == 0:
         avg_loss_global = avg_loss.item() / self.world_size # Divide by world_size to get true average
         print(f"Epoch {epoch} finished. Average Loss: {avg_loss_global:.4f}")
    elif self.rank == 0:
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss.item():.4f}")

  def save_checkpoint(self, epoch, path="checkpoint.pth"):
    if self.rank == 0:
      torch.save(
                {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.amp_enabled else None,
            }, path)
      print(f"Checkpoint saved at {path}")