import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from layers import *
from trainer import DistributedTrainer
from inference import run_inference
from dataset import WebVideoDataset
from transform import VideoTransform
import torch.optim as optim
from model import VJEPA
import os
from args import Args
from utils import create_dummy_video
import webdataset as wds
import json

args=Args()
# Let's re-instantiate with consistent parameters
video_frames = 16
image_size = 224
patch_size_t = 2 # Temporal patch size
patch_size_h = 16 # Height patch size
patch_size_w = 16 # Width patch size
embed_dim = 768
num_heads = 12
depth = 12

# 1. VideoEncoder (Student)
student_encoder_instance = VideoEncoder(
    img_size=(video_frames, image_size, image_size),
    patch_size=(patch_size_t, patch_size_h, patch_size_w),
    in_channels=3,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    has_cls_token=True
)

# 2. MaskingGenerator
# Calculate input_size for masking_generator (P_T, P_H, P_W)
num_patches_t = video_frames // patch_size_t
num_patches_h = image_size // patch_size_h
num_patches_w = image_size // patch_size_w
masking_generator_instance = MaskingGenerator(
    input_size=(num_patches_t, num_patches_h, num_patches_w),
    num_masking_patches=int(0.6 * num_patches_t * num_patches_h * num_patches_w) # Example: 60% of patches
)

# 3. PredictionHead
decoder_embed_dim = 512 # Can be different from encoder_embed_dim
prediction_head_instance = PredictionHead(
    embed_dim=embed_dim, # Input dim from student encoder
    decoder_embed_dim=decoder_embed_dim,
    out_dim=embed_dim # Output dim to match teacher features
)

# 4. VJEPA Model
vjepa_model = VJEPA(
    student_encoder=student_encoder_instance,
    masking_generator=masking_generator_instance,
    prediction_head=prediction_head_instance,
    mask_ratio=0.6, # Example mask ratio
    teacher_momentum=0.996
)

print("VJEPA model, StudentEncoder, MaskingGenerator, and PredictionHead instances created.")
print(f"Total parameters in VJEPA model: {sum(p.numel() for p in vjepa_model.parameters()):,}")
print(f"Trainable parameters in VJEPA model (student_encoder and prediction_head): {sum(p.numel() for p in vjepa_model.parameters() if p.requires_grad):,}")

# 1. Create a directory for dummy shards
output_dir = '/content/my_video_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

num_dummy_shards = 2 # Create 2 dummy shards
num_videos_per_shard = 5 # Each shard will have 5 dummy videos

print(f"Creating {num_dummy_shards} dummy webdataset shards in {output_dir}...")

for i in range(num_dummy_shards):
    shard_filename = os.path.join(output_dir, f'shard_{i:06d}.tar')
    with wds.TarWriter(shard_filename) as sink:
        for j in range(num_videos_per_shard):
            video_id = f'video_{i}_{j}'
            temp_video_path = f'/tmp/{video_id}.mp4'

            # Create dummy video
            create_dummy_video(temp_video_path, frames=16, height=224, width=224, fps=10)

            # Read video bytes
            with open(temp_video_path, 'rb') as f:
                video_bytes = f.read()

            # Create dummy metadata
            metadata = {'label': f'category_{j % 2}', 'description': f'A dummy video for testing {video_id}'}

            # Write to shard
            sink.write({
                "__key__": video_id,
                "mp4": video_bytes,
                "json": json.dumps(metadata)
            })

            # Clean up temporary video file
            os.remove(temp_video_path)
    print(f"  Created shard: {shard_filename} with {num_videos_per_shard} videos.")

# 2. Update the data_path_pattern variable to correctly match the generated filenames
data_path_pattern = os.path.join(output_dir, 'shard_{000000..00000' + str(num_dummy_shards-1) + '}.tar')


# 1. Instantiate VideoTransform (re-using the previous instance)
video_transform_instance = VideoTransform(
    target_size=224,
    num_frames=16,
    frame_rate=30,
    temporal_sample_type='uniform'
)

# 2. Use the actual total_dataset_size from kernel state
total_dataset_size = num_dummy_shards * num_videos_per_shard # Ensure this reflects the actual dummy data count
print(f"Using actual total_dataset_size: {total_dataset_size}")

# 3. Instantiate WebVideoDataset with the total_dataset_size
# Re-instantiate to use the updated WebVideoDataset class
web_video_dataset_instance = WebVideoDataset(
    data_path_pattern=data_path_pattern, # From previous steps (now updated)
    video_transform=video_transform_instance, # From previous steps
    total_size=total_dataset_size,
    shard_shuffle=False, # Handled by DistributedSampler
    num_workers=args.world_size if args.world_size > 1 else 4 # Adjust num_workers based on distributed setup
)

# 4. Create the webdataset pipeline
dataset_pipeline = web_video_dataset_instance.create_dataset()

# 5. Instantiate DistributedSampler
# The 'args' object (with rank, world_size) is assumed to be available from init_distributed_mode
if args.world_size > 1:
    sampler = DistributedSampler(
        dataset_pipeline,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True # Sampler handles shuffling
    )
    print(f"Initialized DistributedSampler for rank {args.rank} out of {args.world_size} replicas.")
else:
    sampler = None # No sampler needed for single-process
    print("Running in single-process mode, DistributedSampler not used.")

# 6. Instantiate torch.utils.data.DataLoader
# Adjust batch_size and num_workers based on your GPU memory and system resources
batch_size=8

# The sampler itself takes care of shuffling.
data_loader = data.DataLoader(
    dataset_pipeline,
    batch_size=batch_size, # From kernel state
    num_workers=web_video_dataset_instance.num_workers,
    pin_memory=True, # Speeds up data transfer to GPU
    sampler=sampler, # Pass the instantiated sampler here
    drop_last=True # Useful for distributed training to ensure equal batch sizes
)

# 1. Define the total number of training epochs
num_epochs = 10 # You can adjust this number

# 2. Instantiate an optimizer (e.g., AdamW)
# Using the vjepa_model instance created in a previous step
optimizer = optim.AdamW(vjepa_model.parameters(), lr=1e-4) # Adjust learning rate as needed

# Determine if AMP should be enabled based on CUDA availability
amp_enabled = torch.cuda.is_available()
print(f"AMP (Automatic Mixed Precision) will be {'enabled' if amp_enabled else 'disabled'}.")

# 3. Instantiate the DistributedTrainer class
trainer = DistributedTrainer(
    model=vjepa_model,
    optimizer=optimizer,
    data_loader=data_loader,
    rank=args.rank,
    world_size=args.world_size,
    amp_enabled=amp_enabled
)

print(f"Starting training for {num_epochs} epochs...")

# 4. Create a for loop for training
for epoch in range(num_epochs):
    # 5. Call the train_one_epoch method
    trainer.train_one_epoch(epoch)

    # 6. Save checkpoint periodically (e.g., every 5 epochs, or after the last epoch)
    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
        trainer.save_checkpoint(epoch, path=f"checkpoint_epoch_{epoch+1}.pth")

print("Training complete.")


# 1. Define the path to a dummy video for inference
# Ensure the directory exists and create a dummy video if it doesn't.
inference_video_dir = '/content/my_video_data'
inference_video_path = os.path.join(inference_video_dir, 'inference_test_video.mp4')

if not os.path.exists(inference_video_dir):
    os.makedirs(inference_video_dir)

if not os.path.exists(inference_video_path):
    print(f"Creating dummy inference video: {inference_video_path}...")
    # Assuming create_dummy_video function is available from cell 7e0c5d59
    create_dummy_video(inference_video_path, frames=16, height=224, width=224, fps=10)
    print("Dummy inference video created.")
else:
    print(f"Dummy inference video already exists at: {inference_video_path}")

# 2. Load the trained VJEPA model from the latest checkpoint
# This assumes `vjepa_model` (from cell 428154c0) and `video_transform_instance`
# (from cell fade84b6) are already defined in the notebook's kernel state.

checkpoint_path = "checkpoint_epoch_10.pth" # Or your desired checkpoint

if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # Map to CPU as CUDA was not available during training based on previous output
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    vjepa_model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.")
else:
    print(f"Warning: Checkpoint '{checkpoint_path}' not found. Using randomly initialized model for inference.")
    print("Please ensure you have run the training cell (1bda4e42) to save a checkpoint.")

# 3. Get the device for inference
inference_device = torch.device('cpu') # Based on previous execution, CUDA is not available

# 4. Run inference
print(f"\nRunning inference on {inference_video_path} using device: {inference_device}")
video_features = run_inference(vjepa_model, inference_video_path, video_transform_instance, inference_device)

print("\nInference complete!")
print(f"Shape of extracted video features: {video_features.shape}")
print(f"Example features (first 5 values): {video_features[0, :5].tolist()}")