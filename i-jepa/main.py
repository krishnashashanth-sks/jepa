from train import train
import torch
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from encoder import VisionTransformerEncoder
from predictor import Predictor
from mask import IJEPA_Masking
from dataset import IJEPA_Dataset

IMG_SIZE = 224 # Standard image size for ViT
PATCH_SIZE = 16 # Standard patch size
IN_CHANNELS = 3 # RGB images
EMBED_DIM = 768 # Typical embedding dimension for ViT-base
NUM_LAYERS = 12 # Number of transformer blocks
NUM_HEADS = 12 # Number of attention heads
MLP_RATIO = 4. # MLP hidden dimension factor
DROPOUT = 0.1

# 6. Instantiate the VisionTransformerEncoder for the ContextEncoder
context_encoder_vit = VisionTransformerEncoder(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=IN_CHANNELS,
    embed_dim=EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    dropout=DROPOUT,
    include_pooling=True
)

predictor_vit = Predictor(input_dim=CONTEXT_ENCODER_OUTPUT_DIM, hidden_dim=PREDICTOR_HIDDEN_DIM, output_dim=PREDICTOR_OUTPUT_DIM)


transform_cifar_base = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize images to the expected input size for ViT
])

train_dataset_raw = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_cifar_base) # Use base transform
test_dataset_raw = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_cifar_base) # Use base transform
# Define the transformation pipeline for IJEPA_Dataset (ToTensor and Normalize)
ijepa_post_resize_transform = transforms.Compose([
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

# Instantiate the masking strategy
NUM_CONTEXT_BLOCKS = 16 # Example: Number of patches for context
NUM_TARGET_BLOCKS = 4   # Example: Number of patches for target

ijepa_masking = IJEPA_Masking(IMG_SIZE, PATCH_SIZE, NUM_CONTEXT_BLOCKS, NUM_TARGET_BLOCKS)

# Create the IJEPA Dataset, passing the new transform pipeline
ijepa_train_dataset = IJEPA_Dataset(train_dataset_raw, ijepa_post_resize_transform, ijepa_masking)
ijepa_test_dataset = IJEPA_Dataset(test_dataset_raw, ijepa_post_resize_transform, ijepa_masking)


BATCH_SIZE = 64

# Create DataLoaders for training and testing
ijepa_train_dataloader = torch.utils.data.DataLoader(
    ijepa_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
ijepa_test_dataloader = torch.utils.data.DataLoader(
    ijepa_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

epochs = 1
EMA_DECAY = 0.996 # Typical decay rate for EMA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for target_param, context_param in zip(target_encoder_vit.parameters(), context_encoder_vit.parameters()):
        target_param.data.copy_(context_param.data)
    
LEARNING_RATE = 1e-4

optimizer = torch.optim.AdamW(list(context_encoder_vit.parameters()) + list(predictor_vit.parameters()), lr=LEARNING_RATE)

loss_function = nn.MSELoss()

train(epochs,context_encoder_vit,predictor_vit,ijepa_train_dataloader,optimizer,target_encoder_vit,loss_function,device)