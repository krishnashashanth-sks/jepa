from sklearn.manifold import TSNE # Ensure scikit-learn is installed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from layers import *
from transform import HJEPADatasetTransform
from losses import HJEPALoss
from model import HJEPAModel
from train import train
from dataset import DummyDataset
from torch.utils.data import DataLoader
from train import train_model

IMAGE_SIZE = 64
gb_kernel_size = 7 # Should be odd and positive

weak_augmentation_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

strong_augmentation_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=gb_kernel_size),
    transforms.ToTensor(),
])

# Instantiate the masking strategy
random_patch_masking = RandomPatchMasking(num_patches=3, min_patch_size_ratio=0.1, max_patch_size_ratio=0.3)

# Instantiate the combined data transformation class
hjepa_transform = HJEPADatasetTransform(
    weak_augment=weak_augmentation_pipeline,
    strong_augment=strong_augmentation_pipeline,
    patch_masking=random_patch_masking
)
NUM_SAMPLES=1000
BATCH_SIZE=32
NUM_WORKERS = 2 # Set to 0 for simpler debugging if issues arise

# Instantiate DummyDataset
dummy_dataset = DummyDataset(num_samples=NUM_SAMPLES, transform=hjepa_transform)
# Instantiate DataLoader
data_loader = DataLoader(
    dummy_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True # For faster data transfer to GPU
)
EMBEDDING_DIM=256
INPUT_CHANNELS=3
ENCODER_HIDDEN_DIMS_REVISED = [32, 64, 128, 256] # This will lead to 4 conv layers, resulting in 4x4 feature map
FINAL_CONV_FEATURE_MAP_DIM = 4 # Based on 4 conv layers with stride 2 from 64x64 input
FC_INPUT_DIM_FOR_ENCODERS = ENCODER_HIDDEN_DIMS_REVISED[-1] * (FINAL_CONV_FEATURE_MAP_DIM * FINAL_CONV_FEATURE_MAP_DIM)

# Instantiate encoders with revised hidden dims
context_encoder = ContextEncoder(INPUT_CHANNELS, ENCODER_HIDDEN_DIMS_REVISED, EMBEDDING_DIM)
target_encoder = TargetEncoder(INPUT_CHANNELS, ENCODER_HIDDEN_DIMS_REVISED, EMBEDDING_DIM)

# Instantiate predictor network
predictor_network = PredictorNetwork(EMBEDDING_DIM, EMBEDDING_DIM, [EMBEDDING_DIM * 2, EMBEDDING_DIM])

# Instantiate the loss function
hjepa_loss = HJEPALoss(similarity_loss_weight=0.1)

    
hjepa_model = HJEPAModel(context_encoder, target_encoder, predictor_network)

# --- Optimizer and Scheduler Hyperparameters ---
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6 # Common for regularization
NUM_EPOCHS = 10 # Placeholder for the full training loop

# --- Define Optimizer ---
# Optimize all parameters of the HJEPAModel
optimizer = optim.AdamW(hjepa_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- Define Learning Rate Scheduler (Optional) ---
# Using CosineAnnealingLR for smooth learning rate decay
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

print(f"Optimizer ({type(optimizer).__name__}) initialized with learning rate {LEARNING_RATE}.")
print(f"Learning rate scheduler ({type(scheduler).__name__}) initialized.")
# --- Training Loop Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
hjepa_model.to(device)

train_model(NUM_EPOCHS,hjepa_model,data_loader,optimizer,hjepa_loss,scheduler,device)

# --- Visualization Parameters ---
num_samples_to_display = 3

print(f"Displaying {num_samples_to_display} samples of original, context, and target views...")

plt.figure(figsize=(12, num_samples_to_display * 4))

for i in range(num_samples_to_display):
    # a. Generate a new random PIL.Image
    img_width, img_height, img_channels = IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNELS
    random_array = bytearray([random.randint(0, 255) for _ in range(img_width * img_height * img_channels)])
    original_image = Image.frombytes('RGB', (img_width, img_height), bytes(random_array))

    # c. Apply the hjepa_transform
    context_view_tensor, target_view_tensor = hjepa_transform(original_image)

    # d. Convert tensors to numpy arrays for matplotlib display (CHW to HWC)
    context_view_np = context_view_tensor.permute(1, 2, 0).cpu().numpy()
    target_view_np = target_view_tensor.permute(1, 2, 0).cpu().numpy()

    # --- Plotting ---
    # Original Image
    plt.subplot(num_samples_to_display, 3, i * 3 + 1)
    plt.imshow(original_image) # PIL Image is directly displayable
    plt.title('Original Image')
    plt.axis('off')

    # Context View (Masked)
    plt.subplot(num_samples_to_display, 3, i * 3 + 2)
    plt.imshow(context_view_np)
    plt.title('Context View (Masked)')
    plt.axis('off')

    # Target View (Augmented)
    plt.subplot(num_samples_to_display, 3, i * 3 + 3)
    plt.imshow(target_view_np)
    plt.title('Target View (Augmented)')
    plt.axis('off')

plt.tight_layout()
plt.show()

print("Visualization complete.")

# --- Parameters for Embedding Visualization ---
num_embeddings_to_visualize = 200 # Number of samples to extract embeddings for

print(f"Extracting embeddings for {num_embeddings_to_visualize} samples...")

# 1. Extract Embeddings
hjepa_model.eval() # Set model to evaluation mode
all_embeddings = []

# We'll use the dummy_dataset directly for this, creating a fresh instance to avoid iterating a consumed DataLoader
dummy_dataset_for_viz = DummyDataset(num_samples=num_embeddings_to_visualize, transform=hjepa_transform)
data_loader_for_viz = DataLoader(
    dummy_dataset_for_viz,
    batch_size=BATCH_SIZE, # Use the same batch size as training
    shuffle=False, # No need to shuffle for visualization
    num_workers=NUM_WORKERS,
    pin_memory=True
)

with torch.no_grad(): # Disable gradient calculations for inference
    for batch_idx, (context_view, target_view) in enumerate(data_loader_for_viz):
        context_view = context_view.to(device)
        # We can extract embeddings from either context_encoder or target_encoder
        # Let's use context_encoder for this example
        embedding = hjepa_model.context_encoder(context_view)
        all_embeddings.append(embedding.cpu().numpy())

        if (batch_idx + 1) * BATCH_SIZE >= num_embeddings_to_visualize:
            break # Stop once enough embeddings are collected

all_embeddings = np.concatenate(all_embeddings, axis=0)
# Trim to exact number if last batch overshot
if all_embeddings.shape[0] > num_embeddings_to_visualize:
    all_embeddings = all_embeddings[:num_embeddings_to_visualize]

print(f"Extracted {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}.")

# 2. Perform Dimensionality Reduction using t-SNE
print("Applying t-SNE for dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000) # Common t-SNE parameters
embeddings_2d = tsne.fit_transform(all_embeddings)

print("t-SNE completed. Plotting results...")

# 3. Plot Embeddings
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10) # Removed cmap as there are no labels for dummy data
plt.title('2D t-SNE Projection of Learned Embeddings (Dummy Data)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
