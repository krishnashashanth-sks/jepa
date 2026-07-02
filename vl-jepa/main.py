import torchvision.transforms as transforms
from tokenizer import DummyTokenizer
from mask import IJEPA_Masking,TextMasking
import torchvision
from dataset import VLJEPA_Dataset
from language_encoder import LanguageTransformerEncoder
from vision_encoder import VisionTransformerEncoder
import torch
from predictor import MultiModalPredictor
from train import train_model
from inference import vljepa_inference

IMG_SIZE = 224
PATCH_SIZE = 16
NUM_CONTEXT_BLOCKS = 16
NUM_TARGET_BLOCKS = 4
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 77


# Create a dummy CIFAR10 dataset for VL-JEPA
transform_cifar_base = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])

train_dataset_raw = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_cifar_base)

# Define the transformation pipeline for VLJEPA_Dataset (ToTensor and Normalize)
vljepa_post_resize_transform = transforms.Compose([
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

# Create an instance of DummyTokenizer
dummy_tokenizer = DummyTokenizer(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN)

# Create an instance of IJEPA_Masking for vision
vision_masking_strategy = IJEPA_Masking(IMG_SIZE, PATCH_SIZE, NUM_CONTEXT_BLOCKS, NUM_TARGET_BLOCKS)

# Create an instance of TextMasking for language
NUM_CONTEXT_TOKENS = 20 # Example: Number of tokens for text context
NUM_TARGET_TOKENS = 5   # Example: Number of tokens for text target

text_masking_strategy = TextMasking(MAX_SEQ_LEN, NUM_CONTEXT_TOKENS, NUM_TARGET_TOKENS, mask_token_id=0)

# Instantiate the VLJEPA_Dataset
vljepa_train_dataset = VLJEPA_Dataset(
    base_dataset=train_dataset_raw,
    vision_transform=vljepa_post_resize_transform,
    dummy_tokenizer=dummy_tokenizer,
    vision_masking_strategy=vision_masking_strategy,
    text_masking_strategy=text_masking_strategy
)
BATCH_SIZE = 16

# Instantiate a dummy test dataset for DataLoader creation demonstration
dummy_test_dataset_raw = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                 download=True, transform=transform_cifar_base)

vljepa_test_dataset = VLJEPA_Dataset(
    base_dataset=dummy_test_dataset_raw,
    vision_transform=vljepa_post_resize_transform,
    dummy_tokenizer=dummy_tokenizer,
    vision_masking_strategy=vision_masking_strategy,
    text_masking_strategy=text_masking_strategy
)

# Create DataLoaders for training and testing
vljepa_train_dataloader = torch.utils.data.DataLoader(
    vljepa_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
vljepa_test_dataloader = torch.utils.data.DataLoader(
    vljepa_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
# Global constants for model configuration
IMG_SIZE = 224 # Standard image size for ViT
PATCH_SIZE = 16 # Standard patch size
IN_CHANNELS = 3 # RGB images
EMBED_DIM = 768 # Typical embedding dimension for ViT-base
NUM_LAYERS = 12 # Number of transformer blocks
NUM_HEADS = 12 # Number of attention heads
MLP_RATIO = 4. # MLP hidden dimension factor
DROPOUT = 0.1


#Language encoder parameters
VOCAB_SIZE = 10000 # Example vocabulary size
MAX_SEQ_LEN = 77  # Max sequence length for text (e.g., for CLIP)
LANGUAGE_EMBED_DIM = EMBED_DIM # Use the same embed_dim for consistency across modalities

# Instantiate the LanguageTransformerEncoder for the language context encoder
language_context_encoder = LanguageTransformerEncoder(
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    embed_dim=LANGUAGE_EMBED_DIM,
    num_layers=NUM_LAYERS, # Reuse NUM_LAYERS from ViT
    num_heads=NUM_HEADS,   # Reuse NUM_HEADS from ViT
    mlp_ratio=MLP_RATIO,   # Reuse MLP_RATIO from ViT
    dropout=DROPOUT,       # Reuse DROPOUT from ViT
    include_pooling=True
)
print(f"Language Context Encoder (Transformer) initialized: {language_context_encoder}\n")

# Instantiate an identical LanguageTransformerEncoder for the language target encoder
language_target_encoder = LanguageTransformerEncoder(
    vocab_size=VOCAB_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    embed_dim=LANGUAGE_EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    dropout=DROPOUT,
    include_pooling=True
)
print(f"Language Target Encoder (Transformer) initialized (identical to Language Context Encoder): {language_target_encoder}\n")

#vision encoder parameters
VISION_EMBED_DIM = EMBED_DIM # Use the same embed_dim as previously defined for I-JEPA ViT

# Instantiate the VisionTransformerEncoder for the vision context encoder
vision_context_encoder = VisionTransformerEncoder(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=IN_CHANNELS,
    embed_dim=VISION_EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    dropout=DROPOUT,
    include_pooling=True
)
print(f"Vision Context Encoder (ViT) initialized: {vision_context_encoder}\n")

# Instantiate an identical VisionTransformerEncoder for the vision target encoder
vision_target_encoder = VisionTransformerEncoder(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=IN_CHANNELS,
    embed_dim=VISION_EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    dropout=DROPOUT,
    include_pooling=True
)
MM_PREDICTOR_HIDDEN_DIM = EMBED_DIM * 2 # Example hidden dimension, can be tuned
MM_PREDICTOR_OUTPUT_DIM = EMBED_DIM # Assuming it predicts an embedding of this size

multi_modal_predictor = MultiModalPredictor(
    vision_embed_dim=VISION_EMBED_DIM,
    language_embed_dim=LANGUAGE_EMBED_DIM,
    hidden_dim=MM_PREDICTOR_HIDDEN_DIM,
    output_dim=MM_PREDICTOR_OUTPUT_DIM
)

print(f"Vision Target Encoder (ViT) initialized (identical to Vision Context Encoder): {vision_target_encoder}\n")

loss_function = torch.nn.MSELoss()

LEARNING_RATE = 1e-4

optimizer = torch.optim.AdamW(
    list(vision_context_encoder.parameters()) +
    list(language_context_encoder.parameters()) +
    list(multi_modal_predictor.parameters()),
    lr=LEARNING_RATE
)

print(f"Optimizer (AdamW) initialized for vision and language context encoders and multi-modal predictor with learning rate {LEARNING_RATE}.")

EPOCHS = 1
EMA_DECAY = 0.996 # Typical decay rate for EMA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vision_context_encoder.to(DEVICE)
language_context_encoder.to(DEVICE)
multi_modal_predictor.to(DEVICE)
vision_target_encoder.to(DEVICE)
language_target_encoder.to(DEVICE)

train_model(EPOCHS,vision_context_encoder,language_context_encoder,multi_modal_predictor,vision_target_encoder,language_target_encoder,vljepa_train_dataloader,EMA_DECAY,loss_function,optimizer,DEVICE)

cifar10_test = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=None
)

# Get a sample image from the test set
sample_image_pil, _ = cifar10_test[0]

# Prepare dummy text input
sample_text_input = "A dummy caption for a test image."

# Perform multi-modal inference
multi_modal_output_embedding = vljepa_inference(
    image_pil=sample_image_pil,
    text_input=sample_text_input,
    vision_encoder=vision_context_encoder,
    language_encoder=language_context_encoder,
    multi_modal_predictor=multi_modal_predictor,
    device=DEVICE,
    dummy_tokenizer=dummy_tokenizer
)

print(f"\nSample image loaded. Dummy text input: '{sample_text_input}'")
print(f"Predicted Multi-modal embedding shape: {multi_modal_output_embedding.shape}")
print(f"Sample Multi-modal embedding (first 5 values): {multi_modal_output_embedding.flatten()[:5].tolist()}")