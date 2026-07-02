import torch
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from torchvision.transforms import Compose
from dataset import DummyAudioDataset
from train import train
from layers import MelSpectrogramExtractor,RandomAudioMasking,AudioContextEncoder,AudioTargetEncoder,PredictorNetwork
from model import AJEPAModel
from losses import AJEPALoss
from transform import AJEPADatasetTransform
from inference import ajepa_inference

# --- Constants for DummyAudioDataset and MelSpectrogramExtractor ---
SAMPLE_RATE = 16000 # Hz
DURATION = 2 # seconds
NUM_SAMPLES = 100

# MelSpectrogramExtractor parameters
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64 # This will be the height of our spectrograms
TOP_DB = 80

# RandomAudioMasking parameters
TIME_MASK_RATE = 0.5
FREQ_MASK_RATE = 0.5
MAX_TIME_MASK_WIDTH = 50 # Example: 50 time frames
MAX_FREQ_MASK_HEIGHT = 10 # Example: 10 mel bins
mel_spectrogram_extractor=MelSpectrogramExtractor(
    sample_rate=SAMPLE_RATE,n_fft=N_FFT,hop_length=HOP_LENGTH,n_mels=N_MELS,top_db=TOP_DB
)
weak_augmentation_pipeline=Compose([
    T.FrequencyMasking(freq_mask_param=5),
    T.TimeMasking(time_mask_param=10)
])
strong_augmentation_pipeline=Compose([
    T.FrequencyMasking(freq_mask_param=15),
    T.TimeMasking(time_mask_param=30)
])
random_audio_masking=RandomAudioMasking(
    time_mask_rate=TIME_MASK_RATE,
    freq_mask_rate=FREQ_MASK_RATE,
    max_time_mask_width=MAX_TIME_MASK_WIDTH,
    max_freq_mask_height=MAX_FREQ_MASK_HEIGHT,
    mask_value=0.0
)
ajepa_transform=AJEPADatasetTransform(
    mel_spectrogram_extractor=mel_spectrogram_extractor,
    weak_augmentation=weak_augmentation_pipeline,
    strong_augmentation=strong_augmentation_pipeline,
    audio_masking=random_audio_masking
)

# --- Instantiate DummyAudioDataset ---
dummy_audio_dataset = DummyAudioDataset(
    num_samples=NUM_SAMPLES,
    sample_rate=SAMPLE_RATE,
    duration=DURATION,
    transform=ajepa_transform
)

# --- Define DataLoader parameters ---
BATCH_SIZE = 16
NUM_WORKERS = 2 # Set to 0 for simpler debugging if issues arise

# --- Create DataLoader ---
audio_data_loader = DataLoader(
    dummy_audio_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True # For faster data transfer to GPU
)

import torch.nn as nn
TIME_FRAMES = (SAMPLE_RATE * DURATION - N_FFT) // HOP_LENGTH + 1
print(f"Calculated TIME_FRAMES: {TIME_FRAMES}")
# --- Define Hyperparameters for A-JEPA Modules ---
INPUT_CHANNELS = 1 # For mono audio mel-spectrograms
ENCODER_HIDDEN_DIMS = [32, 64, 128, 256] # Example convolutional layer hidden dimensions
EMBEDDING_DIM = 256 # Output dimension for encoders and predictor input/output

# --- Instantiate A-JEPA Core Modules ---
# N_MELS and TIME_FRAMES are retrieved from kernel variables automatically
audio_context_encoder = AudioContextEncoder(INPUT_CHANNELS, ENCODER_HIDDEN_DIMS, EMBEDDING_DIM, n_mels=N_MELS, time_frames=TIME_FRAMES)
audio_target_encoder = AudioTargetEncoder(INPUT_CHANNELS, ENCODER_HIDDEN_DIMS, EMBEDDING_DIM, n_mels=N_MELS, time_frames=TIME_FRAMES)

# Predictor network hidden dimensions example
PREDICTOR_HIDDEN_DIMS = [EMBEDDING_DIM * 2, EMBEDDING_DIM]
predictor_network = PredictorNetwork(EMBEDDING_DIM, EMBEDDING_DIM, PREDICTOR_HIDDEN_DIMS)

ajepa_model = AJEPAModel(audio_context_encoder, audio_target_encoder, predictor_network)

ajepa_loss = AJEPALoss(similarity_loss_weight=0.1)

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Optimizer and Scheduler Hyperparameters ---
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6 # Common for regularization
NUM_EPOCHS = 50 # Example number of epochs for training

# --- Define Optimizer ---
# Optimize all parameters of the AJEPAModel
optimizer = optim.AdamW(ajepa_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- Define Learning Rate Scheduler (Optional) ---
# Using CosineAnnealingLR for smooth learning rate decay
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to device
ajepa_model.to(device)

train(NUM_EPOCHS,ajepa_model,audio_data_loader,optimizer,ajepa_loss,scheduler,device)

dummy_waveform = torch.randn(SAMPLE_RATE * DURATION)
print(f"Dummy waveform created with shape: {dummy_waveform.shape}")

# 2. Perform inference
# Ensure ajepa_model and mel_spectrogram_extractor are instantiated from previous steps
try:
    inferred_embedding = ajepa_inference(ajepa_model, mel_spectrogram_extractor, dummy_waveform, device)
    print(f"Inferred embedding shape: {inferred_embedding.shape}")
    print(f"Sample of inferred embedding (first 5 elements): {inferred_embedding[0, :5]}")
except Exception as e:
    print(f"Error during inference: {e}")