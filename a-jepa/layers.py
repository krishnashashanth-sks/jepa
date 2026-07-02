import torch
import torch.nn as nn
import random
import torchaudio.transforms as T

class AudioContextEncoder(nn.Module):
  def __init__(self, input_channels, hidden_dims, output_dim, n_mels, time_frames):
    super(AudioContextEncoder, self).__init__()
    layers=[]
    in_channels=input_channels

    # Dynamically calculate the output shape after convolutional layers
    temp_h, temp_w = n_mels, time_frames

    for h_dim in hidden_dims:
      layers.append(nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1))
      layers.append(nn.ReLU())
      layers.append(nn.BatchNorm2d(h_dim)) # Changed to BatchNorm2d
      in_channels=h_dim

      # Update temporary dimensions for stride=2 convolution with padding=1
      # Formula: floor((input_size - kernel_size + 2 * padding) / stride) + 1
      temp_h = (temp_h - 3 + 2 * 1) // 2 + 1
      temp_w = (temp_w - 3 + 2 * 1) // 2 + 1

    self.conv_layers=nn.Sequential(*layers)

    self.fc_input_dim = in_channels * temp_h * temp_w # Use the final calculated dimensions
    self.fc=nn.Linear(self.fc_input_dim, output_dim)

  def forward(self, x):
    x=self.conv_layers(x)
    x=torch.flatten(x, 1)
    return self.fc(x)

class AudioTargetEncoder(nn.Module):
  def __init__(self, input_channels, hidden_dims, output_dim, n_mels, time_frames):
    super(AudioTargetEncoder, self).__init__()
    layers=[]
    in_channels=input_channels

    temp_h, temp_w = n_mels, time_frames

    for h_dim in hidden_dims:
      layers.append(nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1))
      layers.append(nn.ReLU())
      layers.append(nn.BatchNorm2d(h_dim))
      in_channels=h_dim

      temp_h = (temp_h - 3 + 2 * 1) // 2 + 1
      temp_w = (temp_w - 3 + 2 * 1) // 2 + 1

    self.conv_layers=nn.Sequential(*layers)

    self.fc_input_dim = in_channels * temp_h * temp_w
    self.fc=nn.Linear(self.fc_input_dim, output_dim)

  def forward(self, x):
    x=self.conv_layers(x)
    x=torch.flatten(x, 1)
    return self.fc(x)

class PredictorNetwork(nn.Module):
    def __init__(self, context_embedding_dim, target_embedding_dim, hidden_dims):
        # Fix: Class name in super() must match the class definition
        super(PredictorNetwork, self).__init__()

        layers = []
        curr_dim = context_embedding_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            # Advanced Tip: LayerNorm is often more stable than BatchNorm for Predictors
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU()) # GELU is standard for Transformer-based JEPAs
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, target_embedding_dim))
        self.predictor = nn.Sequential(*layers)

    def forward(self, context_embedding):
        return self.predictor(context_embedding)

class MelSpectrogramExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=64, top_db=80):
        super().__init__()
        # MelSpectrogram is usually calculated on 'power' (magnitude squared)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            power=2.0
        )
        # top_db=80 is standard for keeping relevant audio info while clipping noise
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=top_db)

    def forward(self, waveform):
        # The DataLoader will handle batching, so waveform is already (Batch, Time) or (Time)
        # torchaudio.transforms.MelSpectrogram can handle both (Time) and (Batch, Time)

        # 1. Generate Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # 2. Convert to Decibel Scale (Log scale)
        # This is vital because human hearing (and CNNs) perceive audio logarithmically
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # 3. Normalization (Advanced Step)
        # Deep learning models perform better when inputs are roughly Mean 0, Std 1
        # We normalize based on the top_db range
        mel_spec_db = (mel_spec_db + 40) / 40

        # Add channel dimension: (B, C, Freq, Time) where C=1. unsqueeze(-3) places it correctly.
        return mel_spec_db.unsqueeze(-3)
    
class RandomAudioMasking(nn.Module):
    def __init__(self, time_mask_rate, freq_mask_rate, max_time_mask_width, max_freq_mask_height, mask_value=0.0):
        super(RandomAudioMasking, self).__init__()
        self.time_mask_rate = time_mask_rate
        self.freq_mask_rate = freq_mask_rate
        self.max_time_mask_width = max_time_mask_width
        self.max_freq_mask_height = max_freq_mask_height
        self.mask_value = mask_value

    def forward(self, x):
        # x shape: (batch_size, channels, n_mels, time_frames)
        batch_size, channels, n_mels, time_frames = x.shape
        masked_x = x.clone()

        for i in range(batch_size):
            # 1. Time Masking
            if random.random() < self.time_mask_rate:
                mask_width = random.randint(1, min(self.max_time_mask_width, time_frames))
                start_time = random.randint(0, time_frames - mask_width)
                masked_x[i, :, :, start_time : start_time + mask_width] = self.mask_value

            # 2. Frequency Masking
            if random.random() < self.freq_mask_rate:
                mask_height = random.randint(1, min(self.max_freq_mask_height, n_mels))
                start_freq = random.randint(0, n_mels - mask_height)
                masked_x[i, :, start_freq : start_freq + mask_height, :] = self.mask_value

        return masked_x