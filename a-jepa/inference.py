import torch

def ajepa_inference(model, mel_extractor, waveform, device):
    """
    Performs inference with the A-JEPA model to extract embeddings from a raw audio waveform.

    Args:
        model (nn.Module): The trained AJEPAModel instance.
        mel_extractor (nn.Module): The MelSpectrogramExtractor instance.
        waveform (torch.Tensor): A raw audio waveform tensor (1D, e.g., [Time]).
        device (torch.device): The device (cpu or cuda) to perform inference on.

    Returns:
        numpy.ndarray: The extracted context embedding as a NumPy array.
    """
    model.eval()  # Set the model to evaluation mode
    mel_extractor.eval() # Set the extractor to evaluation mode

    with torch.no_grad():  # Disable gradient calculations
        # Ensure waveform is on the correct device and has a batch dimension
        # MelSpectrogramExtractor expects (Batch, Time) or (Time)
        # If given (1, Time), it will produce (1, Channels, Freq, Time)
        waveform_batch = waveform.to(device).unsqueeze(0)
        
        # Extract mel-spectrogram
        mel_spec = mel_extractor(waveform_batch)
        
        # Extract context embedding from the model's audio_context_encoder
        context_embedding = model.audio_context_encoder(mel_spec)
        
    return context_embedding.cpu().numpy()