import torch

def run_inference(model, video_path, video_transform_instance, device):
    """
    Performs inference using the V-JEPA student encoder on a single video.

    Args:
        model: The VJEPA model instance (containing the student_encoder).
        video_path (str): Path to the input video file.
        video_transform_instance: An instance of the VideoTransform class.
        device (torch.device): The device to run inference on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: The learned feature representation of the video.
    """
    model.eval() # Set model to evaluation mode

    # Load and preprocess the video
    video_tensor = video_transform_instance(video_path)
    video_tensor = video_tensor.unsqueeze(0) # Add batch dimension (B, C, T, H, W)
    video_tensor = video_tensor.to(device)

    with torch.no_grad(): # Disable gradient calculations
        # Pass through the student encoder to get features
        features = model.student_encoder(video_tensor)

    return features
