
import torch
import torchvision.transforms as transforms
from main import IMG_SIZE

def ijepa_inference(image_pil, context_encoder, device, img_size=IMG_SIZE):
    """
    Performs inference with the I-JEPA context encoder.

    Args:
        image_pil (PIL.Image): The input image in PIL format.
        context_encoder (nn.Module): The trained context encoder (VisionTransformerEncoder).
        device (str): The device to run the inference on ('cuda' or 'cpu').
        img_size (int): The image size expected by the model.

    Returns:
        torch.Tensor: The learned embedding of the image.
    """
    # Define the same transformations used for context views during training
    inference_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transformations
    transformed_image = inference_transform(image_pil)

    # Add batch dimension and move to device
    transformed_image = transformed_image.unsqueeze(0).to(device)

    # Set encoder to evaluation mode
    context_encoder.eval()

    with torch.no_grad():
        embedding = context_encoder(transformed_image)

    return embedding
