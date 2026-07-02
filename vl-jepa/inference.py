import torchvision.transforms as transforms
import torch


def vljepa_inference(image_pil, text_input, vision_encoder, language_encoder, multi_modal_predictor, device, img_size, dummy_tokenizer):
    """
    Performs multi-modal inference with the VL-JEPA context encoders and predictor.

    Args:
        image_pil (PIL.Image): The input image in PIL format.
        text_input (str): The input text string.
        vision_encoder (nn.Module): The trained vision context encoder.
        language_encoder (nn.Module): The trained language context encoder.
        multi_modal_predictor (nn.Module): The trained multi-modal predictor.
        device (str): The device to run the inference on ('cuda' or 'cpu').
        img_size (int): The image size expected by the vision model.
        dummy_tokenizer: The tokenizer to use for text input. If None, a new DummyTokenizer is created.

    Returns:
        torch.Tensor: The predicted multi-modal embedding.
    """
    # Define image transformations (same as during training)
    inference_vision_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply image transformations
    transformed_image = inference_vision_transform(image_pil).unsqueeze(0).to(device)

    token_ids = dummy_tokenizer.tokenize(text_input).unsqueeze(0).to(device)

    # Set models to evaluation mode
    vision_encoder.eval()
    language_encoder.eval()
    multi_modal_predictor.eval()

    with torch.no_grad():
        # Get vision context embedding
        vision_embedding = vision_encoder(transformed_image)

        # Get language context embedding
        language_embedding = language_encoder(token_ids)

        # Predict multi-modal embedding
        predicted_embedding = multi_modal_predictor(vision_embedding, language_embedding)

    return predicted_embedding