import torch
import torch.nn as nn
from types import Any

def inference_llm_jepa(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    device: torch.device,
    block_size: int
) -> torch.Tensor:
    """
    Performs inference with the LLM-JEPA model to extract contextualized embeddings
    for a given input text.

    Args:
        model: The trained LLM_JEPA model.
        tokenizer: The tokenizer used for data preparation.
        text: The input text string.
        device: The device to run inference on (e.g., 'cuda' or 'cpu').
        block_size: The maximum sequence length used during training.

    Returns:
        A tensor containing the contextualized embeddings from the online_encoder.
        Shape: (1, sequence_length, d_model)
    """
    model.eval() # Set the model to evaluation mode

    # Tokenize the input text
    tokenized_input = tokenizer(text, truncation=True, max_length=block_size, return_tensors="pt")

    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    with torch.no_grad(): # Disable gradient calculations for inference
        # Pass through the online_encoder to get contextualized embeddings
        # Note: For inference, we typically don't mask the input if we want
        # embeddings of the entire sequence. The encoder_input_ids and
        # encoder_attention_mask are directly used here for the online encoder.
        contextualized_embeddings = model.online_encoder(
            src=input_ids,
            src_mask=attention_mask
        )

    return contextualized_embeddings
