import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> float:
    """
    Performs a single evaluation step over all batches in the dataloader.

    Args:
        model: The LLM_JEPA model.
        dataloader: DataLoader for the validation data.
        loss_fn: The loss function (LLM_JEPALoss).
        device: The device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        The average loss for the evaluation step.
    """
    model.eval() # Set the model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(): # Disable gradient calculations
        for i, batch in enumerate(dataloader):
            # Move all relevant tensors within the batch to the specified device
            encoder_input_ids = batch["encoder_input_ids"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)
            target_input_ids = batch["target_input_ids"].to(device)
            target_attention_mask = batch["target_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Perform a forward pass
            predicted_embeddings, target_embeddings = model(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                target_input_ids=target_input_ids,
                target_attention_mask=target_attention_mask,
                labels=labels
            )

            # Calculate the loss
            loss = loss_fn(predicted_embeddings, target_embeddings)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches