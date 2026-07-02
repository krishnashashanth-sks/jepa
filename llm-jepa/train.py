import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from evaluate import evaluate_step
from losses import LLM_JEPALoss

def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    scheduler,
    device: torch.device,
    momentum_rate: float
) -> float:
    """
    Performs a single training step over all batches in the dataloader.

    Args:
        model: The LLM_JEPA model.
        dataloader: DataLoader for the training data.
        optimizer: Optimizer for the online encoder and predictor.
        loss_fn: The loss function (LLM_JEPALoss).
        scheduler: Learning rate scheduler.
        device: The device to run the training on (e.g., 'cuda' or 'cpu').
        momentum_rate: Momentum rate for the target network update.

    Returns:
        The average loss for the training step.
    """
    model.train() # Set the model to training mode
    total_loss = 0.0
    num_batches = 0

    for i, batch in enumerate(dataloader):
        # Move all relevant tensors within the batch to the specified device
        encoder_input_ids = batch["encoder_input_ids"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        target_input_ids = batch["target_input_ids"].to(device)
        target_attention_mask = batch["target_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad() # Zero out the gradients

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

        # Perform a backward pass
        loss.backward()

        # Update the weights of the online encoder and predictor
        optimizer.step()

        # Update the target network's parameters using momentum
        model.target_encoder.update_parameters(model.online_encoder, momentum_rate)

        # Update the learning rate scheduler
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if i % 100 == 0: # Print loss every 100 batches
            print(f"Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches

def train_model(n_epochs,model,train_dataloader,val_dataloader,optimizer,scheduler,momentum_rate,device):
    # Initialize lists to store losses for plotting/analysis
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Create a directory to save model checkpoints
    checkpoint_dir = "llm_jepa_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        # Training step
        avg_train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=LLM_JEPALoss(), # Instantiate loss function here if not a global instance
            scheduler=scheduler,
            device=device,
            momentum_rate=momentum_rate
        )
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        # Evaluation step
        avg_val_loss = evaluate_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=LLM_JEPALoss(), # Instantiate loss function here if not a global instance
            device=device
        )
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")

        # Save the model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(checkpoint_dir, f"llm_jepa_best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation loss {best_val_loss:.4f} to {model_save_path}")
