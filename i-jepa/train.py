import torch
from utils import update_ema_parameters

def train(epochs,context_encoder_vit,predictor_vit,ijepa_train_dataloader,ema_decay,optimizer,target_encoder_vit,loss_function,device):
    for epoch in range(epochs):
        context_encoder_vit.train() # Set context encoder to training mode
        predictor_vit.train()     # Set predictor to training mode
        total_loss = 0

        for batch_idx, (context_view, target_view) in enumerate(ijepa_train_dataloader):
            context_view = context_view.to(device)
            target_view = target_view.to(device)

            optimizer.zero_grad()

            # Forward pass through context encoder and predictor
            context_embeddings = context_encoder_vit(context_view)
            predicted_target_embeddings = predictor_vit(context_embeddings)

            # Generate actual target embeddings with target encoder (no grad)
            with torch.no_grad():
                actual_target_embeddings = target_encoder_vit(target_view)

            # Calculate loss
            loss = loss_function(predicted_target_embeddings, actual_target_embeddings)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update target encoder with EMA
            update_ema_parameters(target_encoder_vit, context_encoder_vit, ema_decay)

            total_loss += loss.item()

            if batch_idx % 100 == 0: # Print loss every 100 batches
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(ijepa_train_dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(ijepa_train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}\n")

    print("Training complete.")