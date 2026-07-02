def train(num_epochs,ajepa_model,audio_data_loader,optimizer,ajepa_loss,scheduler,device):
      # List to store average loss per epoch for later visualization
    train_losses = []

    # Training loop
    print("Starting A-JEPA training...")
    for epoch in range(num_epochs):
        ajepa_model.train() # Set model to training mode
        total_loss_epoch = 0.0

        for batch_idx, (context_audio_view, target_audio_view) in enumerate(audio_data_loader):
            # Move data to device
            context_audio_view = context_audio_view.to(device)
            target_audio_view = target_audio_view.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predicted_target_embedding, actual_target_embedding = ajepa_model(context_audio_view, target_audio_view)

            # Calculate loss
            loss = ajepa_loss(predicted_target_embedding, actual_target_embedding)
            total_loss_epoch += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0: # Log every 50 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(audio_data_loader)}], Loss: {loss.item():.4f}")

        # Step the learning rate scheduler (typically after each epoch)
        scheduler.step()

        avg_loss_epoch = total_loss_epoch / len(audio_data_loader)
        train_losses.append(avg_loss_epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {avg_loss_epoch:.4f}, Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("A-JEPA training finished.")