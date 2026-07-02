def train_model(num_epochs,hjepa_model,data_loader,optimizer,hjepa_loss,scheduler,device):
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        hjepa_model.train() # Set model to training mode
        total_loss_epoch = 0.0

        for batch_idx, (context_view, target_view) in enumerate(data_loader):
            # Move data to device
            context_view = context_view.to(device)
            target_view = target_view.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            predicted_target_embedding, actual_target_embedding = hjepa_model(context_view, target_view)

            # Calculate loss
            loss = hjepa_loss(predicted_target_embedding, actual_target_embedding)
            total_loss_epoch += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0: # Log every 50 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")

        # Step the learning rate scheduler (if applicable, typically after each epoch)
        scheduler.step()

        avg_loss_epoch = total_loss_epoch / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {avg_loss_epoch:.4f}, Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("Training finished.")