import torch
from utils import update_ema_parameters

def train_model(epochs,vision_context_encoder,language_context_encoder,multi_modal_predictor,vision_target_encoder,language_target_encoder,vljepa_train_dataloader,ema_decay,loss_function,optimizer,device):
    for epoch in range(epochs):
        vision_context_encoder.train()
        language_context_encoder.train()
        multi_modal_predictor.train()
        total_loss = 0

        for batch_idx, (context_image, target_image, context_text, target_text) in enumerate(vljepa_train_dataloader):
            # 8. Move data to DEVICE
            context_image = context_image.to(device)
            target_image = target_image.to(device)
            context_text = context_text.to(device)
            target_text = target_text.to(device)

            optimizer.zero_grad()

            # 10. Perform the forward pass
            # a. Pass context_image through vision_context_encoder
            vision_context_embeddings = vision_context_encoder(context_image)
            # b. Pass context_text through language_context_encoder
            language_context_embeddings = language_context_encoder(context_text)
            # c. Pass vision_context_embeddings and language_context_embeddings through multi_modal_predictor
            predicted_embeddings = multi_modal_predictor(vision_context_embeddings, language_context_embeddings)

            # 11. Generate actual target embeddings (ground truth) without gradient tracking
            with torch.no_grad():
                # a. Pass target_image through vision_target_encoder
                actual_vision_target_embeddings = vision_target_encoder(target_image)
                # b. Pass target_text through language_target_encoder
                actual_language_target_embeddings = language_target_encoder(target_text)

            # 12. Calculate the combined loss
            # Note: For simplicity and based on the instruction to calculate loss between predicted_embeddings and *both* actual_target_embeddings,
            # we'll assume the multi_modal_predictor predicts a common embedding space, and the task is to predict both vision and language targets from the combined context.
            # This might be simplified or made more complex depending on the exact VL-JEPA variant.
            # Here, we will assume predicted_embeddings tries to match actual_vision_target_embeddings and actual_language_target_embeddings separately.

            loss_vision = loss_function(predicted_embeddings, actual_vision_target_embeddings)
            loss_language = loss_function(predicted_embeddings, actual_language_target_embeddings)
            total_loss_batch = loss_vision + loss_language

            # 13. Perform the backward pass
            total_loss_batch.backward()
            # 14. Update the model weights
            optimizer.step()

            # 15. Update vision_target_encoder with EMA
            update_ema_parameters(vision_target_encoder, vision_context_encoder, ema_decay)
            # 16. Update language_target_encoder with EMA
            update_ema_parameters(language_target_encoder, language_context_encoder, ema_decay)

            total_loss += total_loss_batch.item()

            # 17. Accumulate and print the training loss periodically
            if batch_idx % 100 == 0: # Print loss every 100 batches
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(vljepa_train_dataloader)}, Loss: {total_loss_batch.item():.4f}")

        avg_loss = total_loss / len(vljepa_train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}\n")

        print("VL-JEPA training complete.")