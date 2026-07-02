import torch.nn as nn
import torch.nn.functional as F

class AJEPALoss(nn.Module):
    def __init__(self, similarity_loss_weight=0.1):
        super(AJEPALoss, self).__init__()
        self.similarity_loss_weight = similarity_loss_weight
        # Can add other loss components or weights here if needed

    def forward(self, predicted_target_embedding, actual_target_embedding):
        # 1. Primary Prediction Loss (MSE)
        prediction_loss_mse = F.mse_loss(predicted_target_embedding, actual_target_embedding)

        # 2. Similarity-Based Loss (Cosine Similarity)
        # Ensure embeddings are normalized before computing cosine similarity
        predicted_norm = F.normalize(predicted_target_embedding, p=2, dim=-1)
        actual_norm = F.normalize(actual_target_embedding, p=2, dim=-1)

        # Cosine similarity ranges from -1 (opposite) to 1 (same)
        # We want to maximize similarity, so we minimize (1 - cos_sim) or (-cos_sim)
        cosine_similarity = F.cosine_similarity(predicted_norm, actual_norm, dim=-1).mean()
        similarity_loss = (1 - cosine_similarity) # Minimize (1 - cos_sim) to maximize cos_sim

        # 3. Combine Losses
        total_loss = prediction_loss_mse + (self.similarity_loss_weight * similarity_loss)

        return total_loss

