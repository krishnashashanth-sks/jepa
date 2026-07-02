import torch.nn as nn
import torch

# Define the custom loss function for LLM-JEPA
class LLM_JEPALoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        """Computes the Mean Squared Error between predicted and target embeddings.

        Args:
            predicted_embeddings: The embeddings predicted by the online encoder and predictor.
            target_embeddings: The embeddings from the target network.

        Returns:
            The Mean Squared Error loss.
        """
        return self.mse_loss(predicted_embeddings, target_embeddings)
