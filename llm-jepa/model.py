import torch
import torch.nn as nn
from encoder import Encoder
from predictor import Predictor
from target_network import TargetNetwork
from types import Tuple

# Ensure Encoder, Predictor, and TargetNetwork classes are defined and accessible
# (Assuming they are defined in previous cells and are in the global scope)

class LLM_JEPA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        max_len: int,
        predictor_hidden_dim: int,
        predictor_num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # 2a. Instantiate the Encoder network (online_encoder)
        self.online_encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout
        )

        # 2b. Instantiate the Predictor network
        self.predictor = Predictor(
            d_model=d_model,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers
        )

        # 2c. Instantiate the TargetNetwork network (target_encoder)
        self.target_encoder = TargetNetwork(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout
        )

        # 2d. Initialize target_encoder's parameters to be identical to online_encoder's
        # This copies the state dict of the online_encoder's internal encoder to the target_encoder's internal encoder
        self.target_encoder.encoder.load_state_dict(self.online_encoder.state_dict())
        # Ensure target_encoder remains frozen during online_encoder training (except for momentum update)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        labels: torch.Tensor # This 'labels' tensor from data collator indicates masked positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Derive mask_indices from labels (where labels are not -100)
        # mask_indices will be a boolean tensor indicating the positions of masked tokens
        mask_indices = (labels != -100)

        # 3a. Pass encoder_input_ids and encoder_attention_mask through the online_encoder
        encoder_output_embeddings = self.online_encoder(encoder_input_ids, encoder_attention_mask)

        # 3b. Use mask_indices to extract the embeddings corresponding to the masked tokens for predictor input
        # We need to flatten the batch and sequence dimensions to apply the mask correctly, then unflatten
        batch_size, seq_len, d_model = encoder_output_embeddings.shape
        flattened_encoder_output = encoder_output_embeddings.view(-1, d_model)
        flattened_mask_indices = mask_indices.view(-1)

        # Extract only the masked embeddings that need to be predicted
        masked_encoder_embeddings = flattened_encoder_output[flattened_mask_indices]

        # 3c. Pass the extracted masked embeddings through the Predictor
        predicted_embeddings = self.predictor(masked_encoder_embeddings)

        # 3d. Pass target_input_ids and target_attention_mask through the target_encoder
        # Ensure target_encoder is in eval mode for inference
        with torch.no_grad():
            target_output_embeddings = self.target_encoder(target_input_ids, target_attention_mask)

        # 3e. Use mask_indices to extract the embeddings corresponding to the masked tokens from target_output_embeddings
        flattened_target_output = target_output_embeddings.view(-1, d_model)
        target_embeddings = flattened_target_output[flattened_mask_indices]

        # 3f. Return both predicted_embeddings and target_embeddings
        return predicted_embeddings, target_embeddings