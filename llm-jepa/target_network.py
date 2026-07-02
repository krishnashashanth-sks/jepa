import torch
import torch.nn as nn
from encoder import Encoder

class TargetNetwork(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        max_len: int,
        dropout: float = 0.1
    ):
        super().__init__()
        # Initialize an instance of the Encoder class
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Passes input through the internal Encoder instance."""
        return self.encoder(src, src_mask)

    @torch.no_grad()
    def update_parameters(self, primary_encoder: nn.Module, momentum: float):
        """Updates target network parameters using a momentum-based mechanism."""
        for target_param, encoder_param in zip(self.encoder.parameters(), primary_encoder.parameters()):
            target_param.data = target_param.data * momentum + encoder_param.data * (1. - momentum)
