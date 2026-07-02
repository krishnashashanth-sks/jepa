
#1.Import
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 2. Define PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input embeddings.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model).
        Returns:
            Tensor of shape (batch_size, sequence_length, d_model) with positional encodings added.
        """
        # Ensure positional encoding matches input sequence length
        return x + self.pe[:, :x.size(1)]

# 3. Define MultiHeadSelfAttention class
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)

        # 1. Linear projections to Q, K, V
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention (Q @ K^T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # The mask provided is (batch_size, sequence_length), representing padding.
            # We need to expand it for broadcasting to (batch_size, 1, 1, sequence_length)
            # so it can correctly mask the scores tensor of shape (batch_size, num_heads, seq_len, seq_len).
            mask = mask.unsqueeze(1).unsqueeze(2) # Expands mask to (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 3. Multiply with V and concatenate heads
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Final linear layer
        output = self.out_linear(output)
        return output

# 4. Define FeedForwardBlock class
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x) # Using GELU as an advanced activation function
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 5. Define TransformerEncoderBlock class
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention part
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output) # Residual connection
        x = self.norm1(x)

        # Feed-forward part
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output) # Residual connection
        x = self.norm2(x)
        return x

# 6. Define the main Encoder class
class Encoder(nn.Module):
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
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # 7. Define forward pass
        # Token embeddings
        x = self.token_embedding(src) * math.sqrt(self.token_embedding.embedding_dim)

        # Add positional embeddings
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through transformer encoder blocks
        for layer in self.layers:
            x = layer(x, src_mask)

        return x