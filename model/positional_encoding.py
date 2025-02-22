"""
Positional Encoding implementation as described in 'Attention Is All You Need'.

This module implements the sinusoidal positional encoding that adds
position information to the input embeddings. The encoding uses
sine and cosine functions of different frequencies.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding layer.

    Implements the positional encoding using sine and cosine functions
    as described in Section 3.5 of the paper. The encoding allows the
    model to be aware of the sequence order without any recurrence.

    The encoding follows the formula:
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

    Args:
        d_model (int): Dimension of the model
        max_seq_length (int): Maximum sequence length to pre-compute

    Attributes:
        pe (Tensor): Pre-computed positional encoding matrix
            of shape (1, max_seq_length, d_model)

    Example:
        >>> pos_encoder = PositionalEncoding(d_model=512, max_seq_length=1000)
        >>> x = torch.randn(32, 100, 512)  # (batch_size, seq_len, d_model)
        >>> output = pos_encoder(x)  # Adds positional information
    """

    def __init__(
        self,
        d_model: int = 512,
        max_seq_length: int = 5000,
    ) -> None:
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_length

        # Initialize positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (won't be updated during training)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Input combined with positional encoding of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1)] * math.sqrt(self.d_model)
