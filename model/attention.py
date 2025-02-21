"""
Multi-Head Attention implementation as described in 'Attention Is All You Need'.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism as described in 'Attention Is All You Need'.

    This module performs multi-head attention by first projecting the inputs (Q, K, V)
    into multiple heads, then applying scaled dot-product attention in parallel, and
    finally concatenating and projecting the results.

    Args:
        num_of_heads (int): Number of attention heads
        d_model (int): Model's dimension, must be divisible by num_of_heads

    Attributes:
        num_of_heads (int): Number of attention heads
        d_model (int): Model's dimension
        d_k (int): Dimension of keys/queries in each head (d_model // num_of_heads)
        W_Q (nn.Linear): Query projection matrix
        W_K (nn.Linear): Key projection matrix
        W_V (nn.Linear): Value projection matrix
        W_O (nn.Linear): Output projection matrix
    """

    def __init__(
        self,
        num_of_heads: int,
        d_model: int,
    ) -> None:
        super(MultiHeadAttention, self).__init__()

        # Check if d_model is divisible by num_of_heads.
        assert d_model % num_of_heads == 0

        # Initialize params.
        self.num_of_heads = num_of_heads
        self.d_model = d_model
        self.d_k = d_model // num_of_heads

        # Initialize the W^Q, W^K, W^V matrices.
        # Note that we assume d_model = d_v * num_of_heads = d_k * num_of_heads.
        # Therefore, d_k = d_v.
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_head(self, x: Tensor) -> Tensor:
        """Split the last dimension of input into (num_of_heads, d_k) and transpose.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Reshaped tensor of shape (batch_size, num_of_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_of_heads, self.d_k).transpose(1, 2)

    def scaled_dot_product_attention(
        self, Q: Tensor, K: Tensor, V: Tensor, masks: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """Compute scaled dot-product attention as described in the paper.

        Args:
            Q (Tensor): Query tensor of shape (batch_size, num_of_heads, seq_len, d_k)
            K (Tensor): Key tensor of shape (batch_size, num_of_heads, seq_len, d_k)
            V (Tensor): Value tensor of shape (batch_size, num_of_heads, seq_len, d_k)
            apply_masks (bool): Whether to apply masks to the attention weights

        Returns:
            Tuple[Tensor, Tensor]:
                - Output tensor of shape (batch_size, num_of_heads, seq_len, d_k)
                - Attention weights of shape (batch_size, num_of_heads, seq_len, seq_len)
        """
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=Q.dtype, device=Q.device)
        )

        if masks is not None:
            # Apply masks to the attention weights.
            attn_scores = attn_scores.masked_fill(
                masks == 0,
                float("-inf"),
            )

        attn_weights = torch.softmax(
            attn_scores,
            -1,
        )

        return torch.matmul(attn_weights, V), attn_weights

    def concatenate_heads(self, x: Tensor) -> Tensor:
        """Transpose and reshape the input back to original dimensions.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_of_heads, seq_len, d_k)

        Returns:
            Tensor: Reshaped tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, masks: Tensor = None) -> Tensor:
        """Compute multi-head attention.

        Args:
            Q (Tensor): Query tensor of shape (batch_size, seq_len, d_model)
            K (Tensor): Key tensor of shape (batch_size, seq_len, d_model)
            V (Tensor): Value tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Get Q, K, V after apply W_Q, W_K, and W_V.
        Q = self.split_head(self.W_Q(Q))
        K = self.split_head(self.W_K(K))
        V = self.split_head(self.W_V(V))

        # Run through scaled dot-product attention.
        v, _ = self.scaled_dot_product_attention(Q, K, V, masks)

        # Combine heads.
        combined_v = self.concatenate_heads(v)

        return self.W_O(combined_v)
