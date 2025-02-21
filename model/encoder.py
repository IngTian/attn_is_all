"""
Transformer Encoder implementation as described in 'Attention Is All You Need'.

This module implements the Encoder part of the Transformer architecture, consisting of
multiple EncoderLayers stacked together. Each EncoderLayer contains a multi-head
self-attention mechanism followed by a position-wise feed-forward network.

Input Shape:
    - Input tensor: (batch_size, sequence_length, d_model)
    - Attention mask: (batch_size, 1, sequence_length, sequence_length)
"""

import torch
from torch import Tensor
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """A single layer of the Transformer encoder.

    Each encoder layer consists of:
    1. Multi-head self-attention mechanism
    2. Add & Norm layer (residual connection + layer normalization)
    3. Position-wise feed-forward network
    4. Add & Norm layer (residual connection + layer normalization)

    Regularization is implemented through:
    - Dropout after each major component (attention and feed-forward)
    - Layer dropout that occasionally skips entire layers
    - Layer normalization to stabilize training

    Residual connections:
    - Added around both the attention and feed-forward layers
    - Help with gradient flow and enable training of deep networks
    - Follow the form: LayerNorm(x + Sublayer(x))

    Expected Tensor Shapes:
        Input x: (batch_size, sequence_length, d_model)
        Attention mask: (batch_size, 1, sequence_length, sequence_length)
        Output: (batch_size, sequence_length, d_model)

    Args:
        num_of_heads (int, optional): Number of attention heads. Defaults to 8.
        d_model (int, optional): Model's dimension. Defaults to 512.
        d_ff (int, optional): Feed-forward network's hidden dimension. Defaults to 2048.
        p_drop (float, optional): Dropout probability for attention and feed-forward. Defaults to 0.1.
        p_drop_layer (float, optional): Probability of skipping entire layer. Defaults to 0.1.

    Attributes:
        d_model (int): Model's dimension
        num_of_heads (int): Number of attention heads
        d_ff (int): Feed-forward network's hidden dimension
        p_drop (float): Dropout probability
        p_drop_layer (float): Layer dropout probability
        multi_head_attention (MultiHeadAttention): Multi-head attention module
        feed_forward (FeedForward): Position-wise feed-forward network
        dropout (nn.Dropout): Dropout layer for regularization
        norm1 (nn.LayerNorm): First layer normalization
        norm2 (nn.LayerNorm): Second layer normalization
    """

    def __init__(
        self,
        num_of_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        p_drop: float = 0.1,
        p_drop_layer: float = 0.1,
    ) -> None:
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.p_drop_layer = p_drop_layer

        self.multi_head_attention = MultiHeadAttention(num_of_heads, d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, masks: Tensor = None) -> Tensor:
        """Forward pass of the encoder layer.

        Applies the following operations:
        1. Multi-head self-attention with residual connection and layer norm
        2. Feed-forward network with residual connection and layer norm
        Optionally skips the entire layer based on layer dropout probability.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            masks (Tensor, optional): Attention masks. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        if torch.rand(1).item() < self.p_drop_layer:
            return x

        attn_values = self.multi_head_attention(x, x, x, masks)
        ff_input = self.norm1(self.dropout(attn_values + x))
        return self.norm2(self.dropout(self.feed_forward(ff_input)) + ff_input)


class Encoder(nn.Module):
    """Complete encoder stack of the Transformer model.

    Consists of multiple EncoderLayers stacked on top of each other.
    Each layer maintains the same dimensionality, allowing for deep architectures
    through residual connections and proper regularization.

    Expected Tensor Shapes:
        Input x: (batch_size, sequence_length, d_model)
        Attention mask: (batch_size, 1, sequence_length, sequence_length)
        Output: (batch_size, sequence_length, d_model)

    Regularization Features:
    - Dropout in each encoder layer
    - Layer dropout that can skip entire layers
    - Layer normalization in each encoder layer

    Residual Architecture:
    - Each layer contains two residual connections
    - Allows for effective training of deep networks
    - Maintains constant dimensionality throughout the stack

    Args:
        layer_count (int, optional): Number of encoder layers. Defaults to 6.
        d_model (int, optional): Model's dimension. Defaults to 512.
        num_of_heads (int, optional): Number of attention heads. Defaults to 8.
        d_ff (int, optional): Feed-forward network's hidden dimension. Defaults to 2048.
        p_drop (float, optional): Dropout probability. Defaults to 0.1.
        p_drop_layer (float, optional): Layer dropout probability. Defaults to 0.1.

    Attributes:
        layer_count (int): Number of encoder layers
        d_model (int): Model's dimension
        num_of_heads (int): Number of attention heads
        layers (nn.ModuleList): List of EncoderLayer modules
    """

    def __init__(
        self,
        layer_count: int = 6,
        d_model: int = 512,
        num_of_heads: int = 8,
        d_ff: int = 2048,
        p_drop: float = 0.1,
        p_drop_layer: float = 0.1,
    ) -> None:
        super(Encoder, self).__init__()

        self.layer_count = layer_count
        self.d_model = d_model
        self.num_of_heads = num_of_heads

        # Initialize the encoder layers.
        self.layers = nn.ModuleList(
            [
                EncoderLayer(num_of_heads, d_model, d_ff, p_drop, p_drop_layer)
                for _ in range(layer_count)
            ]
        )

    def forward(self, x: Tensor, masks: Tensor = None) -> Tensor:
        """Forward pass through the entire encoder stack.

        Sequentially applies each encoder layer to the input.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            masks (Tensor, optional): Attention masks. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, masks)
        return x
