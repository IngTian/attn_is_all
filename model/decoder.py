"""
Transformer Decoder implementation as described in 'Attention Is All You Need'.

This module implements the Decoder part of the Transformer architecture, consisting of
multiple DecoderLayers stacked together. Each DecoderLayer contains:
1. Masked Multi-Head Self-Attention
2. Cross Multi-Head Attention with Encoder output
3. Position-wise Feed-Forward Network

Expected Tensor Shapes throughout the decoder:
    - Decoder input (x): (batch_size, target_seq_len, d_model)
    - Encoder output: (batch_size, source_seq_len, d_model)
    - Source mask: (batch_size, 1, 1, source_seq_len)
    - Target mask: (batch_size, 1, target_seq_len, target_seq_len)
"""

import torch
import torch.nn as nn
from torch import Tensor

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    """A single layer of the Transformer decoder.

    Each decoder layer consists of:
    1. Masked Multi-head self-attention
    2. Add & Norm layer (residual connection + layer normalization)
    3. Cross Multi-head attention with encoder output
    4. Add & Norm layer
    5. Position-wise feed-forward network
    6. Add & Norm layer

    Regularization is implemented through:
    - Dropout after each major component
    - Layer dropout that occasionally skips entire layers
    - Layer normalization for training stability

    Residual connections:
    - Added around all three main components
    - Help with gradient flow in deep networks
    - Follow the form: LayerNorm(x + Sublayer(x))

    Expected Tensor Shapes:
        Input x: (batch_size, target_seq_len, d_model)
        Encoder output: (batch_size, source_seq_len, d_model)
        Source mask: (batch_size, 1, 1, source_seq_len)
        Target mask: (batch_size, 1, target_seq_len, target_seq_len)
        Output: (batch_size, target_seq_len, d_model)

    Args:
        num_of_heads (int, optional): Number of attention heads. Defaults to 8.
        d_model (int, optional): Model's dimension. Defaults to 512.
        d_ff (int, optional): Feed-forward network's hidden dimension. Defaults to 2048.
        p_drop (float, optional): Dropout probability. Defaults to 0.1.
        p_drop_layer (float, optional): Layer dropout probability. Defaults to 0.1.

    Attributes:
        num_of_heads (int): Number of attention heads
        d_model (int): Model's dimension
        d_ff (int): Feed-forward network's hidden dimension
        p_drop (float): Dropout probability
        p_drop_layer (float): Layer dropout probability
        masked_multi_head_attention (MultiHeadAttention): Self-attention module
        cross_multi_head_attention (MultiHeadAttention): Cross-attention module
        feed_forward (FeedForward): Position-wise feed-forward network
        norm1, norm2, norm3 (nn.LayerNorm): Layer normalization modules
    """

    def __init__(
        self,
        num_of_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        p_drop=0.1,
        p_drop_layer=0.1,
    ):
        super(DecoderLayer, self).__init__()

        self.num_of_heads = num_of_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.p_drop_layer = p_drop_layer

        self.masked_multi_head_attention = MultiHeadAttention(num_of_heads, d_model)
        self.cross_multi_head_attention = MultiHeadAttention(num_of_heads, d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(p_drop)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        enc_output: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """Forward pass through decoder layer.

        Applies in sequence:
        1. Masked self-attention with residual connection and layer norm
        2. Cross-attention with encoder output, residual connection and layer norm
        3. Feed-forward network with residual connection and layer norm

        Args:
            x (Tensor): Input tensor (batch_size, target_seq_len, d_model)
            enc_output (Tensor): Encoder output (batch_size, source_seq_len, d_model)
            src_mask (Tensor): Source mask (batch_size, 1, 1, source_seq_len)
            tgt_mask (Tensor): Target mask (batch_size, 1, target_seq_len, target_seq_len)

        Returns:
            Tensor: Output tensor (batch_size, target_seq_len, d_model)
        """
        if torch.rand(1).item() < self.p_drop_layer:
            return x

        masked_attn_output = self.masked_multi_head_attention(x, x, x, tgt_mask)
        reg_masked_attn_output = self.norm1(self.dropout(masked_attn_output + x))

        cross_attn_output = self.cross_multi_head_attention(
            reg_masked_attn_output, enc_output, enc_output, src_mask
        )
        reg_cross_attn_output = self.norm2(
            self.dropout(cross_attn_output + reg_masked_attn_output)
        )

        ff_output = self.norm3(
            self.dropout(
                self.feed_forward(reg_cross_attn_output) + reg_cross_attn_output
            )
        )

        return ff_output


class Decoder(nn.Module):
    """Complete decoder stack of the Transformer model.

    Consists of multiple DecoderLayers stacked on top of each other.
    Each layer maintains the same dimensionality, allowing for deep architectures
    through residual connections and proper regularization.

    Regularization Features:
    - Dropout in each decoder layer
    - Layer dropout that can skip entire layers
    - Layer normalization in each decoder layer

    Expected Tensor Shapes:
        Input x: (batch_size, target_seq_len, d_model)
        Encoder output: (batch_size, source_seq_len, d_model)
        Source mask: (batch_size, 1, 1, source_seq_len)
        Target mask: (batch_size, 1, target_seq_len, target_seq_len)
        Output: (batch_size, target_seq_len, d_model)

    Args:
        layer_count (int, optional): Number of decoder layers. Defaults to 6.
        num_of_heads (int, optional): Number of attention heads. Defaults to 8.
        d_model (int, optional): Model's dimension. Defaults to 512.
        d_ff (int, optional): Feed-forward network's hidden dimension. Defaults to 2048.
        p_drop (float, optional): Dropout probability. Defaults to 0.1.
        p_drop_layer (float, optional): Layer dropout probability. Defaults to 0.1.

    Attributes:
        layer_count (int): Number of decoder layers
        num_of_heads (int): Number of attention heads
        d_model (int): Model's dimension
        layers (nn.ModuleList): List of DecoderLayer modules
    """

    def __init__(
        self,
        layer_count: int = 6,
        num_of_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        p_drop=0.1,
        p_drop_layer=0.1,
    ):
        super(Decoder, self).__init__()

        self.layer_count = layer_count
        self.num_of_heads = num_of_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.p_drop_layer = p_drop_layer

        self.layers = nn.ModuleList(
            [
                DecoderLayer(num_of_heads, d_model, d_ff, p_drop, p_drop_layer)
                for _ in range(layer_count)
            ]
        )

    def forward(
        self,
        x: Tensor,
        enc_output: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """Forward pass through the entire decoder stack.

        Sequentially applies each decoder layer to the input.

        Args:
            x (Tensor): Input tensor (batch_size, target_seq_len, d_model)
            enc_output (Tensor): Encoder output (batch_size, source_seq_len, d_model)
            src_mask (Tensor): Source mask (batch_size, 1, 1, source_seq_len)
            tgt_mask (Tensor): Target mask (batch_size, 1, target_seq_len, target_seq_len)

        Returns:
            Tensor: Output tensor (batch_size, target_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
