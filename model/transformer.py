"""
Complete Transformer model implementation as described in 'Attention Is All You Need'.

This module implements the full Transformer architecture, combining:
- Input Embeddings
- Positional Encoding
- Encoder Stack
- Decoder Stack
- Output Linear Layer

Expected Tensor Shapes throughout the model:
    - Source input: (batch_size, src_seq_len)
    - Target input: (batch_size, tgt_seq_len)
    - Source mask: (batch_size, 1, 1, src_seq_len)
    - Target mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
    - Output: (batch_size, tgt_seq_len, vocab_size)
"""

import torch.nn as nn
from torch import Tensor

from model.decoder import Decoder
from model.encoder import Encoder
from model.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks.

    This implementation follows the architecture described in the paper
    "Attention Is All You Need". It includes both encoder and decoder stacks,
    with positional encoding and final output projection.

    Architecture Overview:
    1. Input Embedding + Positional Encoding
    2. Encoder Stack (N identical layers)
    3. Decoder Stack (N identical layers)
    4. Linear projection to vocabulary size

    Args:
        vocab_size (int): Size of the vocabulary
        max_seq_length (int): Maximum sequence length
        enc_layer_count (int, optional): Number of encoder layers. Defaults to 6.
        dec_layer_count (int, optional): Number of decoder layers. Defaults to 6.
        d_model (int, optional): Model's dimension. Defaults to 512.
        num_of_heads (int, optional): Number of attention heads. Defaults to 8.
        d_ff (int, optional): Feed-forward network's hidden dimension. Defaults to 2048.
        p_drop (float, optional): Dropout probability. Defaults to 0.1.
        p_drop_layer (float, optional): Layer dropout probability. Defaults to 0.1.

    Attributes:
        vocab_size (int): Size of the vocabulary
        max_seq_length (int): Maximum sequence length
        d_model (int): Model's dimension
        encoder (Encoder): Encoder stack
        decoder (Decoder): Decoder stack
        positional_encoding (PositionalEncoding): Positional encoding layer
        output_layer (nn.Linear): Final projection to vocabulary size
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int,
        enc_layer_count: int = 6,
        dec_layer_count: int = 6,
        d_model: int = 512,
        num_of_heads: int = 8,
        d_ff: int = 2048,
        p_drop: float = 0.1,
        p_drop_layer: float = 0.1,
    ):
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.enc_layer_count = enc_layer_count
        self.dec_layer_count = dec_layer_count
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.p_drop_layer = p_drop_layer

        self.encoder = Encoder(
            enc_layer_count,
            d_model,
            num_of_heads,
            d_ff,
            p_drop,
            p_drop_layer,
        )

        self.decoder = Decoder(
            dec_layer_count,
            num_of_heads,
            d_model,
            d_ff,
            p_drop,
            p_drop_layer,
        )

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_masks: Tensor,
        tgt_masks: Tensor,
    ) -> Tensor:
        """Forward pass through the Transformer.

        The forward pass sequence:
        1. Add positional encoding to source and target inputs
        2. Pass through encoder stack
        3. Pass through decoder stack with encoder output
        4. Project to vocabulary size

        Args:
            src (Tensor): Source tensor of shape (batch_size, src_seq_len, d_model)
            tgt (Tensor): Target tensor of shape (batch_size, tgt_seq_len, d_model)
            src_masks (Tensor): Source attention mask (batch_size, 1, 1, src_seq_len)
            tgt_masks (Tensor): Target attention mask (batch_size, 1, tgt_seq_len, tgt_seq_len)

        Returns:
            Tensor: Output logits of shape (batch_size, tgt_seq_len, vocab_size)
        """
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        enc_output = self.encoder(src, src_masks)
        dec_output = self.decoder(tgt, enc_output, src_masks, tgt_masks)
        output = self.output_layer(dec_output)
        return output
