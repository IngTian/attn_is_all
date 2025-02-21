"""
Multi-Head Attention implementation as described in 'Attention Is All You Need'.
"""

import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_of_heads: int,
        d_model: int,
    ):
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

    def split_head(self, x):
        """
        x: (batch_size, seq_len, d_model)

        return: (batch_size, num_of_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_of_heads, self.d_k).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Q: (batch_size, num_of_heads, seq_len, d_k)
        K: (batch_size, num_of_heads, seq_len, d_k)
        V: (batch_size, num_of_heads, seq_len, d_k)
        """

        # Attention weights should be of (batch_size, num_of_heads, seq_len, seq_len).
        attn_weights = torch.softmax(
            torch.matmul(Q, K.transpose(-1, -2))
            / torch.sqrt(torch.tensor(self.d_k, dtype=Q.dtype, device=Q.device)),
            -1,
        )

        return torch.matmul(attn_weights, V), attn_weights

    def concatenate_heads(self, x):
        """
        x: (batch_size, num_of_heads, seq_len, d_k)
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V):
        # Get Q, K, V after apply W_Q, W_K, and W_V.
        Q = self.split_head(self.W_Q(Q))
        K = self.split_head(self.W_K(K))
        V = self.split_head(self.W_V(V))

        # Run through scaled dot-product attention.
        v, _ = self.scaled_dot_product_attention(Q, K, V)

        # Combine heads.
        combined_v = self.concatenate_heads(v)

        return self.W_O(combined_v)
