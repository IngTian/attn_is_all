"""
The feed forward neural network model.
"""

from torch import Tensor
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize FeedForward model.

        Args:
            d_model (int): The number of expected features in the input.
            d_ff (int): The number of expected features in the output.
            dropout (float, optional): The dropout probability. Defaults to 0.1.
        """
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute FeedForward.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.W2(self.relu(self.W1(x)))
