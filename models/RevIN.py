import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for Time Series Forecasting
    """

    def __init__(self, num_features, eps=1e-8, affine=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('mean', torch.zeros(1, 1, num_features))
        self.register_buffer('stdev', torch.ones(1, 1, num_features))

    def forward(self, x, mode='norm'):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]
            mode: 'norm' for normalization, 'denorm' for denormalization
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError

        return x

    def _get_statistics(self, x):
        """Calculate mean and standard deviation along batch and time dimensions"""
        # x: [batch_size, seq_len, num_features] or [batch_size, pred_len, num_features]
        # Compute statistics along [batch_size, seq_len] dimensions
        dim = 1
        self.mean = torch.mean(x, dim=dim, keepdim=True)
        self.stdev = torch.std(x, dim=dim, keepdim=True) + self.eps

    def _normalize(self, x):
        """Normalize input tensor"""
        # x: [batch_size, seq_len, num_features]
        # Mean and std have shape [batch_size, 1, num_features]
        # Normalize
        x = (x - self.mean) / self.stdev

        # Apply affine transformation if enabled
        if self.affine:
            weight = self.affine_weight.view(1, 1, -1)
            bias = self.affine_bias.view(1, 1, -1)
            x = x * weight + bias

        return x

    def _denormalize(self, x):
        """Denormalize output tensor"""
        # x: [batch_size, pred_len, num_features]

        # Remove affine transformation if applied
        if self.affine:
            weight = self.affine_weight.view(1, 1, -1)
            bias = self.affine_bias.view(1, 1, -1)
            x = (x - bias) / weight

        # Denormalize
        x = x * self.stdev + self.mean

        return x