#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#
# Copyright (c) 2021 Seunghun Jeong
#
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import torch
from torch import nn
import torch.nn.functional as F


class MultiLayeredDSConv1d(torch.nn.Module):
    """Multi-layered Depthwise Seperable conv1d for Transformer block.
       A variant of MultiLayeredConv1d

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize MultiLayeredConv1d module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(MultiLayeredDSConv1d, self).__init__()

        self.pointwise_conv1 = nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.pointwise_conv2 = nn.Conv1d(
            hidden_chans,
            in_chans,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.depthwise_conv1 = nn.Conv1d(
            in_chans,
            in_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=in_chans,
        )

        self.depthwise_conv2 = nn.Conv1d(
            hidden_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=hidden_chans,
        )

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Batch of input tensors (B, T, in_chans).

        Returns:
            torch.Tensor: Batch of output tensors (B, T, hidden_chans).

        """
        x = torch.relu(self.pointwise_conv1(self.depthwise_conv1(x.transpose(-1, 1)))).transpose(-1, 1)
        return self.pointwise_conv2(self.depthwise_conv2(self.dropout(x).transpose(-1, 1))).transpose(-1, 1)


class DyConv1dLinear(torch.nn.Module):
    """Depthwise Seperable Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate, wshare):
        """Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(DyConv1dLinear, self).__init__()

        assert in_chans % wshare == 0
        self.wshare = wshare
        self.kernel_size = kernel_size
        
        self.linear_weight = nn.Linear(in_chans, self.wshare * 1 * kernel_size)
        nn.init.xavier_uniform_(self.linear_weight.weight)
        self.bias = nn.Parameter(torch.Tensor(in_chans))

        self.pointwise_conv1 = nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=0,
        )
        self.activation = nn.SELU()
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Batch of input tensors (B, T, in_chans).

        Returns:
            torch.Tensor: Batch of output tensors (B, T, hidden_chans).

        """
        B, T, C = x.size()
        H = self.wshare
        k = self.kernel_size

        weight = self.linear_weight(x) 
        weight = weight.view(B, T, H, k).transpose(1, 2).contiguous()  # B x H x T x k
        weight_new = torch.zeros(B * H * T * (T + k - 1), dtype=weight.dtype)
        weight_new = weight_new.view(B, H, T, T + k - 1).fill_(float("-inf"))
        weight_new = weight_new.to(x.device)  # B x H x T x T+k-1
        weight_new.as_strided(
            (B, H, T, k), ((T + k - 1) * T * H, (T + k - 1) * T, T + k, 1)
        ).copy_(weight)
        weight_new = weight_new.narrow(-1, int((k - 1) / 2), T)  # B x H x T x T(k)
        weight_new = F.softmax(weight_new, dim=-1)
        self.attn = weight_new
        weight_new = weight_new.view(B * H, T, T)

        x = x.transpose(1, 2).contiguous()  # B x C x T
        x = x.view(B * H, int(C / H), T).transpose(1, 2)
        x = torch.bmm(weight_new, x)  # BH x T x C/H
        x = x.transpose(1, 2).contiguous().view(B, C, T)
        x = x + self.bias.view(1, -1, 1)

        x = self.activation(self.pointwise_conv1(x).transpose(1, 2))

        return self.w_2(self.dropout(x))
