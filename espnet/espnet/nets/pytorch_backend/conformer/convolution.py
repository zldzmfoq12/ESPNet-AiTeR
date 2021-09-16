#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#
# Copyright (c) 2021 Seunghun Jeong
#
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

from torch import nn
import torch
import torch.nn.functional as F
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetSEModule

class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(
        self, 
        channels, 
        kernel_size, 
        activation=nn.ReLU(),
        se_type="se", 
        bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

        if se_type == "se":
            self.se_layer = ContextNetSEModule(channels, self.activation)
        else: self.se_layer = None

    def forward(self, x, mask):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, time)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        if self.se_layer is not None:
            x = self.se_layer(x, mask.squeeze(1).sum(1))

        x = self.pointwise_conv2(x)
   #     x = self.activation(self.se_layer(x, mask.squeeze(1).sum(1)))

        return x.transpose(1, 2)

class FConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(
        self, 
        channels, 
        kernel_size, 
        activation=nn.ReLU(),
        se_type="se", 
        bias=True):
        """Construct an ConvolutionModule object."""
        super(FConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.conv = nn.Conv1d(
            channels,
            4 * channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

        self.norm = nn.BatchNorm1d(4 * channels)
        self.pointwise_conv = nn.Conv1d(
            4 * channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation
        if se_type == "se":
            self.se_layer = ContextNetSEModule(2*channels, self.activation)
        else: self.se_layer = None

    def forward(self, x, mask):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.conv(x)  # (batch, 2*channel, time)

        x = self.activation(self.norm(x))

        if self.se_layer is not None:
            x = self.se_layer(x, mask.squeeze(1).sum(1))

        x = self.pointwise_conv(x)
   #     x = self.activation(self.se_layer(x, mask.squeeze(1).sum(1)))

        return x.transpose(1, 2)

class LConvolutionModule(nn.Module):
    """Lightweight ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        conv_wshare (int): The number of channels that shared weights.

    """

    def __init__(
        self, 
        channels, 
        kernel_size, 
        activation=nn.ReLU(), 
        wshare=4, 
        dropout_rate=0.0, 
        bias=True):
        """Construct an ConvolutionModule object."""
        super(LConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0
        assert channels % wshare == 0

        self.wshare = wshare
        self.kernel_size = kernel_size
        self.padding_size = (kernel_size - 1) // 2

        self.weight = nn.Parameter(
            torch.Tensor(wshare, 1, kernel_size).uniform_(0, 1)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(channels))
        else:
            self.bias = None

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.dropout_rate = dropout_rate

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """

        B, T, C = x.size()
        H = self.wshare

        weight = self.weight
        weight = F.softmax(weight, dim=-1)
        weight = F.dropout(weight, self.dropout_rate, training=self.training)

        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        x = x.contiguous().view(-1, H, T)  # B x C x T

        x = F.conv1d(x, weight, padding=self.padding_size, groups=self.wshare).view(
            B, C, T
        )
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1)

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)

class DConvolutionModule(nn.Module):
    """Dynamic ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        conv_wshare (int): The number of channels that shared weights.

    """

    def __init__(
        self, 
        channels, 
        kernel_size, 
        activation=nn.ReLU(), 
        wshare=4, 
        dropout_rate=0.0, 
        bias=True):
        
        """Construct an ConvolutionModule object."""
        super(DConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0
        assert channels % wshare == 0

        self.wshare = wshare
        self.kernel_size = kernel_size
        self.padding_size = (kernel_size - 1) // 2

        if bias:
            self.bias = nn.Parameter(torch.Tensor(channels))
        else:
            self.bias = None

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        
        self.linear_weight = nn.Linear(channels, self.wshare * 1 * kernel_size)
        nn.init.xavier_uniform_(self.linear_weight.weight)

        self.dropout_rate = dropout_rate
        self.attn = None
        



    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        B, T, C = x.size()
        H = self.wshare
        k = self.kernel_size

        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2).contiguous()
        # first convolution
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        # GLU mechanism
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # get kernel of convolution
        x = x.transpose(1, 2).contiguous()  # B x T x C
        weight = self.linear_weight(x)  # B x T x kH
        weight = F.dropout(weight, self.dropout_rate, training=self.training)
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

        # 1D dynamic Conv
        x = x.transpose(1, 2).contiguous()  # B x C x T
        x = x.view(B * H, int(C / H), T).transpose(1, 2)
        x = torch.bmm(weight_new, x)  # BH x T x C/H
        x = x.transpose(1, 2).contiguous().view(B, C, T)
        
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1)

        x = self.pointwise_conv2(x)   

        return x.transpose(1, 2)