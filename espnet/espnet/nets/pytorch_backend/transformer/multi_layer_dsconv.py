#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#
# Copyright (c) 2021 Seunghun Jeong
#
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import torch
from torch import nn
# from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetSEModule
# from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetLSEModule
# from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetECAModule
# from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetGCModule
# from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetGCAModule
from espnet.nets.pytorch_backend.conformer.swish import Swish


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

        self.activation = Swish()
        # if se_type == "se":
        #     self.se_layer = ContextNetSEModule(in_chans, self.activation)
        # elif se_type == "eca":      # don't work
        #     self.se_layer = ContextNetECAModule(in_chans)
        # elif se_type == "lse":
        #     self.se_layer = ContextNetLSEModule(in_chans)
        # elif se_type == "bam":
        # elif se_type == "gc":
        #     self.se_layer = ContextNetGCAModule(in_chans)
        # elif se_type == "da":
        # else: self.se_layer = None

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Batch of input tensors (B, T, in_chans).

        Returns:
            torch.Tensor: Batch of output tensors (B, T, hidden_chans).

        """
        # if self.se_layer is not None:
        #     residual = x
        x = self.activation(self.pointwise_conv1(self.depthwise_conv1(x.transpose(-1, 1)))).transpose(-1, 1)
        x = self.pointwise_conv2(self.depthwise_conv2(self.dropout(x).transpose(-1, 1)))
        # if self.se_layer is not None:
        #     x = self.se_layer(x, mask.squeeze(1).sum(1))
        #     x = self.dropout(self.activation(x))
        #     return x.transpose(-1, 1) + residual
        # else:
        #     return x.transpose(-1, 1)
        return x.transpose(-1, 1)

class SpinalFC(torch.nn.Module):
    def __init__(self, in_chans, dropout_rate):

        super(SpinalFC, self).__init__()
        
        self.Half_width = in_chans//2

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_chans//2, in_chans//4),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_chans//2 + in_chans//4, in_chans//4),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_chans//2 + in_chans//4, in_chans//4),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_chans//2 + in_chans//4, in_chans//4),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:,:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,:,self.Half_width:2*self.Half_width], x1], dim=2))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,:,0:self.Half_width], x2], dim=2))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,:,self.Half_width:2*self.Half_width], x3], dim=2))

        x = torch.cat([x1, x2], dim=2)
        x = torch.cat([x, x3], dim=2)
        x = torch.cat([x, x4], dim=2)

        return x

class DSConv1dLinear(torch.nn.Module):
    """Depthwise Seperable Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.

        """
        super(DSConv1dLinear, self).__init__()

        self.pointwise_conv1 = nn.Conv1d(
            in_chans,
            hidden_chans,
            1,
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

        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Batch of input tensors (B, T, in_chans).

        Returns:
            torch.Tensor: Batch of output tensors (B, T, hidden_chans).

        """
        x = torch.relu(self.pointwise_conv1(self.depthwise_conv1(x.transpose(-1, 1)))).transpose(-1, 1)
        return self.w_2(self.dropout(x))
