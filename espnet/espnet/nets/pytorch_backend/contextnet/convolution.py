#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Copyright (c) 2021 Seunghun Jeong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from re import S
from torch import Tensor
from typing import Tuple
from torch import nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.conformer.swish import Swish
import torch
import math
import numpy as np


class MaskConv1d(nn.Conv1d):
    """1D convolution with sequence masking """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
    ) -> None:
        super(MaskConv1d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias)

    def get_sequence_lengths(self, seq_lengths):
        return (
            (seq_lengths + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        """
        inputs: BxDxT
        input_lengths: B
        """
        max_length = inputs.size(2)

        indices = torch.arange(max_length).to(input_lengths.dtype).to(input_lengths.device)
        indices = indices.expand(len(input_lengths), max_length)

        mask = indices >= input_lengths.unsqueeze(1)
        inputs = inputs.masked_fill(mask.unsqueeze(1).to(device=inputs.device), 0)

        output_lengths = self.get_sequence_lengths(input_lengths)
        output = super(MaskConv1d, self).forward(inputs)

        del mask, indices

        return output, output_lengths


class ContextNetConvModule(nn.Module):
    """
    When the stride is 1, it pads the input so the output has the shape as the input.
    And when the stride is 2, it does not pad the input.
    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        dropout_rate (float, optional): Value of dropout_rate (default: 0.1)
        activation (nn.Function, optional): Activation function (default : Relu)
        bias (bool, optional): Flag indication use bias or not (default : True)
        residual (bool, optional): Flag indication that is residual or not (default : False)
    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution layer `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output, output_lengths
        - **output**: Output of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 5,
            stride = 1,
            padding = 0,
            dropout_rate = 0.1,
            activation = nn.ReLU(),
            bias = True,
            residual = False,
    ):
        super(ContextNetConvModule, self).__init__()

        if residual==False:
            self.stride=1
            self.padding = (kernel_size - 1) // 2
            self.kernel_size=kernel_size
            self.activation = activation
            self.dropout = nn.Dropout(dropout_rate)
            self.depthwise_conv = nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=in_channels,
                bias=bias,
            )
            
        else:
            self.stride = stride
            self.padding = 0
            self.kernel_size = 1
        self.stride_size = stride
            
        self.pointwise_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

        # self.conv = MaskConv1d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=(kernel_size - 1) // 2,
        #     bias=bias,
        #     dilation=1
        # )

        self.norm = nn.BatchNorm1d(out_channels)
        

        self.residual = residual


    def forward(
            self,
            x,
            input_lengths
            ):
        """
        Forward propagate a `inputs` for convolution layer.
        Args:
            **inputs** (torch.FloatTensor): Input of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """

        if self.residual == False:
            x, output_lengths = self.pointwise_conv(self.depthwise_conv(x)), self._get_sequence_lengths(input_lengths)
            x = self.norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x, output_lengths = self.pointwise_conv(x), self._get_sequence_lengths(input_lengths)
            x = self.norm(x)
            

        return x, output_lengths

    def _get_sequence_lengths(self, seq_lengths):
        return (
                (seq_lengths + 2 * self.padding
                 - self.kernel_size) // self.stride_size + 1
        )



class ContextNetDConvModule(nn.Module):
    """
    --------------------------------------------------------------------------------
    This is a dynamic convolution module. 
    --------------------------------------------------------------------------------
    When the stride is 1, it pads the input so the output has the shape as the input.
    And when the stride is 2, it does not pad the input.
    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        wshare (int): The number of kernel of convolution(the number of grouped channels)
        dropout_rate (float): Dropout_rate of dynamic convolution
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        activation (nn.Function, optional): Activation function (default : Relu)
        bias (bool, optional): Flag indication use bias or not (default : True)
        residual (bool, optional): Flag indication that is residual or not (default : False)
    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution layer `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output, output_lengths
        - **output**: Output of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            wshare=4,
            dropout_rate=0.1,
            kernel_size = 5,
            stride = 1,
            padding = 0,
            activation = nn.ReLU(),
            bias = True,
            residual = False,
    ):
        super(ContextNetDConvModule, self).__init__()

        assert in_channels % wshare == 0

        self.wshare = wshare
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.attn = None
        self.use_bias = bias

        self.linear = nn.Linear(in_channels, in_channels * 2)
        self.act = nn.GLU()

        self.linear_weight = nn.Linear(in_channels, self.wshare * 1 * self.kernel_size)
        nn.init.xavier_uniform_(self.linear_weight.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))        

        self.pointwise_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        if stride == 1:
            self.padding_size = (self.kernel_size - 1) // 2
        elif stride == 2:
            self.padding_size = padding

#        self.layer_norm = LayerNorm(out_channels)
#        self.norm = nn.BatchNorm1d(out_channels)
        self.norm = nn.GroupNorm(1, out_channels)
#        self.activation = activation
        self.stride = stride


    def forward(
            self,
            x,
            input_lengths
            ):
        """
        Forward propagate a `inputs` for convolution layer.
        Args:
            **inputs** (torch.FloatTensor): Input of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of convolution layer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        x = x.transpose(1, 2) # B x T x C

#        x = self.layer_norm(x)

        B, T, C = x.size()
        H = self.wshare
        k = self.kernel_size

        # liner layer
        x = self.linear(x)
        # GLU activation
        x = self.act(x)

        weight = self.linear_weight(x)
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

        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2).contiguous() # B x C x T

        # 1D dynamic Conv
        x = x.view(B * H, int(C / H), T).transpose(1, 2)
        x = torch.bmm(weight_new, x)  # BH x T x C/H
        x = x.transpose(1, 2).contiguous().view(B, C, T)
        if self.use_bias:
            x = x + self.bias.view(1, -1, 1)

        x, output_lengths = self.pointwise_conv(x), self._get_sequence_lengths(input_lengths)

        x = self.norm(x)
        
#        x = self.activation(x)

        return x, output_lengths

    def _get_sequence_lengths(self, seq_lengths):
        return (
                (seq_lengths + 2 * self.padding_size
                 - self.kernel_size) // self.stride + 1
        )


class ContextNetSEModule(nn.Module):
    """
    Squeeze-and-excitation module.
    Args:
        dim (int): Dimension to be used for two fully connected (FC) layers
        activation (nn.Module): Activation function to be used for two fully connected (FC) layers
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim, activation=nn.ReLU()):
        super(ContextNetSEModule, self).__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'

        self.dim = dim
        self.sequential = nn.Sequential(
            nn.Linear(dim, dim // 8),
            activation,
            nn.Linear(dim // 8, dim),
        )

    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        residual = x
        seq_lengths = x.size(2)

        x = x.sum(dim=2) / input_lengths.unsqueeze(1)
        x = self.sequential(x)

        x = x.sigmoid().unsqueeze(2)
        x = x.repeat(1, 1, seq_lengths)

        return x * residual


class ContextNetECAModule(nn.Module):
    """
    Efficient-Channel-Attention module.
    https://arxiv.org/pdf/1910.03151
    Args:
        dim (int): Dimension to be used for calculating kernel size in convolution layers
        gamma (int): Value to be used for calculating kernel size in convolution layers
        b (int): Value to be used for calculating kernel size in convolution layers
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of ECA-Layer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim):
        super(ContextNetECAModule, self).__init__()

#        tmp = int(abs((math.log2(dim) + b) / gamma))
#        kernel_size = tmp if tmp % 2 else tmp + 1
        kernel_size = 41
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)


    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        residual = x
        seq_lengths = x.size(2)

        x = x.sum(dim=2) / input_lengths.unsqueeze(1)
        y = self.conv(x.unsqueeze(1)).squeeze(1)
        y = y.sigmoid().unsqueeze(2)
        y= y.repeat(1, 1, seq_lengths)
        
        return y * residual


class ContextNetWSEModule(nn.Module):
    """
    Channel wise Squeeze-and-excitation module.
    Args:
        dim (int): Dimension to be used for channelwise fully connected (FC) layers
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim):
        super(ContextNetWSEModule, self).__init__()

        self.weights = nn.Parameter(torch.Tensor(1, dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        residual = x
        seq_lengths = x.size(2)

        x = x.sum(dim=2) / input_lengths.unsqueeze(1)
        x = x * self.weights

        x = x.sigmoid().unsqueeze(2)
        x = x.repeat(1, 1, seq_lengths)

        return x * residual


class ContextNetLSEModule(nn.Module):
    """
    Squeeze-and-excitation module with 1 FC layer.
    Args:
        dim (int): Dimension to be used for fully connected (FC) layers without channel reduction
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim):
        super(ContextNetLSEModule, self).__init__()

        self.linear = nn.Linear(dim, dim)


    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        residual = x
        seq_lengths = x.size(2)

        x = x.sum(dim=2) / input_lengths.unsqueeze(1)
        x = self.linear(x)

        x = x.sigmoid().unsqueeze(2)
        x = x.repeat(1, 1, seq_lengths)

        return x * residual


class ContextNetBAMModule(nn.Module):
    """
    Channel Attention module in Convolutional Block Attention module.
    https://arxiv.org/pdf/1807.06521
    Args:
        dim (int): Dimension to be used for two fully connected (FC) layers
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim, activation=nn.ReLU()):
        super(ContextNetBAMModule, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(dim, dim // 8),
            activation,
            nn.Linear(dim // 8, dim),
        )

    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        residual = x
        seq_lengths = x.size(2)

        avgx = x.sum(dim=2) / input_lengths.unsqueeze(1)
        avgx = self.sequential(avgx)
        

        maxx = F.max_pool1d(x, (x.size(2)), stride=(x.size(2)))
        maxx = self.sequential(maxx.squeeze(2))

        total_x = avgx + maxx

        total_x = total_x.sigmoid().unsqueeze(2)
        
        total_x = total_x.repeat(1, 1, seq_lengths)

        return total_x * residual


class ContextNetGMModule(nn.Module):
    """
    Convolutional Cross-channel Interaction module.
    Args:
        dim (int): Dimension to be used for convolutional layers
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim):
        super(ContextNetGMModule, self).__init__()

        kernel_size = 1
        self.conv = nn.Conv1d(dim, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.conv2 = nn.Conv1d(dim, dim // 8, 1, bias=False)
        self.groupnorm = nn.GroupNorm(1, dim // 8)
        self.activation = Swish()
        self.conv3 = nn.Conv1d(dim // 8, dim, 1, bias=False)


    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        x
        seq_lengths = x.size(2)
        
        # # Take Global Average Pooling to Input
        # gap = x.sum(dim=2) / input_lengths.unsqueeze(1)     # B x C
        # gap = gap.unsqueeze(2)      # B x C x 1

        # # Calculate Gram Matrix of GAP resluts
        # Gm = torch.bmm(gap, gap.transpose(1,2))     # B x C x C
        # Take Pointwise Convolution to Gram Matrix

        context = torch.bmm(x, x.transpose(1,2))   # B x C x C

        context = self.conv(context)      # B x 1 x C
        context = context.sigmoid().transpose(1,2)      # B x C x 1

        transform = self.conv2(context)
        transform = self.groupnorm(transform)
        transform = self.activation(transform)
        transform = self.conv3(transform)
        
        transform = transform.repeat(1, 1, seq_lengths)     # B x C x T

        return transform * x



class ContextNetGCModule(nn.Module):
    """
    Global Context module.
    https://arxiv.org/pdf/1904.11492v1.pdf
    Args:
        dim (int): Dimension to be used for convolutional layers
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim):
        super(ContextNetGCModule, self).__init__()
        kernel_size = 41
        self.conv = nn.Conv1d(dim, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
#        self.softmax = nn.Softmax(dim=-1)
        self.conv2 = nn.Conv1d(dim, dim // 8, 1, bias=False)
        self.groupnorm = nn.GroupNorm(1, dim // 8)
        self.activation = Swish()
        self.conv3 = nn.Conv1d(dim // 8, dim, 1, bias=False)

    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        
        seq_lengths = x.size(2)
        context = self.conv(x)      # B x 1 x T
        context = context.sigmoid()
        context = torch.bmm(x, context.transpose(1, 2))     # B x C x 1
        
        transform = self.conv2(context)
        transform = self.groupnorm(transform)
        transform = self.activation(transform)
        transform = self.conv3(transform)

        transform = transform.repeat(1, 1, seq_lengths) 

        return transform * x


class ContextNetGCAModule(nn.Module):
    """
    Global Context module.
    https://arxiv.org/pdf/1904.11492v1.pdf
    Args:
        dim (int): Dimension to be used for convolutional layers
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim):
        super(ContextNetGCAModule, self).__init__()
        kernel_size = 41
        self.conv = nn.Conv1d(dim, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)


    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        
        seq_lengths = x.size(2)
        context = self.conv(x)      # B x 1 x T
        
        context = context.sigmoid()
        context = torch.bmm(x, context.transpose(1, 2))     # B x C x 1
        
        context = context.transpose(1, 2)

        transform = self.conv2(context)
        transform = transform.transpose(1, 2)
        transform = transform.sigmoid()

        transform = transform.repeat(1, 1, seq_lengths) 

        return transform * x

class ContextNetDAModule(nn.Module):
    """
    Channel Attention module of Dual Attention Network.
    https://arxiv.org/pdf/1809.02983
    Args:
        dim (int): Dimension to be used for convolutional layers *not used in here*
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim):
        super(ContextNetDAModule, self).__init__()

        # self.softmax = nn.Softmax(dim=-1)
        kernel_size = 41
        self.depth_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=dim, bias=False)
        self.point_conv = nn.Conv1d(dim, dim, kernel_size=1, padding=0, bias=False)


    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        residual = x
        
        G = torch.bmm(x, x.transpose(1,2))

        G = self.point_conv(self.depth_conv(G))

        G = G.sigmoid()

        GM = torch.bmm(G, residual)

        return GM

def conv1d_sample_by_sample(
    x: torch.Tensor,
    weight: torch.Tensor,
    oup: int,
    inp: int,
    ksize: int,
    stride: int,
    padding: int,
    groups: int,
) -> torch.Tensor:
    
    batch_size = x.shape[0]
    out = F.conv1d(
        x.view(1, -1, x.shape[2]),
        weight.view(batch_size * oup, inp, ksize),
        stride=stride,
        padding=padding,
        groups=groups * batch_size,
    )
    out = out.view(batch_size, oup, out.shape[2])

    return out

class ContextNetWeightModule(nn.Module):
    """
    Channel Attention module of Dual Attention Network.
    https://arxiv.org/pdf/1809.02983
    Args:
        dim (int): Dimension to be used for convolutional layers *not used in here*
    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(self, dim, ksize=5, stride=1, M=2, G=2):
        super(ContextNetWeightModule, self).__init__()

        self.dim = dim
        self.ksize = ksize
        self.stride = stride
        self.padding = (ksize - 1) // 2

        self.wn_fc1 = nn.Conv1d(dim, M * dim, 1, 1, 0, groups=1, bias=True)
        self.wn_fc2 = nn.Conv1d(
            M * dim, dim * dim * ksize, 1, 1, 0, groups=G * dim, bias=False
        )


    def forward(
            self,
            x,
            input_lengths
    ):
        """
        Forward propagate a `inputs` for SE Layer.
        Args:
            **inputs** (torch.FloatTensor): The output of the last convolution layer. `FloatTensor` of size
                ``(batch, dimension, seq_length), B x C x T``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of SELayer `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        """
        x_gap = x.sum(dim=2) / input_lengths.unsqueeze(1)
        x_w = self.wn_fc1(x_gap.unsqueeze(2))
        x_w = torch.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)

        out = conv1d_sample_by_sample(
            x, x_w, self.dim, self.dim, self.ksize, self.stride, self.padding, 1
        )

        return out


