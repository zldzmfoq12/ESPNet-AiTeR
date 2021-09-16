#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
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

from typing import Tuple
from torch import Tensor
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetConvModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetDConvModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetSEModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetECAModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetWSEModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetLSEModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetBAMModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetGMModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetGCModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetDAModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetWeightModule
import torch.nn as nn


class ContextNetBlock(nn.Module):
    """
    Convolution block contains a number of convolutions, each followed by batch normalization, activation and dropout.
    Squeeze-and-excitation (SE) block operates on the output of the last convolution layer.
    Skip connection with projection is applied on the output of the squeeze-and-excitation block.
    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        num_layers (int, optional): The number of convolutional layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        dropout_rate (float, optional): Value of dropout_rate (default: 0.1)
        activation (nn.Function, optional): Activation function (default : Relu)
        residual (bool, optional): Flag indication residual or not (default : False)
    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution block `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output, output_lengths
        - **output**: Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            num_layers = 5,
            kernel_size = 5,
            stride = 1,
            padding = 0,
            dropout_rate = 0.1,
            activation = nn.ReLU(),
            se_type = "se",
            residual = False,
    ):
        super(ContextNetBlock, self).__init__()
        self.num_layers = num_layers
        
        
        self.residual = None

        if residual:
            self.residual = ContextNetConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout_rate=0,
                activation=activation,
                residual=True,
            )
            self.dropout = nn.Dropout(dropout_rate)
            self.activation = activation

        if self.num_layers == 1:
            self.conv_layers = ContextNetConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dropout_rate=dropout_rate,
                        activation=activation,
                    )

        else:
            stride_list = [1 for _ in range(num_layers - 1)] + [stride]
            in_channel_list = [in_channels] + [out_channels for _ in range(num_layers - 1)]

            self.conv_layers = nn.ModuleList(list())
            for in_channels, stride in zip(in_channel_list, stride_list):
                self.conv_layers.append(
                    ContextNetConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dropout_rate=dropout_rate,
                        activation=activation,
                    )
                )
            if se_type == "se":
                self.se_layer = ContextNetSEModule(out_channels, activation)
            elif se_type == "eca":      # don't work
                self.se_layer = ContextNetECAModule(out_channels, 2, 1)
            elif se_type == "wse":
                self.se_layer = ContextNetWSEModule(out_channels)
            elif se_type == "lse":
                self.se_layer = ContextNetLSEModule(out_channels)
            elif se_type == "bam":      # don't work
                self.se_layer = ContextNetBAMModule(out_channels, activation)
            elif se_type == "gm":       # don't work
                self.se_layer = ContextNetGMModule(out_channels)
            elif se_type == "gc":
                self.se_layer = ContextNetGCModule(out_channels)
            elif se_type == "da":
                self.se_layer = ContextNetDAModule(out_channels)
            elif se_type == "weight":
                self.se_layer = ContextNetWeightModule(out_channels, 5)
        

    def forward(
            self,
            inputs,
            input_lengths,
    ):
        """
        Forward propagate a `inputs` for convolution block.
        Args:
            **inputs** (torch.FloatTensor): Input of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        output = inputs
        output_lengths = input_lengths

        if self.num_layers == 1:
            output, output_lengths = self.conv_layers(output, output_lengths)
        else:
            for conv_layer in self.conv_layers:
                output, output_lengths = conv_layer(output, output_lengths)
            output = self.se_layer(output, output_lengths)

        if self.residual is not None:
            residual, _ = self.residual(inputs, input_lengths)
            output += residual
            output = self.dropout(self.activation(output))

        return output, output_lengths

    @staticmethod
    def make_conv_blocks(
            input_dim,
            num_layers = 5,
            kernel_size = 5,
            num_channels = 256,
            output_dim = 640,
            dropout_rate = 0.1,
            activation = nn.ReLU(),
            se_type = "se",
    ):
        """
        Create 23 convolution blocks.
        Args:
            input_dim (int, optional): Dimension of input vector 
            num_layers (int, optional): The number of convolutional layers (default : 5)
            kernel_size (int, optional): Value of convolution kernel size (default : 5)
            num_channels (int, optional): The number of channels in the convolution filter (default: 256)
            output_dim (int, optional): Dimension of encoder output vector (default: 640)
            dropout_rate (float): Dropout_rate of convolution block (default : 0.1)
            activation (nn.Function, optional): Activation function (default : Relu)
        Returns:
            **conv_blocks** (nn.ModuleList): ModuleList with 23 convolution blocks
        """
        conv_blocks = nn.ModuleList()

        # C0 : 1 conv layer, init_dim output channels, stride 1, no residual
        conv_blocks.append(ContextNetBlock(input_dim, num_channels, 1, kernel_size, 1, 0, dropout_rate, activation, se_type, False))

        # C1-2 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(1, 2 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, dropout_rate, activation, se_type, True))

        # C3 : 5 conv layer, init_dim output channels, stride 2
        conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, dropout_rate, activation, se_type, True))

        # C4-6 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(4, 6 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, dropout_rate, activation, se_type, True))

        # C7 : 5 conv layers, init_dim output channels, stride 2
        conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, dropout_rate, activation, se_type, True))

        # C8-10 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(8, 10 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, dropout_rate, activation, se_type, True))

        # C11-13 : 5 conv layers, middle_dim output channels, stride 1
        conv_blocks.append(ContextNetBlock(num_channels, num_channels << 1, num_layers, kernel_size, 1, 0, dropout_rate, activation, se_type, True))
        for _ in range(12, 13 + 1):
            conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, dropout_rate, activation, se_type, True))

        # C14 : 5 conv layers, middle_dim output channels, stride 2
        conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 2, 0, dropout_rate, activation, se_type, True))

        # C15-21 : 5 conv layers, middle_dim output channels, stride 1
        for i in range(15, 21 + 1):
            conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, dropout_rate, activation, se_type, True))

        # C22 : 1 conv layer, final_dim output channels, stride 1, no residual
        conv_blocks.append(ContextNetBlock(num_channels << 1, output_dim, 1, kernel_size, 1, 0, dropout_rate, activation, se_type, False))

        return conv_blocks


class DContextNetBlock(nn.Module):
    """
    Convolution block contains a number of convolutions, each followed by batch normalization, activation and dropout.
    Dynamic block operates ont on output of the last convolution layer.
    Squeeze-and-excitation (SE) block operates on the output of dynamic block.
    Skip connection with projection is applied on the output of the squeeze-and-excitation block.
    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        wshare (int): The number of kernel of convolution(the number of grouped channels) (default : 4)
        dropout_rate (float): Dropout_rate of dynamic convolution (default : 0.1)
        num_layers (int, optional): The number of convolutional layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        conv_dropout_rate (float): Dropout_rate of convolution block (default : 0.1)
        activation (nn.Function, optional): Activation function (default : Relu)
        residual (bool, optional): Flag indication residual or not (default : False)
    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution block `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output, output_lengths
        - **output**: Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            wshare = 4,
            dropout_rate = 0.1,
            num_layers = 5,
            kernel_size = 5,
            stride = 1,
            padding = 0,
            conv_dropout_rate = 0.1,
            activation = nn.ReLU(),
            se_type = "se",
            residual = False,
    ):
        super(DContextNetBlock, self).__init__()
        self.num_layers = num_layers
        
        self.residual = None

        if residual:
            self.residual = ContextNetConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout_rate=0,
                activation=activation,
                residual=True,
            )
            self.dropout = nn.Dropout(conv_dropout_rate)
            self.activation = activation

        if self.num_layers == 1:
            self.conv_layers = ContextNetConvModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dropout_rate=conv_dropout_rate,
                    activation=activation,
                )

        else:
            self.dynamic_layer = ContextNetDConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    wshare=wshare,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    activation=activation,
                    residual=True,
                )

            stride_list = [1 for _ in range(num_layers - 1)] + [stride]
            in_channel_list = [in_channels] + [out_channels for _ in range(num_layers - 1)]

            self.conv_layers = nn.ModuleList(list())
            for in_channels, stride in zip(in_channel_list, stride_list):
                self.conv_layers.append(
                    ContextNetConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dropout_rate=conv_dropout_rate,
                        activation=activation,
                    )
                )
            self.se_layer = ContextNetSEModule(out_channels, activation)
        

    def forward(
            self,
            inputs,
            input_lengths,
    ):
        """
        Forward propagate a `inputs` for convolution block.
        Args:
            **inputs** (torch.FloatTensor): Input of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        output = inputs
        output_lengths = input_lengths

        if self.num_layers == 1:
            output, output_lengths = self.conv_layers(output, output_lengths)
        else:
            for conv_layer in self.conv_layers:
                output, output_lengths = conv_layer(output, output_lengths)
            output, output_lengths = self.dynamic_layer(output, output_lengths)
            output = self.se_layer(output, output_lengths)

        if self.residual is not None:
            residual, _ = self.residual(inputs, input_lengths)
            output += residual
            output = self.dropout(self.activation(output))

        return output, output_lengths

    @staticmethod
    def make_conv_blocks(
            input_dim,
            wshare = 4,
            dropout_rate = 0.1,
            num_layers = 5,
            kernel_size = 5,
            num_channels = 256,
            output_dim = 640,
            conv_dropout_rate = 0.1,
            activation = nn.ReLU(),
            se_type = "se"
    ):
        """
        Create 23 convolution blocks.
            1 + 3k (k=0,1,...,6) blocks : convolution block -> dynamic block -> se block
            else blocks : convolution block -> se block
        Args:
            input_dim (int, optional): Dimension of input vector
            wshare (int): The number of kernel of convolution(the number of grouped channels) (default : 4)
            dropout_rate (float): Dropout_rate of dynamic convolution (default : 0.1)
            num_layers (int, optional): The number of convolutional layers (default : 5)
            kernel_size (int, optional): Value of convolution kernel size (default : 5)
            num_channels (int, optional): The number of channels in the convolution filter (default: 256)
            output_dim (int, optional): Dimension of encoder output vector (default: 640)
            conv_dropout_rate (float): Dropout_rate of convolution block (default : 0.1)
            activation (nn.Function, optional): Activation function (default : Relu)
        Returns:
            **conv_blocks** (nn.ModuleList): ModuleList with 23 convolution blocks
        """
        conv_blocks = nn.ModuleList()

        # C0 : 1 conv layer, init_dim output channels, stride 1, no residual
        conv_blocks.append(DContextNetBlock(input_dim, num_channels, wshare, dropout_rate, 1, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, False))

        # C1 : 5 conv layers, init_dim output channels, stride 1 with dynamic
        conv_blocks.append(DContextNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C2 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C3 : 5 conv layer, init_dim output channels, stride 2
        conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, conv_dropout_rate, activation, se_type, True))
       
        # C4 : 5 conv layers, init_dim output channels, stride 1 with dynamic
        conv_blocks.append(DContextNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C5-6 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(5, 6 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C7 : 5 conv layers, init_dim output channels, stride 2 with dynamic
        conv_blocks.append(DContextNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, kernel_size, 2, 0, conv_dropout_rate, activation, True))

        # C8-9 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(8, 9 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C10 : 5 conv layers, init_dim output channels, stride 1 with dynamic
        conv_blocks.append(DContextNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C11-12 : 5 conv layers, middle_dim output channels, stride 1
        conv_blocks.append(ContextNetBlock(num_channels, num_channels << 1, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))
        conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C13 : 5 conv layers, middle_dim output channels, stride 1 with dynamic
        conv_blocks.append(DContextNetBlock(num_channels << 1, num_channels << 1, wshare, dropout_rate, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C14 : 5 conv layers, middle_dim output channels, stride 2
        conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 2, 0, conv_dropout_rate, activation, se_type, True))

        # C15 : 5 conv layers, middle_dim output channels, stride 1
        conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C16 : 5 conv layers, middle_dim output channels, stride 1 with dynamic
        conv_blocks.append(DContextNetBlock(num_channels << 1, num_channels << 1, wshare, dropout_rate, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C17-18 : 5 conv layers, middle_dim output channels, stride 1
        for i in range(17, 18 + 1):
            conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))
        
        # C19 : 5 conv layers, middle_dim output channels, stride 1 with dynamic
        conv_blocks.append(DContextNetBlock(num_channels << 1, num_channels << 1, wshare, dropout_rate, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))

         # C20-21 : 5 conv layers, middle_dim output channels, stride 1
        for i in range(20, 21 + 1):
            conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, True))
        
        # C22 : 1 conv layer, final_dim output channels, stride 1, no residual
        conv_blocks.append(DContextNetBlock(num_channels << 1, output_dim, wshare, dropout_rate, 1, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, False))

        return conv_blocks



class CitriNetBlock(nn.Module):
    """
    Convolution block contains a number of convolutions, each followed by batch normalization, activation and dropout.
    Squeeze-and-excitation (SE) block operates on the output of the last convolution layer.
    Skip connection with projection is applied on the output of the squeeze-and-excitation block.
    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        num_layers (int, optional): The number of convolutional layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        dropout_rate (float, optional): Value of dropout_rate (default: 0.1)
        activation (nn.Function, optional): Activation function (default : Relu)
        residual (bool, optional): Flag indication residual or not (default : False)
    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution block `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output, output_lengths
        - **output**: Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            num_layers = 5,
            kernel_size = 5,
            stride = 1,
            padding = 0,
            dropout_rate = 0.1,
            activation = nn.ReLU(),
            se_type = "se",
            residual = False,
    ):
        super(CitriNetBlock, self).__init__()
        self.num_layers = num_layers
        
        self.residual = None

        if residual:
            self.residual = ContextNetConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout_rate=0,
                activation=activation,
                residual=True,
            )
            self.dropout = nn.Dropout(dropout_rate)
            self.activation = activation

        if self.num_layers == 1:
            self.conv_layers = ContextNetConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dropout_rate=dropout_rate,
                        activation=activation,
                    )

        else:
            stride_list = [1 for _ in range(num_layers - 1)] + [stride]
            in_channel_list = [in_channels] + [out_channels for _ in range(num_layers - 1)]

            self.conv_layers = nn.ModuleList(list())
            for in_channels, stride in zip(in_channel_list, stride_list):
                self.conv_layers.append(
                    ContextNetConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dropout_rate=dropout_rate,
                        activation=activation,
                    )
                )
            if se_type == "se":
                self.se_layer = ContextNetSEModule(out_channels, self.activation)
            elif se_type == "eca":      #work    
                self.se_layer = ContextNetECAModule(out_channels, kernel_size)
            elif se_type == "wse":
                self.se_layer = ContextNetWSEModule(out_channels)
            elif se_type == "lse":
                self.se_layer = ContextNetLSEModule(out_channels)
            elif se_type == "bam":      # don't work
                self.se_layer = ContextNetBAMModule(out_channels, self.activation)
            elif se_type == "gm":       # don't work
                self.se_layer = ContextNetGMModule(out_channels)
            elif se_type == "gc":       # work
                self.se_layer = ContextNetGCModule(out_channels, kernel_size)
            elif se_type == "da":       # don't work
                self.se_layer = ContextNetDAModule(out_channels)
            elif se_type == "weight":   # don't work
                self.se_layer = ContextNetWeightModule(out_channels, 3)
        

    def forward(
            self,
            inputs,
            input_lengths,
    ):
        """
        Forward propagate a `inputs` for convolution block.
        Args:
            **inputs** (torch.FloatTensor): Input of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        output = inputs
        output_lengths = input_lengths

        if self.num_layers == 1:
            output, output_lengths = self.conv_layers(output, output_lengths)
        else:
            for conv_layer in self.conv_layers:
                output, output_lengths = conv_layer(output, output_lengths)
            output = self.se_layer(output, output_lengths)

        if self.residual is not None:
            residual, _ = self.residual(inputs, input_lengths)
            output += residual
            output = self.dropout(self.activation(output))

        return output, output_lengths

    @staticmethod
    def make_conv_blocks(
            input_dim,
            num_layers = 5,
            kernel_size = 5,
            num_channels = 384,
            output_dim = 384,
            dropout_rate = 0.1,
            activation = nn.ReLU(),
            se_type = "se",
    ):
        """
        Create 23 convolution blocks.
        Args:
            input_dim (int, optional): Dimension of input vector 
            num_layers (int, optional): The number of convolutional layers (default : 5)
            kernel_size (int, optional): Value of convolution kernel size (default : 5)
            num_channels (int, optional): The number of channels in the convolution filter (default: 256)
            output_dim (int, optional): Dimension of encoder output vector (default: 640)
            dropout_rate (float): Dropout_rate of convolution block (default : 0.1)
            activation (nn.Function, optional): Activation function (default : Relu)
        Returns:
            **conv_blocks** (nn.ModuleList): ModuleList with 23 convolution blocks
        """
        conv_blocks = nn.ModuleList()

        # C0 : 1 conv layer, init_dim output channels, stride 1, no residual
        conv_blocks.append(CitriNetBlock(input_dim, num_channels, 1, kernel_size, 1, 0, dropout_rate, activation, se_type, False))

        # C1 : 5 conv layers, init_dim output channels, stride 2
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, dropout_rate, activation, se_type, True))

        # C2-3 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(2, 3 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 7, 1, 0, dropout_rate, activation, se_type, True))

        # C4-5 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(4, 5 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 9, 1, 0, dropout_rate, activation, se_type, True))

        # C6 : 5 conv layer, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 11, 1, 0, dropout_rate, activation, se_type, True))
        
        # C7 : 5 conv layers, init_dim output channels, stride 2
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 7, 2, 0, dropout_rate, activation, se_type, True))

        # C8 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 7, 1, 0, dropout_rate, activation, se_type, True))

        # C9-10 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(9, 10 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 9, 1, 0, dropout_rate, activation, se_type, True))

        # C11-12 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(11, 12 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 11, 1, 0, dropout_rate, activation, se_type, True))

        # C13 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 13, 1, 0, dropout_rate, activation, se_type, True))

        # C14 : 5 conv layers, init_dim output channels, stride 2
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 13, 2, 0, dropout_rate, activation, se_type, True))

        # C15 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 13, 1, 0, dropout_rate, activation, se_type, True))
        
        # C16-17 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(16, 17 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 15, 1, 0, dropout_rate, activation, se_type, True))
        
        # C18-19 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(18, 19 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 17, 1, 0, dropout_rate, activation, se_type, True))
        
        # C20-21 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(20, 21 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 19, 1, 0, dropout_rate, activation, se_type, True))
        
        # C22 : 1 conv layer, final_dim output channels, stride 1, no residual
        conv_blocks.append(CitriNetBlock(num_channels, output_dim, 1, 41, 1, 0, dropout_rate, activation, se_type, False))

        return conv_blocks


class DCitriNetBlock(nn.Module):
    """
    Convolution block contains a number of convolutions, each followed by batch normalization and activation.
    Squeeze-and-excitation (SE) block operates on the output of the last convolution layer.
    Skip connection with projection is applied on the output of the squeeze-and-excitation block.
    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        wshare (int): The number of kernel of convolution(the number of grouped channels) (default : 4)
        dropout_rate (float): Dropout_rate of dynamic convolution (default : 0.1)
        num_layers (int, optional): The number of convolutional layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        conv_dropout_rate (float): Dropout_rate of convolution block (default : 0.1)
        activation (nn.Function, optional): Activation function (default : Relu)
        residual (bool, optional): Flag indication residual or not (default : False)
    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution block `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``
    Returns: output, output_lengths
        - **output**: Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            wshare = 4,
            dropout_rate = 0.1,
            num_layers = 5,
            kernel_size = 5,
            stride = 1,
            padding = 0,
            conv_dropout_rate = 0.1,
            activation = nn.ReLU(),
            se_type = "se",
            residual = False,
    ):
        super(DCitriNetBlock, self).__init__()
        self.num_layers = num_layers
        
        self.residual = None

        if residual:
            self.residual = ContextNetConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout_rate=0,
                activation=activation,
                residual=True,
            )
            self.dropout = nn.Dropout(conv_dropout_rate)
            self.activation = activation

        if self.num_layers == 1:
            self.conv_layers = ContextNetConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dropout_rate=conv_dropout_rate,
                        activation=activation,
                    )

        else:
            self.dynamic_layer = ContextNetDConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    wshare=wshare,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    activation=activation,
                    residual=True,
                )

            stride_list = [1 for _ in range(num_layers - 1)] + [stride]
            in_channel_list = [in_channels] + [out_channels for _ in range(num_layers - 1)]

            self.conv_layers = nn.ModuleList(list())
            for in_channels, stride in zip(in_channel_list, stride_list):
                self.conv_layers.append(
                    ContextNetConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dropout_rate=conv_dropout_rate,
                        activation=activation,
                    )
                )
            if se_type == "se":
                self.se_layer = ContextNetSEModule(out_channels, self.activation)
            elif se_type == "eca":
                self.se_layer = ContextNetECAModule(out_channels, 2, 1)
            elif se_type == "wse":
                self.se_layer = ContextNetWSEModule(out_channels)
            elif se_type == "lse":
                self.se_layer = ContextNetLSEModule(out_channels)
            elif se_type == "bam":
                self.se_layer = ContextNetBAMModule(out_channels)
            elif se_type == "gm":
                self.se_layer = ContextNetGMModule(out_channels)
        

    def forward(
            self,
            inputs,
            input_lengths,
    ):
        """
        Forward propagate a `inputs` for convolution block.
        Args:
            **inputs** (torch.FloatTensor): Input of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **output** (torch.FloatTensor): Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        output = inputs
        output_lengths = input_lengths

        if self.num_layers == 1:
            output, output_lengths = self.conv_layers(output, output_lengths)
        else:
            for conv_layer in self.conv_layers:
                output, output_lengths = conv_layer(output, output_lengths)
            output, output_lengths = self.dynamic_layer(output, output_lengths)
            output = self.se_layer(output, output_lengths)

        if self.residual is not None:
            residual, _ = self.residual(inputs, input_lengths)
            output += residual
            output = self.dropout(self.activation(output))

        return output, output_lengths

    @staticmethod
    def make_conv_blocks(
            input_dim,
            wshare = 4,
            dropout_rate = 0.1,
            num_layers = 5,
            kernel_size = 5,
            num_channels = 384,
            output_dim = 384,
            conv_dropout_rate = 0.1,
            activation = nn.ReLU(),
            se_type = "se",
    ):
        """
        Create 23 convolution blocks.
            1 + 3k (k=0,1,...,6) blocks : convolution block -> dynamic block -> se block
            else blocks : convolution block -> se block
        Args:
            input_dim (int, optional): Dimension of input vector
            wshare (int): The number of kernel of convolution(the number of grouped channels) (default : 4)
            dropout_rate (float): Dropout_rate of dynamic convolution (default : 0.1)
            num_layers (int, optional): The number of convolutional layers (default : 5)
            kernel_size (int, optional): Value of convolution kernel size (default : 5)
            num_channels (int, optional): The number of channels in the convolution filter (default: 256)
            output_dim (int, optional): Dimension of encoder output vector (default: 640)
            conv_dropout_rate (float): Dropout_rate of convolution block (default : 0.1)
            activation (nn.Function, optional): Activation function (default : Relu)
        Returns:
            **conv_blocks** (nn.ModuleList): ModuleList with 23 convolution blocks
        """
        conv_blocks = nn.ModuleList()

        # C0 : 1 conv layer, init_dim output channels, stride 1, no residual
        conv_blocks.append(CitriNetBlock(input_dim, num_channels, 1, kernel_size, 1, 0, conv_dropout_rate, activation, se_type, False))

        # C1 : 5 conv layers, init_dim output channels, stride 2 with dynamic
        conv_blocks.append(DCitriNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, kernel_size, 2, 0, conv_dropout_rate, activation, se_type, True))

        # C2-3 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(2, 3 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 7, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C4 : 5 conv layers, init_dim output channels, stride 1 with dynamic
        conv_blocks.append(DCitriNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, 9, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C5 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 9, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C6 : 5 conv layer, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 11, 1, 0, conv_dropout_rate, activation, se_type, True))
        
        # C7 : 5 conv layers, init_dim output channels, stride 2 with dynamic
        conv_blocks.append(DCitriNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, 7, 2, 0, conv_dropout_rate, activation, se_type, True))

        # C8 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 7, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C9 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 9, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C10 : 5 conv layers, init_dim output channels, stride 1 with dynamic
        conv_blocks.append(DCitriNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, 9, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C11-12 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(11, 12 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 11, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C13 : 5 conv layers, init_dim output channels, stride 1 with dynamic
        conv_blocks.append(DCitriNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, 13, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C14 : 5 conv layers, init_dim output channels, stride 2
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 13, 2, 0, conv_dropout_rate, activation, se_type, True))

        # C15 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 13, 1, 0, conv_dropout_rate, activation, se_type, True))
        
        # C16 : 5 conv layers, init_dim output channels, stride 1 with dynamic
        conv_blocks.append(DCitriNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, 13, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C17 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 15, 1, 0, conv_dropout_rate, activation, se_type, True))
        
        # C18 : 5 conv layers, init_dim output channels, stride 1
        conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 17, 1, 0, conv_dropout_rate, activation, se_type, True))        
        
        # C19 : 5 conv layers, init_dim output channels, stride 1 with dynamic
        conv_blocks.append(DCitriNetBlock(num_channels, num_channels, wshare, dropout_rate, num_layers, 17, 1, 0, conv_dropout_rate, activation, se_type, True))

        # C20-21 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(20, 21 + 1):
            conv_blocks.append(CitriNetBlock(num_channels, num_channels, num_layers, 19, 1, 0, conv_dropout_rate, activation, se_type, True))
        
        # C22 : 1 conv layer, final_dim output channels, stride 1, no residual
        conv_blocks.append(CitriNetBlock(num_channels, output_dim, 1, 41, 1, 0, conv_dropout_rate, activation, se_type, False))

        return conv_blocks



