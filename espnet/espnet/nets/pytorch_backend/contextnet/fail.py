from torch import Tensor
from typing import Tuple
from torch import nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
import torch
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetSEModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetConvModule
from espnet.nets.pytorch_backend.contextnet.convolution import ContextNetDConvModule

class ContextNetCConvModule(nn.Module):
    """
    --------------------------------------------------------------------------------
    This is a convolution module inspired by conformers.
    --------------------------------------------------------------------------------
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
        super(ContextNetCConvModule, self).__init__()

        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
            padding=(kernel_size - 1) // 2,
            groups=in_channels,
            bias=bias,
        )

        self.pointwise_conv2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        
        if residual==False:
            self.activation = activation
            self.dropout = nn.Dropout(dropout_rate)
            self.norm = nn.BatchNorm1d(in_channels)

            self.pointwise_conv1 = nn.Conv1d(
                in_channels,
                in_channels * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
        else:
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
            x = self.pointwise_conv1(x)  # (batch, 2*channel, time)
            x = nn.functional.glu(x, dim=1)
            x, output_lengths = self.depthwise_conv(x), self._get_sequence_lengths(input_lengths)
            x = self.norm(x)                    
            x = self.activation(x)
            x = self.pointwise_conv2(x)
            x = self.dropout(x)
        
        else:
            x, output_lengths = self.pointwise_conv2(self.depthwise_conv(x)), self._get_sequence_lengths(input_lengths)
            x = self.norm(x)

        return x, output_lengths

    def _get_sequence_lengths(self, seq_lengths):
        return (
                (seq_lengths + 2 * self.depthwise_conv.padding[0]
                 - self.depthwise_conv.kernel_size[0]) // self.depthwise_conv.stride[0] + 1
        )


class CContextNetBlock(nn.Module):
    """
    Convolution block contains a number of convolutions inspired by conformer, each followed by batch normalization, activation and dropout.
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
            residual = False,
    ):
        super(CContextNetBlock, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.se_layer = ContextNetSEModule(out_channels, activation)
        self.residual = None

        if residual:
            self.residual = ContextNetCConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dropout_rate=0,
                activation=activation,
                residual=True,
            )

        if self.num_layers == 1:
            self.conv_layers = ContextNetCConvModule(
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
                    ContextNetCConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dropout_rate=dropout_rate,
                        activation=activation,
                    )
                )
        self.dropout = nn.Dropout(dropout_rate)

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
        conv_blocks.append(CContextNetBlock(input_dim, num_channels, 1, kernel_size, 1, 0, dropout_rate, activation, False))

        # C1-2 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(1, 2 + 1):
            conv_blocks.append(CContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, dropout_rate, activation, True))

        # C3 : 5 conv layer, init_dim output channels, stride 2
        conv_blocks.append(CContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, dropout_rate, activation, True))

        # C4-6 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(4, 6 + 1):
            conv_blocks.append(CContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, dropout_rate, activation, True))

        # C7 : 5 conv layers, init_dim output channels, stride 2
        conv_blocks.append(CContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, dropout_rate, activation, True))

        # C8-10 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(8, 10 + 1):
            conv_blocks.append(CContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, dropout_rate, activation, True))

        # C11-13 : 5 conv layers, middle_dim output channels, stride 1
        conv_blocks.append(CContextNetBlock(num_channels, num_channels << 1, num_layers, kernel_size, 1, 0, dropout_rate, activation, True))
        for _ in range(12, 13 + 1):
            conv_blocks.append(CContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, dropout_rate, activation, True))

        # C14 : 5 conv layers, middle_dim output channels, stride 2
        conv_blocks.append(CContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 2, 0, dropout_rate, activation, True))

        # C15-21 : 5 conv layers, middle_dim output channels, stride 1
        for i in range(15, 21 + 1):
            conv_blocks.append(CContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, dropout_rate, activation, True))

        # C22 : 1 conv layer, final_dim output channels, stride 1, no residual
        conv_blocks.append(CContextNetBlock(num_channels << 1, output_dim, 1, kernel_size, 1, 0, dropout_rate, activation, False))

        return conv_blocks
