# Copyright 2020 Tomoki Hayashi
#
# Copyright (c) 2021 Seunghun Jeong
#
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ContextNet encoder definition."""

from typing import Optional
from typing import Tuple

import logging
import torch

import torch.nn as nn

from typeguard import check_argument_types
from espnet.nets.pytorch_backend.contextnet.contextnet_block import ContextNetBlock
from espnet.nets.pytorch_backend.contextnet.contextnet_block import DContextNetBlock
from espnet.nets.pytorch_backend.contextnet.contextnet_block import CitriNetBlock
from espnet.nets.pytorch_backend.contextnet.contextnet_block import DCitriNetBlock
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt

from espnet.nets.pytorch_backend.nets_utils import get_activation

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class ContextnetEncoder(AbsEncoder):
    """Contextnet encoder module.

    Args:
        input_size (int): Size of input vector
        model_size (str, optional): Size of the model['small', 'medium', 'large'] (default : 'medium')
        output_size (int, optional): Size of encoder output vector (default: 640)
        num_layers (int, optional): The number of convolutional layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        num_channels (int, optional): The number of channels in the convolution filter (default: 256)
        activation_type (str): Encoder activation function type.
        conv_layer_type (str) : Encoder conv layer type.
        wshare (int): the number of kernel of convolution (default : 4)
        dropout_rate (float): conv dropout rate

    Inputs: inputs, input_lengths
        - **inputs**: Parsed audio of batch size number `FloatTensor` of size ``(batch, seq_length, dimension)``
        - **input_lengths**: Tensor representing the sequence length of the input ``(batch)``
    Returns: output, output_lengths
        - **output**: Tensor of encoder output `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        - **output_lengths**: Tensor representing the length of the encoder output ``(batch)``
    """
    supported_models = {
        'small': 0.5,
        'medium': 1,
        'large': 2,
    }

    def __init__(
        self,
        input_size: int,
        model_size: str = 'medium',
        output_size: int = 640,
        num_layers: int = 5,
        kernel_size: int = 5,
        num_channels: int = 256,
        dropout_rate: float = 0.1,
        activation_type: str = "swish",
        conv_layer_type: str = "base",
        wshare: int = 4,
        dy_dropout_rate: float = 0.0,
        se_type: str = "se",
    ):
        assert check_argument_types()
        super().__init__()

        alpha = self.supported_models[model_size]

        num_channels = int(num_channels * alpha)
        output_size = int(output_size * alpha)
        self._output_size = output_size

        activation = get_activation(activation_type)

        if conv_layer_type == "base":
            self.blocks = ContextNetBlock.make_conv_blocks(input_size, num_layers, kernel_size, num_channels, output_size, dropout_rate, activation, se_type)
        elif conv_layer_type == "dynamic":
            self.blocks = DContextNetBlock.make_conv_blocks(input_size, wshare, dy_dropout_rate, num_layers, kernel_size, num_channels, output_size, dropout_rate, activation, se_type)
        elif conv_layer_type == "citrinet":
            self.blocks = CitriNetBlock.make_conv_blocks(input_size, num_layers, kernel_size, num_channels, output_size, dropout_rate, activation, se_type)
        elif conv_layer_type == "dynamic_citrinet":
            self.blocks = DCitriNetBlock.make_conv_blocks(input_size, wshare, dy_dropout_rate, num_layers, kernel_size, num_channels, output_size, dropout_rate, activation, se_type)
        else:
            raise NotImplementedError("Support only base or dynamic.")


    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        limit_size = 15

        if xs_pad.size(1) < 15 :
            raise TooShortUttError(
                f"has {xs_pad.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {limit_size} frames), return empty results",
                xs_pad.size(1),
                limit_size,
            )

        output = xs_pad.transpose(1, 2)
        output_lengths = ilens

        for block in self.blocks:
            output, output_lengths = block(output, output_lengths)

        output = output.transpose(1, 2)

        

        return output, output_lengths, None
