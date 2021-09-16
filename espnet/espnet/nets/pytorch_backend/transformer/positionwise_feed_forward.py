#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch
from torch import nn

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SpinalFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(SpinalFeedForward, self).__init__()
        self.Half_width = idim//2
        self.layer_width = hidden_units//4

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward funciton."""
        x1 = self.fc_spinal_layer1(x[:,:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,:,self.Half_width:2*self.Half_width], x1], dim=2))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,:,0:self.Half_width], x2], dim=2))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,:,self.Half_width:2*self.Half_width], x3], dim=2))

        x = torch.cat([x1, x2], dim=2)
        x = torch.cat([x, x3], dim=2)
        x = torch.cat([x, x4], dim=2)

        return self.w_2(self.dropout(self.activation(x)))


class SpinalFeedForward8(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(SpinalFeedForward8, self).__init__()
        self.Half_width = idim//2
        self.layer_width = hidden_units//8

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer5 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer6 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer7 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.fc_spinal_layer8 = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
            )
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward funciton."""
        x1 = self.fc_spinal_layer1(x[:,:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,:,self.Half_width:2*self.Half_width], x1], dim=2))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,:,0:self.Half_width], x2], dim=2))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,:,self.Half_width:2*self.Half_width], x3], dim=2))
        x5 = self.fc_spinal_layer5(torch.cat([ x[:,:,0:self.Half_width], x4], dim=2))
        x6 = self.fc_spinal_layer6(torch.cat([ x[:,:,self.Half_width:2*self.Half_width], x5], dim=2))
        x7 = self.fc_spinal_layer7(torch.cat([ x[:,:,0:self.Half_width], x6], dim=2))
        x8 = self.fc_spinal_layer8(torch.cat([ x[:,:,self.Half_width:2*self.Half_width], x7], dim=2))

        x = torch.cat([x1, x2], dim=2)
        x = torch.cat([x, x3], dim=2)
        x = torch.cat([x, x4], dim=2)
        x = torch.cat([x, x5], dim=2)
        x = torch.cat([x, x6], dim=2)
        x = torch.cat([x, x7], dim=2)
        x = torch.cat([x, x8], dim=2)

        return self.w_2(self.dropout(self.activation(x)))
