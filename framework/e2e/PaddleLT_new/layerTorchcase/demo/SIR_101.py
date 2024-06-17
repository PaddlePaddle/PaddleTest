#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
layer
"""

import torch
import numpy as np


class LayerCase(torch.nn.Module):
    """
    layercase
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,
    ):
        """
        forward
        """
        var_1 = torch.nn.functional.max_pool2d(var_0, kernel_size=3, stride=2, padding=1)
        return var_1


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (torch.rand([145, 24, 112, 112], dtype=torch.float32),)
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.random(size=[145, 24, 112, 112]).astype("float32"),)
    return inputs
