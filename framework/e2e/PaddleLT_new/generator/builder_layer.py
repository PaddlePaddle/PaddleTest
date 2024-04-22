#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
layer builder
"""

import os

if os.environ.get("FRAMEWORK") == "PaddlePaddle":
    import paddle
    import diy
    import layerApicase
    import layercase
elif os.environ.get("FRAMEWORK") == "PyTorch":
    import torch
    import layerTorchcase


class BuildLayer(object):
    """BuildLayer"""

    def __init__(self, layerfile):
        """init"""
        self.layername = layerfile + ".LayerCase"

    def get_layer(self):
        """get_layer"""
        layer = eval(self.layername)()
        return layer
