#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ocr rec_srn_head
"""
import copy
import numpy as np
import paddle
import ppdet

paddle.seed(33)
np.random.seed(33)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


roi_feat = paddle.to_tensor(randtool("float", -1, 1, shape=[4, 1024, 16, 16]).astype("float32"))
net = ppdet.modeling.backbones.senet.SERes5Head()
# net = paddle.jit.to_static(net)
print("net is: ", net.parameters())
# res = net(roi_feat=roi_feat)
#
# paddle.jit.save(net, path="SERes5Head")
