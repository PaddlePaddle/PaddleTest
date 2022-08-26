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


def main():
    """main"""
    input = {"image": paddle.to_tensor(randtool("float", -1, 1, shape=[4, 3, 224, 224]).astype("float32"))}
    net = ppdet.modeling.backbones.cspresnet.CSPResNet()
    # net = paddle.jit.to_static(net)
    print(net.out_shape)

    net(inputs=input)

    # paddle.jit.save(net, path='CSPResNet')


main()
