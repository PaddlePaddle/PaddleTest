#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
常用tools
"""
import numpy as np
import paddle


def reset(seed):
    """
    重置模型图
    :param seed: 随机种子
    :return:
    """
    paddle.enable_static()
    paddle.disable_static()
    paddle.seed(seed)
    np.random.seed(seed)
