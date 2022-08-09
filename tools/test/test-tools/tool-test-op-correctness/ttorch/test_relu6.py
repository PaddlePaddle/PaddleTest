#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import torch
import numpy as np


def test_relu6():
    x = torch.from_numpy(np.random.random([4, 4, 3, 3]))
    expect = np.minimum(np.maximum(0, x), 6)
    res = torch.nn.functional.relu6(x)
    np.testing.assert_allclose(res, expect)
