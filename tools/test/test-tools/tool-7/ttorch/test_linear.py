#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import torch
import numpy as np


def test_linear():
    x = torch.from_numpy(np.ones(shape=[3, 5]))
    weight = torch.from_numpy(np.ones(shape=[3, 5])*3)
    bias = torch.from_numpy(np.ones(shape=[3]))
    res = torch.nn.functional.linear(x, weight, bias)
    expect = np.ones(shape=[3, 3]) * 16
    np.testing.assert_allclose(res, expect)
