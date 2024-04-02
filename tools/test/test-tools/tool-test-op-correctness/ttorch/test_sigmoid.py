#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import torch
import numpy as np


def test_sigmoid():
    x = torch.from_numpy(np.array([1.0, 2.0, 3.0, 4.0]).astype('float32'))
    expect = np.array([0.7310586, 0.880797, 0.95257413, 0.98201376])
    res = torch.sigmoid(x)
    np.testing.assert_allclose(res, expect)
