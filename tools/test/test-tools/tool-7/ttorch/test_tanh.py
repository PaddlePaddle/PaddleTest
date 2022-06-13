#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import torch
import numpy as np


def test_tanh():
    x = torch.from_numpy(np.random.random([4, 4, 3, 3]))
    expect = np.tanh(x)
    res = torch.tanh(x)
    np.testing.assert_allclose(res, expect)
