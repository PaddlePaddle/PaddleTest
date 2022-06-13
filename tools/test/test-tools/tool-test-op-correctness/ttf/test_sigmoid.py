#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import tensorflow as tf
import numpy as np


def test_sigmoid():
    x = tf.convert_to_tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    expect = np.array([0.7310586, 0.880797, 0.95257413, 0.98201376])
    res = tf.nn.sigmoid(x)
    np.testing.assert_allclose(res, expect)
