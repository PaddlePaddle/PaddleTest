#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import tensorflow as tf
import numpy as np


def test_tanh():
    x = tf.convert_to_tensor(np.random.random([4, 4, 3, 3]))
    expect = np.tanh(x)
    res = tf.nn.tanh(x)
    np.testing.assert_allclose(res, expect)
