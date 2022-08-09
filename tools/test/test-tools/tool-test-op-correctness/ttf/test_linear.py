#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import tensorflow as tf
import numpy as np


def test_linear():
    # no linear in tf
    x = tf.convert_to_tensor(np.ones(shape=[3, 5]))
    pass
