#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import tensorflow as tf
import numpy as np
import time


total_time = 0
for i in range(1000):
    x = np.random.random([33, 100, 100])
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        start_time = time.time()
        res = tf.nn.relu(x)
        res = tf.reduce_mean(res)
    tape.gradient(res, x)
    end_time = time.time()
    total_time += end_time-start_time

print("cost time: " + str(total_time/1000))
