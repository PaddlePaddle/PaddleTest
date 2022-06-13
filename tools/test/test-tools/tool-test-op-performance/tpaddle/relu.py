#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import paddle
import numpy as np
import time


total_time = 0
for i in range(1000):
    x = np.random.random([33, 100, 100])
    x = paddle.to_tensor(x)
    x.stop_gradient = False
    relu = paddle.nn.ReLU()
    start_time = time.time()
    res = relu(x)
    res = paddle.mean(res)
    res.backward()
    end_time = time.time()
    total_time += end_time-start_time

print("cost time: " + str(total_time/1000))
