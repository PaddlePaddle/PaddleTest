#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
case runner
"""

import paddle
import numpy as np

seed = 33
np.random.seed(seed)
paddle.seed(seed)


class Runner(object):
    """Runner"""

    def __init__(self, reader, model, optimizer):
        """init"""
        self.reader = paddle.to_tensor(reader)
        self.model = model
        self.optimizer = optimizer
        self.debug = False
        self.result = []

    def run(self):
        """run your models"""
        for i in range(10):
            out = self.model(self.reader)
            loss = paddle.mean(out)
            loss.backward()
            self.optimizer.step()
            if self.debug:
                print(loss)
            self.result.append(loss.numpy()[0])

    def check(self, expect=None):
        """
        check result
        """
        if self.result is None:
            raise Exception("Model result is None， check your code")
        if self.debug:
            print(self.result)
        try:
            assert np.allclose(self.result, expect), "Error in check loss"
        except Exception as e:
            print(e)
            print("expect loss is {}".format(expect))
            print("Model result is {}".format(self.result))
            assert False
