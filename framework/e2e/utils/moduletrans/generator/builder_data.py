#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
data builder
"""
import paddle
import tool


class BuildData(object):
    """BuildData"""

    def __init__(self, data_loader, data):
        """init"""
        self.data_loader = data_loader
        self.data = data

    def get_single_numpy_data(self):
        """get data"""
        np_data = {}
        for k, v in self.data.items():
            value = tool._randtool(
                dtype=v["input"]["dtype"],
                low=v["input"]["range"][0],
                high=v["input"]["range"][1],
                shape=v["input"]["shape"],
            )
            np_data[k] = value
        return np_data

    def get_single_paddle_data(self):
        """get data"""
        paddle_data = {}
        for k, v in self.data.items():
            value = paddle.to_tensor(
                tool._randtool(
                    dtype=v["input"]["dtype"],
                    low=v["input"]["range"][0],
                    high=v["input"]["range"][1],
                    shape=v["input"]["shape"],
                )
            )
            paddle_data[k] = value
        return paddle_data
