#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
linear
"""
import paddle


class Linear(paddle.nn.Layer):
    """Linear"""

    def __init__(self, in_channels=10, pvam_ch=10, char_num=3):
        """init"""
        super(Linear, self).__init__()
        self.char_num = char_num
        self.fc0 = paddle.nn.Linear(in_features=in_channels, out_features=pvam_ch)
        self.fc1 = paddle.nn.Linear(in_features=pvam_ch, out_features=self.char_num)

    def forward(self, pvam_feature):
        """forward"""
        x = self.fc0(pvam_feature)
        out = self.fc1(x)
        return out
