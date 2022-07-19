#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
diy optimizer
"""
import paddle


def naive_opt(net, opt_api, learning_rate):
    """navie optimizer func"""
    opt = eval(opt_api)(learning_rate=learning_rate, parameters=net.parameters())
    return opt
