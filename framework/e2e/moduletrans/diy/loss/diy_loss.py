#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
diy loss list
"""
import paddle


def naive_loss_list(logit, loss_list):
    """loss list"""
    for l in loss_list:
        logit = eval(l)
    return logit
