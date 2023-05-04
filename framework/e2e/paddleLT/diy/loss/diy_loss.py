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


def mean_loss(logit):
    """base mean loss"""
    if isinstance(logit, list):
        tmp = 0.0
        for l in logit:
            mean = paddle.mean(l)
            tmp += mean
        loss = tmp / len(logit)
        return loss
    elif isinstance(logit, paddle.Tensor):
        loss = paddle.mean(logit)
        return loss
    else:
        raise Exception("something wrong with mean_loss!!")
