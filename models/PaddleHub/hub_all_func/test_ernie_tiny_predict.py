#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ernie_tiny
"""
import os
import paddle

paddle.set_device("gpu")
import paddlehub as hub


def test_ernie_tiny_predict():
    """ernie_tiny"""
    os.system("hub install ernie_tiny")
    data = [
        ["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"],
        ["怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片"],
        ["作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。"],
    ]
    label_map = {0: "negative", 1: "positive"}

    model = hub.Module(
        name="ernie_tiny", version="2.0.2", task="seq-cls", load_checkpoint="/path/to/parameters", label_map=label_map
    )
    results = model.predict(data, max_seq_len=50, batch_size=1, use_gpu=False)
    for idx, text in enumerate(data):
        print("Data: {} \t Lable: {}".format(text, results[idx]))

    os.system("hub uninstall ernie_tiny")
