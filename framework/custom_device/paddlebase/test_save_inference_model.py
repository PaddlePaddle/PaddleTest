#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_save_inference_model
"""

import os
import paddle

import paddle.static as static


def test_save_inference_model():
    """save no params model"""
    pwd = os.getcwd()
    pdmodel_path = os.path.join(pwd, "save_inference_model_temp")
    paddle.enable_static()
    x = static.data(name="x", shape=[10, 10], dtype="float32")
    y = x + x

    place = paddle.CPUPlace()
    exe = static.Executor(place)
    exe.run(static.default_startup_program())
    prog = static.default_main_program()
    static.save_inference_model(pdmodel_path, [x], [y], exe)
    [inference_program, feed_target_names, fetch_targets] = static.load_inference_model(pdmodel_path, exe)

    print(inference_program)
    print(feed_target_names)
    print(fetch_targets)
    print(prog)
