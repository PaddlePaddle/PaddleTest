#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
aaa
"""

import paddlelite.lite as lite

a = lite.Opt()
# 非 combined 形式
a.set_model_dir("mobilenet_v3_ConvBNLayer")

# conmbined 形式，具体模型和参数名称，请根据实际修改
# a.set_model_file("D:\\YOU_MODEL_PATH\\mobilenet_v1\\__model__")
# a.set_param_file("D:\\YOU_MODEL_PATH\\mobilenet_v1\\__params__")

a.set_optimize_out("ConvBNLayer_opt")
a.set_valid_places("arm")

a.run()
