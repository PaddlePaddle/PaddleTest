# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test fast_rcnn model
"""

import os
import pytest
import numpy as np

from benchmark import *
from util import *


# pylint: enable=wrong-import-position
@pytest.mark.server
def test_resnet50_xpu():
    """
    compared gpu fast_rcnn batch_size = [1] outputs with true val
    """
    args = parse_args()
    args.enable_gpu = False
    args.gpu_id = 0
    args.backend_type = "XPU"
    args.batch_size = 1
    args.precision = "fp32"
    args.model_dir = "./Models/ResNet50"
    args.return_result = True
    args.paddle_model_file, args.paddle_params_file = get_model_file(args.model_dir)
    # print(args)

    runner = BenchmarkRunner(args)
    runner.test(args)
    # print(runner.result)
    # print(runner.result["out_diff"]["sum_diff_less_0.01"])
    assert runner.result["out_diff"]["sum_diff_less_0.01"] > 98
