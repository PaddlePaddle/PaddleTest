# encoding: utf-8
"""
paddlecv 测试case
"""


import subprocess
import re
import sys
import platform
import os.path
import yaml
import pytest
import allure
import numpy as np

from PaddleCVTestFramwork import prepare_repo
from PaddleCVTestFramwork import TestPaddleCVPredict


def get_model_list(filename="utils/models_list/paddlecv_list.txt"):
    """
    get_model_list
    """
    result = []
    with open(filename, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            result.append(line.strip("\n"))
    return result


def setup_module():
    """
    setup_modul
    """
    prepare_repo()


@allure.story("paddlecv_gpu_predict")
@pytest.mark.parametrize("model_name", get_model_list())
@pytest.mark.parametrize("run_mode", ["paddle", "trt_fp32", "trt_fp16", "trt_int8"])
def test_paddlecv_gpu_predict(model_name, run_mode):
    """
    test_paddlecv_gpu_predict
    """
    allure.dynamic.title(model_name + "_GPU_code_predict_" + run_mode)
    allure.dynamic.description("GPU_code_predict")
    model = TestPaddleCVPredict(model=model_name)
    model.test_cv_predict(run_mode, "GPU")


@allure.story("paddlecv_cpu_predict")
@pytest.mark.parametrize("model_name", get_model_list())
@pytest.mark.parametrize("run_mode", ["paddle", "mkldnn"])
def test_paddlecv_cpu_predict(model_name, run_mode):
    """
    test_paddlecv_cpu_predict
    """
    allure.dynamic.title(model_name + "_CPU_code_predict_" + run_mode)
    allure.dynamic.description("CPU_code_predict")
    model = TestPaddleCVPredict(model=model_name)
    model.test_cv_predict(run_mode, "CPU")


@allure.story("paddlecv_wheel_predict")
@pytest.mark.parametrize("model_name", get_model_list("utils/models_list/paddlecv_list_wheel.txt"))
def u_test_paddlecv_wheel_predict(model_name):
    """
    test_paddlecv_wheel_predic
    """
    allure.dynamic.title(model_name + "_wheel_predict")
    allure.dynamic.description("wheel_predict")
    model = TestPaddleCVPredict(model=model_name)
    model.test_wheel_predict()
