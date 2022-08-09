# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test lac model
"""

import os
import sys
import logging
import tarfile
import six
import wget
import pytest
import numpy as np

# pylint: disable=wrong-import-position
sys.path.append("..")
from test_case import InferenceTest, clip_model_extra_op

# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    lac_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.2.2/nlp/lac.tgz"
    if not os.path.exists("./lac/inference.pdiparams"):
        wget.download(lac_url, out="./")
        tar = tarfile.open("lac.tgz")
        tar.extractall()
        tar.close()
        clip_model_extra_op(path_prefix="./lac/inference", output_model_path="./lac/inference")


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./lac/inference.pdmodel", params_file="./lac/inference.pdiparams")
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.gpu
def test_gpu():
    """
    compared gpu lac outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./lac/inference.pdmodel", params_file="./lac/inference.pdiparams")
    in1 = np.random.randint(0, 100, (1, 20)).astype(np.int64)
    in2 = np.array([20])
    input_data_dict = {"token_ids": in1, "length": in2}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./lac/inference.pdmodel", params_file="./lac/inference.pdiparams")
    test_suite2.gpu_more_bz_test(input_data_dict, output_data_dict, delta=1e-5)

    del test_suite2  # destroy class to save memory
