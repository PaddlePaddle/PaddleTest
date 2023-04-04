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
from test_case import InferenceTest

# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    lac_url = "https://paddle-qa.bj.bcebos.com/inference_model_clipped/2.2.2/nlp/lac.tgz"
    if not os.path.exists("./lac/inference.pdiparams"):
        wget.download(lac_url, out="./")
        tar = tarfile.open("lac.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./lac/inference.pdmodel",
        params_file="./lac/inference.pdiparams",
    )
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
    test_suite.load_config(
        model_file="./lac/inference.pdmodel",
        params_file="./lac/inference.pdiparams",
    )
    in1 = np.random.randint(0, 100, (1, 20)).astype(np.int64)
    in2 = np.array([20]).astype(np.int64)
    input_data_dict = {"token_ids": in1, "length": in2}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./lac/inference.pdmodel",
        params_file="./lac/inference.pdiparams",
    )
    test_suite2.gpu_more_bz_test(
        input_data_dict,
        output_data_dict,
        delta=1e-5,
    )

    del test_suite2  # destroy class to save memory


@pytest.mark.win
@pytest.mark.server
@pytest.mark.jetson
@pytest.mark.gpu
def test_gpu_mixed_precision_bz1():
    """
    compared gpu lac mixed_precision outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    if not os.path.exists("./lac/inference_mixed.pdmodel"):
        test_suite.convert_to_mixed_precision_model(
            src_model="./lac/inference.pdmodel",
            src_params="./lac/inference.pdiparams",
            dst_model="./lac/inference_mixed.pdmodel",
            dst_params="./lac/inference_mixed.pdiparams",
        )
    test_suite.load_config(
        model_file="./lac/inference.pdmodel",
        params_file="./lac/inference.pdiparams",
    )
    in1 = np.random.randint(0, 100, (1, 20)).astype(np.int64)
    in2 = np.array([20]).astype(np.int64)
    input_data_dict = {"token_ids": in1, "length": in2}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./lac/inference_mixed.pdmodel",
        params_file="./lac/inference_mixed.pdiparams",
    )
    test_suite2.gpu_more_bz_test_mix(
        input_data_dict,
        output_data_dict,
        delta=5e-3,
    )

    del test_suite2  # destroy class to save memory
