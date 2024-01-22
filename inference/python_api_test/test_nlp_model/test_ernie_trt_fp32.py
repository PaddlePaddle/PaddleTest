# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test ernie model
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
    ernie_url = "https://paddle-qa.bj.bcebos.com/inference_model_clipped/2.1.2/nlp/ernie.tgz"
    if not os.path.exists("./ernie/inference.pdiparams"):
        wget.download(ernie_url, out="./")
        tar = tarfile.open("ernie.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./ernie/inference.pdmodel",
        params_file="./ernie/inference.pdiparams",
    )
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.jetson
@pytest.mark.trt_fp32
def test_trt_fp32_bz1():
    """
    compared trt fp32 batch_size=1 ernie outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./ernie/inference.pdmodel",
        params_file="./ernie/inference.pdiparams",
    )
    input_ids = np.load("./ernie/input_ids.npy").astype("int64")
    token_type_ids = np.load("./ernie/token_type_ids.npy").astype("int64")

    input_data_dict = {"input_ids": np.array([input_ids]), "token_type_ids": np.array([token_type_ids])}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./ernie/inference.pdmodel",
        params_file="./ernie/inference.pdiparams",
    )
    test_suite2.trt_more_bz_test(
        input_data_dict,
        output_data_dict,
        delta=0.0002,
        max_batch_size=1,
        min_subgraph_size=1,
        precision="trt_fp32",
        dynamic=True,
        auto_tuned=True,
    )
    del test_suite2  # destroy class to save memory


@pytest.mark.trt_fp32_multi_thread
def test_trt_fp32_bz1_multi_thread():
    """
    compared trt fp32 batch_size=1 multi_thread ernie outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./ernie/inference.pdmodel",
        params_file="./ernie/inference.pdiparams",
    )
    input_ids = np.load("./ernie/input_ids.npy").astype("int64")
    token_type_ids = np.load("./ernie/token_type_ids.npy").astype("int64")

    input_data_dict = {"input_ids": np.array([input_ids]), "token_type_ids": np.array([token_type_ids])}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(
        model_file="./ernie/inference.pdmodel",
        params_file="./ernie/inference.pdiparams",
    )
    test_suite2.trt_bz1_multi_thread_test(
        input_data_dict,
        output_data_dict,
        delta=1e-5,
        precision="trt_fp32",
    )

    del test_suite2  # destroy class to save memory
