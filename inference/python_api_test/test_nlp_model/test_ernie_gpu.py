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
@pytest.mark.config_disablegpu_memory
def test_disable_gpu():
    """
    test no gpu resources occupied after disable gpu
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./ernie/inference.pdmodel",
        params_file="./ernie/inference.pdiparams",
    )
    batch_size = 1
    fake_input0 = np.zeros((batch_size, 128)).astype("int64")
    fake_input1 = np.zeros((batch_size, 128)).astype("int64")
    input_data_dict = {"input_ids": fake_input0, "token_type_ids": fake_input1}
    test_suite.disable_gpu_test(input_data_dict)


@pytest.mark.win
@pytest.mark.server
@pytest.mark.jetson
@pytest.mark.gpu
def test_gpu_bz1():
    """
    compared gpu ernie outputs with true val
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
    compared gpu bert mixed_precision outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    if not os.path.exists("./ernie/inference_mixed.pdmodel"):
        test_suite.convert_to_mixed_precision_model(
            src_model="./ernie/inference.pdmodel",
            src_params="./ernie/inference.pdiparams",
            dst_model="./ernie/inference_mixed.pdmodel",
            dst_params="./ernie/inference_mixed.pdiparams",
        )
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
        model_file="./ernie/inference_mixed.pdmodel",
        params_file="./ernie/inference_mixed.pdiparams",
    )
    test_suite2.gpu_more_bz_test_mix(
        input_data_dict,
        output_data_dict,
        delta=5e-3,
    )

    del test_suite2  # destroy class to save memory
