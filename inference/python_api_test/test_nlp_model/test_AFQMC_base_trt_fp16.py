# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test AFQMC_base model
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
    model_url = "https://paddle-qa.bj.bcebos.com/inference_model/2.2.2/nlp/AFQMC_base.tgz"
    if not os.path.exists("./AFQMC_base/inference.pdmodel"):
        wget.download(model_url, out="./")
        tar = tarfile.open("AFQMC_base.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(model_file="./AFQMC_base/inference.pdmodel", params_file="./AFQMC_base/inference.pdiparams")
    test_suite.config_test()


def set_dynamic_shape(config):
    """
    set dynamic shape
    """
    names = ["input_ids", "token_type_ids", "cumsum_0.tmp_0", "full_like_0.tmp_0", "unsqueeze2_0.tmp_0", "tmp_4"]
    max_batch = 40
    min_batch_seq_len = 1
    max_batch_seq_len = 512
    opt_batch_seq_len = 256

    min_shape = [1, min_batch_seq_len]
    max_shape = [max_batch, max_batch_seq_len]
    opt_shape = [max_batch, opt_batch_seq_len]
    config.set_trt_dynamic_shape_info(
        {
            names[0]: min_shape,
            names[1]: min_shape,
            names[2]: min_shape,
            names[3]: min_shape,
            names[4]: [1, 1, 1, 1],
            names[5]: min_shape,
        },
        {
            names[0]: max_shape,
            names[1]: max_shape,
            names[2]: max_shape,
            names[3]: max_shape,
            names[4]: [40, 1, 1, 128],
            names[5]: max_shape,
        },
        {
            names[0]: opt_shape,
            names[1]: opt_shape,
            names[2]: opt_shape,
            names[3]: opt_shape,
            names[4]: [40, 1, 1, 128],
            names[5]: opt_shape,
        },
    )


@pytest.mark.win
@pytest.mark.server
@pytest.mark.trt_fp16
def test_trt_fp16_bz1():
    """
    compared trt fp16 batch_size=1 bert outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./AFQMC_base/inference.pdmodel", params_file="./AFQMC_base/inference.pdiparams")

    src_ids = (
        np.array(
            [
                1,
                654,
                21,
                778,
                291,
                21,
                2,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                17963,
                9,
                358,
                567,
                1179,
                17963,
                532,
                537,
                358,
                386,
                21,
                2,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                2,
                691,
                736,
                1431,
                1137,
                1279,
                779,
                12049,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                12043,
                2,
                459,
                335,
                263,
                65,
                129,
                37,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                42,
                42,
                42,
                2,
                51,
                23,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                2,
            ]
            + [1] * 31
        )
        .astype(np.int64)
        .reshape(1, 128)
    )
    input_ids = np.tile(src_ids, (40, 1))
    sent_ids = (
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
            + [1] * 31
        )
        .astype(np.int64)
        .reshape(1, 128)
    )
    token_type_ids = np.tile(sent_ids, (40, 1))

    input_data_dict = {"token_type_ids": token_type_ids, "input_ids": input_ids}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./AFQMC_base/inference.pdmodel", params_file="./AFQMC_base/inference.pdiparams")
    set_dynamic_shape(test_suite2.pd_config)
    test_suite2.pd_config.exp_disable_tensorrt_ops(["elementwise_sub"])
    test_suite2.trt_more_bz_test(
        input_data_dict,
        output_data_dict,
        min_subgraph_size=5,
        delta=1e-5,
        max_batch_size=40,
        use_static=False,
        precision="trt_fp16",
    )

    del test_suite2  # destroy class to save memory


@pytest.mark.trt_fp16_multi_thread
def test_trt_fp16_bz1_multi_thread():
    """
    compared trt fp16 batch_size=1 multi_thread bert outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(model_file="./AFQMC_base/inference.pdmodel", params_file="./AFQMC_base/inference.pdiparams")

    src_ids = (
        np.array(
            [
                1,
                654,
                21,
                778,
                291,
                21,
                2,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                17963,
                9,
                358,
                567,
                1179,
                17963,
                532,
                537,
                358,
                386,
                21,
                2,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                2,
                691,
                736,
                1431,
                1137,
                1279,
                779,
                12049,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                12043,
                2,
                459,
                335,
                263,
                65,
                129,
                37,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                42,
                42,
                42,
                2,
                51,
                23,
                654,
                8,
                778,
                326,
                778,
                45,
                291,
                751,
                85,
                155,
                76,
                2,
            ]
            + [1] * 31
        )
        .astype(np.int64)
        .reshape(1, 128)
    )
    input_ids = np.tile(src_ids, (40, 1))
    sent_ids = (
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
            + [1] * 31
        )
        .astype(np.int64)
        .reshape(1, 128)
    )
    token_type_ids = np.tile(sent_ids, (40, 1))

    input_data_dict = {"token_type_ids": token_type_ids, "input_ids": input_ids}
    output_data_dict = test_suite.get_truth_val(input_data_dict, device="cpu")

    del test_suite  # destroy class to save memory

    test_suite2 = InferenceTest()
    test_suite2.load_config(model_file="./AFQMC_base/inference.pdmodel", params_file="./AFQMC_base/inference.pdiparams")
    set_dynamic_shape(test_suite2.pd_config)
    test_suite2.pd_config.exp_disable_tensorrt_ops(["elementwise_sub"])
    test_suite2.trt_bz1_multi_thread_test(
        input_data_dict, output_data_dict, min_subgraph_size=5, delta=1e-5, use_static=False, precision="trt_fp16"
    )

    del test_suite2  # destroy class to save memory

