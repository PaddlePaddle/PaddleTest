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
from test_case import InferenceTest

# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    model_url = "https://paddle-qa.bj.bcebos.com/inference_model_clipped/2.2.2/nlp/AFQMC_base.tgz"
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
    test_suite.load_config(
        model_file="./AFQMC_base/inference.pdmodel",
        params_file="./AFQMC_base/inference.pdiparams",
    )
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.onnxruntime
def test_onnxruntime():
    """
    compared onnxruntime batch_size = [1] AFQMC_base outputs with true val
    """
    check_model_exist()

    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./AFQMC_base/inference.pdmodel",
        params_file="./AFQMC_base/inference.pdiparams",
    )

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
    test_suite2.load_config(
        model_file="./AFQMC_base/inference.pdmodel",
        params_file="./AFQMC_base/inference.pdiparams",
    )
    test_suite2.onnxruntime_test(
        input_data_dict,
        output_data_dict,
        delta=1e-5,
    )

    del test_suite2  # destroy class to save memory
