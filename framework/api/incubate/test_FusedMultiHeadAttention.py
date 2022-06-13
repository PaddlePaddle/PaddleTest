#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_FusedMultiHeadAttention
"""
import sys
from apibase import APIBase
import paddle
import paddle.nn.initializer as initializer
import pytest
import numpy as np

sys.path.append("../../utils/")
from interceptor import skip_not_compile_gpu


class TestFusedMultiHeadAttention(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.enable_backward = False
        # self.delta = 1e-4


obj = TestFusedMultiHeadAttention(paddle.incubate.nn.FusedMultiHeadAttention)
obj.places = [paddle.CUDAPlace(0)]


@skip_not_compile_gpu
@pytest.mark.api_incubate_fused_multi_head_attention_parameters
def test_FusedMultiHeadAttention0():
    """
    normalize_before=False
    """
    np.random.seed(22)
    query_data = np.random.rand(1, 2, 4)
    # attn_mask_data = np.random.rand(1, 2, 2, 2)
    res = np.array(
        [[[-0.41791570, 1.90818739, 1.38764536, 5.12208271], [-0.00312311, 1.70438468, 1.00867295, 5.29018211]]]
    )
    attr = initializer.Constant(2)
    obj.run(
        res=res,
        data=query_data,
        embed_dim=4,
        num_heads=2,
        dropout_rate=0,
        attn_dropout_rate=0,
        qkv_weight_attr=attr,
        qkv_bias_attr=attr,
        linear_weight_attr=attr,
        linear_bias_attr=attr,
        pre_ln_scale_attr=attr,
        pre_ln_bias_attr=attr,
        ln_scale_attr=attr,
        ln_bias_attr=attr,
    )


@skip_not_compile_gpu
@pytest.mark.api_incubate_fused_multi_head_attention_parameters
def test_FusedMultiHeadAttention1():
    """
    normalize_before=True
    """
    np.random.seed(22)
    query_data = np.random.rand(1, 2, 4)
    # attn_mask_data = np.random.rand(1, 2, 2, 2)
    res = np.array(
        [
            [
                [146.20846558, 146.48167419, 146.42053223, 146.85917664],
                [146.17115784, 146.33886719, 146.27053833, 146.69104004],
            ]
        ]
    )
    attr = initializer.Constant(2)
    obj.run(
        res=res,
        data=query_data,
        embed_dim=4,
        num_heads=2,
        dropout_rate=0,
        attn_dropout_rate=0,
        normalize_before=True,
        qkv_weight_attr=attr,
        qkv_bias_attr=attr,
        linear_weight_attr=attr,
        linear_bias_attr=attr,
        pre_ln_scale_attr=attr,
        pre_ln_bias_attr=attr,
        ln_scale_attr=attr,
        ln_bias_attr=attr,
    )
