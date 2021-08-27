#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test summary
"""
import paddle
import paddle.nn as nn
import pytest
import numpy as np
from paddle.static import InputSpec


class LeNet(nn.Layer):
    """LeNet"""
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2D(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2D(2, 2),
        )

        if num_classes > 0:
            self.fc = nn.Sequential(nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, 10))

    def forward(self, inputs):
        """forward"""
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        return x


class LeNetMultiInput(LeNet):
    """LeNetMultiInput"""
    def forward(self, inputs, y):
        """forward"""
        x = self.features(inputs)
        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + y)
        return x


class LeNetListInput(LeNet):
    """LeNetListInput"""
    def forward(self, inputs):
        x = self.features(inputs[0])

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + inputs[1])
        return x


class LeNetDictInput(LeNet):
    """LeNetDictInput"""
    def forward(self, inputs):
        x = self.features(inputs["x1"])

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + inputs["x2"])
        return x


def params_counts(sample_nets):
    """
    parameters counts
    """
    params = 0
    counts = sample_nets.state_dict()
    for k, v in counts.items():
        params += np.prod(v.shape)
    return params


@pytest.mark.fixture(scope="class")
def use_static_mode():
    """
    turn on static graph mode
    """
    paddle.enable_static()
    yield
    paddle.disable_static()


class SummaryTestBase:
    """test for summary"""

    inputsize_params_single = [(1, 1, 28, 28), InputSpec([None, 1, 28, 28], "float32", "image")]
    inputsize_params_multi = [
        [(1, 1, 28, 28), (1, 400)],
        [InputSpec([None, 1, 28, 28], "float32", "input0"), InputSpec([None, 400], "float32", "input1")],
    ]
    input_params_single = [paddle.rand([1, 1, 28, 28])]
    input_params_multi_dict = [{"x1": paddle.rand([1, 1, 28, 28]), "x2": paddle.rand([1, 400])}]
    input_params_multi_list = [[paddle.rand([1, 1, 28, 28]), paddle.rand([1, 400])]]

    @pytest.fixture(scope="function")
    def return_para(self, request):
        """
        return diff parameter
        """
        yield request.param

    @pytest.mark.api_base_summary_parameters
    @pytest.mark.parametrize("return_para", inputsize_params_single, indirect=True)
    def test_inputsize_single(self, return_para):
        """
        test input_size parameter when the model has only one input
        """
        lenet = LeNet()
        params_info = paddle.summary(lenet, input_size=return_para)
        assert params_info["total_params"] == params_counts(lenet)
        print(return_para)
        print(paddle.in_dynamic_mode())

    @pytest.mark.api_base_summary_parameters
    @pytest.mark.parametrize("return_para", inputsize_params_multi, indirect=True)
    def test_inputsize_multi(self, return_para):
        """
        test input_size parameter when the model has multiple input
        """
        lenet_multi_input = LeNetMultiInput()
        print(return_para)
        params_info = paddle.summary(lenet_multi_input, input_size=return_para)
        assert params_info["total_params"] == params_counts(lenet_multi_input)
        print(paddle.in_dynamic_mode())

    @pytest.mark.api_base_summary_parameters
    @pytest.mark.skipif("paddle.in_dynamic_mode() is False", reason="skip test case, need to be fixed")
    @pytest.mark.parametrize("return_para", input_params_single, indirect=True)
    def test_input_single(self, return_para):
        """
        test input parameter when the model has only one input
        """
        lenet = LeNet()
        print(paddle.in_dynamic_mode())
        params_info = paddle.summary(lenet, input=return_para)
        assert params_info["total_params"] == params_counts(lenet)

    @pytest.mark.skipif("paddle.in_dynamic_mode() is False", reason="skip test case, need to be fixed")
    @pytest.mark.parametrize("return_para", input_params_multi_list, indirect=True)
    def test_input_multi_list(self, return_para):
        """
        test input parameter when the model has multiple input
        list input demo
        """
        lenet_multi_input = LeNetListInput()
        params_info = paddle.summary(lenet_multi_input, input=return_para)
        assert params_info["total_params"] == params_counts(lenet_multi_input)
        print(paddle.in_dynamic_mode())

    @pytest.mark.skipif("paddle.in_dynamic_mode() is False", reason="skip test case, need to be fixed")
    @pytest.mark.parametrize("return_para", input_params_multi_dict, indirect=True)
    def test_input_multi_dict(self, return_para):
        """
        test input parameter when the model has multiple input
        list input demo
        """
        lenet_multi_input = LeNetDictInput()
        params_info = paddle.summary(lenet_multi_input, input=return_para)
        assert params_info["total_params"] == params_counts(lenet_multi_input)


class TestSummaryStatic(SummaryTestBase):
    """
    test paddle.summary on static mode
    """

    @classmethod
    def setup_class(cls):
        """setup"""
        paddle.enable_static()
        print("setup class")

    @classmethod
    def teardown_class(cls):
        """teardown"""
        print("tear down")
        paddle.disable_static()


class TestSummaryDynamic(SummaryTestBase):
    """
    test paddle.summary on dynamic mode
    """

    pass
