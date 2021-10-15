#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.RNN float64测试
"""
import copy

from apibase import APIBase
import paddle
import pytest
import numpy as np

from framework.api.nn.rnn_numpy import SimpleRNNCell, RNN


class TestRNN(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = ["float64"]
        self.seed = 100
        self.delta = 0.0001
        self.forward_kwargs = {}  # 前向传播参数
        paddle.set_default_dtype("float64")

    def _static_forward(self, res, data=None, **kwargs):
        """
        _static_forward
        """

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        main_program.random_seed = self.seed

        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                if data is not None:
                    data = data.astype(self.dtype)
                    self.data = paddle.static.data(name="data", shape=data.shape, dtype=self.dtype)
                    self.data.stop_gradient = False
                data = dict({"data": data}, **kwargs)
                static_cell = paddle.nn.SimpleRNNCell(4, 8)
                parameters = {}
                for k, v in kwargs["cell"].named_parameters():
                    parameters[k] = v

                obj = self.func(
                    static_cell,
                    is_reverse=self.kwargs.get("is_reverse", False),
                    time_major=self.kwargs.get("time_major", False),
                )

                output, h = obj(self.data)

                if self.enable_backward:
                    loss = paddle.mean(output)
                    g = paddle.static.gradients(loss, self.data)
                    exe = paddle.static.Executor(self.place)
                    exe.run(startup_program)
                    for k, v in static_cell.named_parameters():
                        v.set_value(parameters[k])
                    res = exe.run(main_program, feed=data, fetch_list=[output, h, g], return_numpy=True)
                    grad = {"data": res[2]}
                    return res[0:2], grad
                else:
                    exe = paddle.static.Executor(self.place)
                    exe.run(startup_program)
                    for k, v in static_cell.named_parameters():
                        v.set_value(parameters[k])
                    res = exe.run(main_program, feed=data, fetch_list=[output, h], return_numpy=True)
                    return res

    def _dygraph_forward(self):
        """
        _dygraph_forward
        """
        cell = copy.deepcopy(self.kwargs.get("cell"))
        obj = self.func(
            cell=cell, is_reverse=self.kwargs.get("is_reverse", False), time_major=self.kwargs.get("time_major", False)
        )
        res = obj(
            self.data,
            initial_states=self.forward_kwargs.get("initial_states", None),
            sequence_length=self.forward_kwargs.get("sequence_length", None),
        )
        return res


obj = TestRNN(paddle.nn.RNN)


def copy_cell_params(np_cell, paddle_cell, dtype="float64"):
    """
    将np_cell的参数复制到paddle_cell中
    """
    paddle.disable_static()
    state = np_cell.parameters
    for k, v in paddle_cell.named_parameters():
        t = state[k].astype(dtype)
        v.set_value(t)
    return paddle_cell


@pytest.mark.api_nn_RNN_vartype
def test_rnn_base():
    """
    base测试，包括动态度、静态图、cpu/gpu，grad，动态静态图的结果一致性
    """
    np.random.seed(obj.seed)
    inputs = np.random.random((2, 3, 4))
    np_cell = SimpleRNNCell(4, 8)
    rnn = RNN(cell=np_cell)
    res_outputs, res_final_states = rnn(inputs)
    res = [res_outputs.astype("float64"), res_final_states.astype("float64")]
    paddle.disable_static()
    cell = paddle.nn.SimpleRNNCell(4, 8)
    cell = copy_cell_params(np_cell, cell)
    obj.base(res, data=inputs, cell=cell)


@pytest.mark.api_nn_RNN_parameters
def test_rnn1():
    """
    测试默认参数
    """
    # numpy
    np.random.seed(obj.seed)
    inputs = np.random.random((2, 3, 4))
    np_cell = SimpleRNNCell(4, 8)
    rnn = RNN(cell=np_cell)
    res_outputs, res_final_states = rnn(inputs)
    res = [res_outputs.astype("float64"), res_final_states.astype("float64")]

    paddle.disable_static()

    cell = paddle.nn.SimpleRNNCell(4, 8)
    cell = copy_cell_params(np_cell, cell)
    obj.run(res, data=inputs, cell=cell)


#
@pytest.mark.apt_nn_RNN_parameters
def test_rnn2():
    """
    测试 is_reverse=True
    """

    # numpy
    np.random.seed(obj.seed)
    inputs = np.random.random((2, 3, 4))
    np_cell = SimpleRNNCell(4, 8)
    rnn = RNN(cell=np_cell, is_reverse=True)
    res_outputs, res_final_states = rnn(inputs)
    res = [res_outputs.astype("float64"), res_final_states.astype("float64")]

    paddle.disable_static()

    cell = paddle.nn.SimpleRNNCell(4, 8)
    cell = copy_cell_params(np_cell, cell)
    obj.run(res, data=inputs, cell=cell, is_reverse=True)


@pytest.mark.apt_nn_RNN_parameters
def test_rnn3():
    """
    测试 time_major = True
    """
    # numpy
    np.random.seed(obj.seed)
    inputs = np.random.random((2, 3, 4))
    np_cell = SimpleRNNCell(4, 8)
    rnn = RNN(cell=np_cell, time_major=True)
    res_outputs, res_final_states = rnn(inputs)
    res = [res_outputs.astype("float64"), res_final_states.astype("float64")]

    paddle.disable_static()

    cell = paddle.nn.SimpleRNNCell(4, 8)
    cell = copy_cell_params(np_cell, cell)

    obj.run(res, data=inputs, cell=cell, time_major=True)
