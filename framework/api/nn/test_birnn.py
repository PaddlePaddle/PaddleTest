#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.RNN float32测试
"""
import copy

from apibase import APIBase, compare
import paddle
import pytest
import numpy as np

from rnn_numpy import SimpleRNNCell, BiRNN


class TestBiRNN(APIBase):
    """
    test RNN float32
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        self.seed = 100
        self.delta = 0.0001
        self.forward_kwargs = {}  # 前向传播参数
        paddle.set_default_dtype("float32")

    def _static_forward(self, res, data=None, **kwargs):
        """
        _static_forward
        """

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        main_program.random_seed = self.seed

        cell_fw = kwargs["cell_fw"]
        cell_bw = kwargs["cell_bw"]

        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                if data is not None:
                    data = data.astype(self.dtype)
                    self.data = paddle.static.data(name="data", shape=data.shape, dtype=self.dtype)
                    self.data.stop_gradient = False
                data = dict({"data": data}, **kwargs)

                static_cell_fw = paddle.nn.SimpleRNNCell(cell_fw.input_size, cell_fw.hidden_size)
                static_cell_bw = paddle.nn.SimpleRNNCell(cell_bw.input_size, cell_bw.hidden_size)

                parameters_fw = {}
                for k, v in kwargs["cell_fw"].named_parameters():
                    parameters_fw[k] = v

                parameters_bw = {}
                for k, v in kwargs["cell_bw"].named_parameters():
                    parameters_bw[k] = v

                obj = self.func(
                    cell_fw=static_cell_fw, cell_bw=static_cell_bw, time_major=self.kwargs.get("time_major", False)
                )

                output, h = obj(self.data)

                if self.enable_backward:
                    loss = paddle.mean(output)
                    g = paddle.static.gradients(loss, self.data)
                    exe = paddle.static.Executor(self.place)
                    exe.run(startup_program)

                    for k, v in static_cell_fw.named_parameters():
                        v.set_value(parameters_fw[k])
                    for k, v in static_cell_bw.named_parameters():
                        v.set_value(parameters_bw[k])

                    res = exe.run(main_program, feed=data, fetch_list=[output, h, g], return_numpy=True)
                    grad = {"data": res[3]}
                    return res[0], grad
                else:
                    exe = paddle.static.Executor(self.place)
                    exe.run(startup_program)

                    for k, v in static_cell_fw.named_parameters():
                        v.set_value(parameters_fw[k])
                    for k, v in static_cell_bw.named_parameters():
                        v.set_value(parameters_bw[k])

                    res = exe.run(main_program, feed=data, fetch_list=[output, h], return_numpy=True)
                    return res[0]

    def _dygraph_forward(self):
        """
        _dygraph_forward
        """
        cell_fw = copy.deepcopy(self.kwargs.get("cell_fw"))
        cell_bw = copy.deepcopy(self.kwargs.get("cell_bw"))
        obj = self.func(cell_fw=cell_fw, cell_bw=cell_bw, time_major=self.kwargs.get("time_major", False))
        res = obj(
            self.data,
            initial_states=self.forward_kwargs.get("initial_states", None),
            sequence_length=self.forward_kwargs.get("sequence_length", None),
        )
        return res[0]


obj = TestBiRNN(paddle.nn.BiRNN)


class TestBiRNN64(TestBiRNN):
    """
    test RNN float64
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64]
        self.seed = 100
        self.delta = 0.0001
        self.forward_kwargs = {}  # 前向传播参数
        paddle.set_default_dtype("float64")


obj64 = TestBiRNN64(paddle.nn.BiRNN)


def copy_cell_params(np_cell, paddle_cell, dtype="float32"):
    """
    将np_cell的参数复制到paddle_cell中
    """
    paddle.disable_static()
    state = np_cell.parameters
    for k, v in paddle_cell.named_parameters():
        t = state[k].astype(dtype)
        v.set_value(t)
    return paddle_cell


def create_cell(input_size, hidden_size, dtype="float32"):
    """
    创建RNNCell，保证numpy版本和Paddle初始化参数相同。
    """
    np_cell_fw = SimpleRNNCell(input_size, hidden_size)
    np_cell_bw = SimpleRNNCell(input_size, hidden_size)
    paddle_cell_fw = paddle.nn.SimpleRNNCell(input_size, hidden_size)
    paddle_cell_bw = paddle.nn.SimpleRNNCell(input_size, hidden_size)

    paddle_cell_fw = copy_cell_params(np_cell_fw, paddle_cell_fw, dtype)
    paddle_cell_bw = copy_cell_params(np_cell_bw, paddle_cell_bw, dtype)

    return np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw


@pytest.mark.api_nn_RNN_vartype
def test_rnn_base():
    """
    base测试，包括动态度、静态图、cpu/gpu，grad，动态静态图的结果一致性
    """
    paddle.set_default_dtype("float32")
    np.random.seed(obj.seed)
    inputs = np.random.random((3, 3, 4))
    np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw = create_cell(4, 8)
    rnn = BiRNN(cell_fw=np_cell_fw, cell_bw=np_cell_bw)
    res_outputs = rnn(inputs)[0]
    res = res_outputs.astype("float32")
    obj.base(res, data=inputs, cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw)


@pytest.mark.api_nn_RNN_parameters
def test_rnn1():
    """
    float32默认参数测试。
    time_major=False
    RnnCell size = (16, 32)
    """
    paddle.set_default_dtype("float32")
    paddle.disable_static()
    # numpy
    np.random.seed(obj.seed)
    inputs = np.random.random((10, 8, 16))
    np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw = create_cell(16, 32)
    rnn = BiRNN(cell_fw=np_cell_fw, cell_bw=np_cell_bw)
    res_outputs = rnn(inputs)[0]
    res = res_outputs.astype("float32")
    obj.run(res, data=inputs, cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw)


@pytest.mark.api_nn_RNN_parameters
def test_rnn2():
    """
    time_major=True
    """
    paddle.set_default_dtype("float32")
    paddle.disable_static()
    # numpy
    np.random.seed(obj.seed)
    inputs = np.random.random((10, 4, 128))
    np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw = create_cell(128, 32)
    rnn = BiRNN(cell_fw=np_cell_fw, cell_bw=np_cell_bw, time_major=True)
    res_outputs = rnn(inputs)[0]
    res = res_outputs.astype("float32")
    obj.run(res, data=inputs, cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw, time_major=True)


def test_rnn3():
    """
    带初始化状态测试, 带有initial_states
    """
    dtype = "float32"
    paddle.set_default_dtype(dtype)
    paddle.disable_static()
    # numpy
    np.random.seed(obj.seed)
    inputs = np.random.random((10, 8, 16))
    initial_states0 = np.random.random((10, 32))
    initial_states1 = np.random.random((10, 32))
    initial_states = [initial_states0, initial_states1]
    np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw = create_cell(16, 32, dtype=dtype)
    np_rnn = BiRNN(cell_fw=np_cell_fw, cell_bw=np_cell_bw)
    np_outputs = np_rnn(inputs, initial_states=initial_states)[0].astype(dtype)

    paddle_rnn = paddle.nn.BiRNN(cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw)
    initial_states = [paddle.to_tensor(t, dtype=dtype) for t in initial_states]
    paddle_outputs = paddle_rnn(paddle.to_tensor(inputs, dtype=dtype), initial_states=initial_states)[0]

    compare(expect=np_outputs, result=paddle_outputs.numpy(), delta=obj.delta, rtol=obj.rtol)


@pytest.mark.api_nn_RNN_parameters
def test_rnn4():
    """
    float32 异常测试，fw_cell和bw_cell尺寸不同
    """
    paddle.set_default_dtype("float32")
    paddle.disable_static()
    # numpy
    np.random.seed(obj64.seed)
    inputs = np.random.random((10, 4, 128))

    paddle_cell_fw = paddle.nn.SimpleRNNCell(4, 16)
    paddle_cell_bw = paddle.nn.SimpleRNNCell(8, 16)

    obj.exception(etype=ValueError, mode="python", data=inputs, cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw)


@pytest.mark.api_nn_RNN_vartype
def test_rnn_base_64():
    """
    float64 base测试，包括动态度、静态图、cpu/gpu，grad，动态静态图的结果一致性
    """
    paddle.disable_static()
    paddle.set_default_dtype("float64")
    np.random.seed(obj64.seed)
    inputs = np.random.random((3, 3, 4))
    np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw = create_cell(4, 8, dtype="float64")
    rnn = BiRNN(cell_fw=np_cell_fw, cell_bw=np_cell_bw)
    res_outputs = rnn(inputs)[0]
    res = res_outputs.astype(np.float64)
    obj64.base(res, data=inputs, cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw)


@pytest.mark.api_nn_RNN_parameters
def test_rnn1_64():
    """
    float64默认参数测试。
    time_major=False
    RnnCell size = (16, 32)
    """
    paddle.set_default_dtype("float64")
    paddle.disable_static()
    # numpy
    np.random.seed(obj64.seed)
    inputs = np.random.random((10, 8, 16))
    np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw = create_cell(16, 32, dtype="float64")
    rnn = BiRNN(cell_fw=np_cell_fw, cell_bw=np_cell_bw)
    res_outputs = rnn(inputs)[0]
    res = res_outputs.astype(np.float64)
    obj64.run(res, data=inputs, cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw)


@pytest.mark.api_nn_RNN_parameters
def test_rnn2_64():
    """
    time_major=True
    """
    paddle.set_default_dtype("float64")
    paddle.disable_static()
    # numpy
    np.random.seed(obj64.seed)
    inputs = np.random.random((10, 4, 128))
    np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw = create_cell(128, 32, dtype="float64")
    rnn = BiRNN(cell_fw=np_cell_fw, cell_bw=np_cell_bw, time_major=True)
    res_outputs = rnn(inputs)[0]
    res = res_outputs.astype(np.float64)
    obj64.run(res, data=inputs, cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw, time_major=True)


def test_rnn3_64():
    """
    带初始化状态测试, 带有initial_states, time_major=False
    """
    dtype = "float64"
    paddle.set_default_dtype(dtype)
    paddle.disable_static()
    # numpy
    np.random.seed(obj.seed)
    inputs = np.random.random((10, 8, 16))
    initial_states0 = np.random.random((10, 32))
    initial_states1 = np.random.random((10, 32))
    initial_states = [initial_states0, initial_states1]
    np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw = create_cell(16, 32, dtype=dtype)
    np_rnn = BiRNN(cell_fw=np_cell_fw, cell_bw=np_cell_bw)
    np_outputs = np_rnn(inputs, initial_states=initial_states)[0].astype(dtype)

    paddle_rnn = paddle.nn.BiRNN(cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw)
    initial_states = [paddle.to_tensor(t, dtype=dtype) for t in initial_states]
    paddle_outputs = paddle_rnn(paddle.to_tensor(inputs, dtype=dtype), initial_states=initial_states)[0]

    compare(expect=np_outputs, result=paddle_outputs.numpy(), delta=obj.delta, rtol=obj.rtol)


def test_rnn3_64():
    """
    带初始化状态测试, 带有initial_states，time_major=False
    """
    dtype = "float64"
    paddle.set_default_dtype(dtype)
    paddle.disable_static()
    # numpy
    np.random.seed(obj.seed)
    inputs = np.random.random((10, 8, 16))
    initial_states0 = np.random.random((10, 32))
    initial_states1 = np.random.random((10, 32))
    initial_states = [initial_states0, initial_states1]
    np_cell_fw, np_cell_bw, paddle_cell_fw, paddle_cell_bw = create_cell(16, 32, dtype=dtype)
    np_rnn = BiRNN(cell_fw=np_cell_fw, cell_bw=np_cell_bw)
    np_outputs = np_rnn(inputs, initial_states=initial_states)[0].astype(dtype)

    paddle_rnn = paddle.nn.BiRNN(cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw)
    initial_states = [paddle.to_tensor(t, dtype=dtype) for t in initial_states]
    paddle_outputs = paddle_rnn(paddle.to_tensor(inputs, dtype=dtype), initial_states=initial_states)[0]

    compare(expect=np_outputs, result=paddle_outputs.numpy(), delta=obj.delta, rtol=obj.rtol)


@pytest.mark.api_nn_RNN_parameters
def test_rnn4_64():
    """
    float64 异常测试，fw_cell和bw_cell尺寸不同
    """
    paddle.set_default_dtype("float64")
    paddle.disable_static()
    # numpy
    np.random.seed(obj64.seed)
    inputs = np.random.random((10, 4, 128))

    paddle_cell_fw = paddle.nn.SimpleRNNCell(4, 16)
    paddle_cell_bw = paddle.nn.SimpleRNNCell(8, 16)

    obj.exception(etype=ValueError, mode="python", data=inputs, cell_fw=paddle_cell_fw, cell_bw=paddle_cell_bw)
