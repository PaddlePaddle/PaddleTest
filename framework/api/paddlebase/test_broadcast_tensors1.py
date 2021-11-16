#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_broadcast_tensors
"""


import paddle
import pytest
import numpy as np


@pytest.mark.api_base_broadcast_tensors_vartype
def test_broadcast_tensors_base():
    """
    base
    """
    types = [np.float32, np.float64, np.int32, np.int64, np.bool]
    if paddle.device.is_compiled_with_cuda() is True:
        places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
    else:
        # default
        places = [paddle.CPUPlace()]

    for place in places:
        for t in types:
            paddle.disable_static(place)
            # paddle.set_default_dtype(t)
            x1 = np.random.randint(-10, 10, (1, 4)).astype(t)
            x2 = np.random.randint(-1, 1, (3, 1)).astype(t)
            y1 = paddle.to_tensor(x1)
            y2 = paddle.to_tensor(x2)
            out = [paddle.broadcast_to(y1, (3, 4)).numpy(), paddle.broadcast_to(y2, (3, 4)).numpy()]
            dynamic_res = paddle.broadcast_tensors([y1, y2])
            for i in range(2):
                assert np.allclose(out[i], dynamic_res[i].numpy())

            paddle.enable_static()
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                input1 = paddle.static.data(name="x1", shape=[1, 4], dtype=t)
                input2 = paddle.static.data(name="x2", shape=[3, 1], dtype=t)
                output = paddle.broadcast_tensors((input1, input2))

                exe = paddle.static.Executor(place)
                exe.run(startup_program)

                static_res = exe.run(main_program, feed={"x1": x1, "x2": x2}, fetch_list=[output])
                for i in range(2):
                    assert np.allclose(out[i], static_res[i])
            paddle.disable_static(place)


@pytest.mark.api_base_uique_broadcast_tensors_parameters
def test_broadcast_tensors0():
    """
    inputs_num = 3
    """
    x1 = np.random.rand(1, 4).astype(np.float32)
    x2 = np.random.rand(3, 1).astype(np.float32)
    x3 = np.random.rand(3, 4).astype(np.float32)
    y1 = paddle.to_tensor(x1)
    y2 = paddle.to_tensor(x2)
    y3 = paddle.to_tensor(x3)
    out = [paddle.broadcast_to(y1, (3, 4)).numpy(), paddle.broadcast_to(y2, (3, 4)).numpy(), y3.numpy()]
    dynamic_res = paddle.broadcast_tensors([y1, y2, y3])
    for i in range(3):
        assert np.allclose(out[i], dynamic_res[i].numpy())

    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
        input1 = paddle.static.data(name="x1", shape=[1, 4])
        input2 = paddle.static.data(name="x2", shape=[3, 1])
        input3 = paddle.static.data(name="x3", shape=[3, 4])
        output = paddle.broadcast_tensors((input1, input2, input3))

        exe = paddle.static.Executor()
        exe.run(startup_program)

        static_res = exe.run(main_program, feed={"x1": x1, "x2": x2, "x3": x3}, fetch_list=[output])
        for i in range(3):
            assert np.allclose(out[i], static_res[i])
    paddle.disable_static()


@pytest.mark.api_base_uique_broadcast_tensors_parameters
def test_broadcast_tensors1():
    """
    inputs_num = 3
    dim_num different
    """
    x1 = np.random.rand(4, 2, 1, 4).astype(np.float32)
    x2 = np.random.rand(3, 1).astype(np.float32)
    x3 = np.random.rand(1, 3, 4).astype(np.float32)
    y1 = paddle.to_tensor(x1)
    y2 = paddle.to_tensor(x2)
    y3 = paddle.to_tensor(x3)
    out = [
        paddle.broadcast_to(y1, (4, 2, 3, 4)).numpy(),
        paddle.broadcast_to(y2, (4, 2, 3, 4)).numpy(),
        paddle.broadcast_to(y3, (4, 2, 3, 4)).numpy(),
    ]
    dynamic_res = paddle.broadcast_tensors([y1, y2, y3])
    for i in range(3):
        assert np.allclose(out[i], dynamic_res[i].numpy())

    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
        input1 = paddle.static.data(name="x1", shape=[4, 2, 1, 4])
        input2 = paddle.static.data(name="x2", shape=[3, 1])
        input3 = paddle.static.data(name="x3", shape=[1, 3, 4])
        output = paddle.broadcast_tensors((input1, input2, input3))

        exe = paddle.static.Executor()
        exe.run(startup_program)

        static_res = exe.run(main_program, feed={"x1": x1, "x2": x2, "x3": x3}, fetch_list=[output])
        for i in range(3):
            assert np.allclose(out[i], static_res[i])
    paddle.disable_static()
