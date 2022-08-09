#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test cast
"""
from apibase import APIBase
from apibase import randtool
import paddle
import pytest
import numpy as np


class TestAbs(APIBase):
    """
    test cast
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.enable_backward = False


obj = TestAbs(paddle.cast)
# obj1 = TestAbs(paddle.cast)
# obj1.types = [np.float16, np.int32]


@pytest.mark.api_base_cast_vartype
def test_cast_base():
    """
    base
    """
    x = randtool("float", -10, 10, (3, 3, 3))
    res = x.astype(np.float64)
    obj.base(res=res, x=x, dtype="float64")


@pytest.mark.api_base_cast_parameters
def test_cast():
    """
    default
    """
    paddle.disable_static()
    if paddle.device.is_compiled_with_cuda() is True:
        place = ["cpu", "gpu"]
        type_list = [np.int32, np.int64, np.bool, np.float32, np.float64, np.float16]
    else:
        place = ["cpu"]
        type_list = [np.int32, np.int64, np.bool, np.float32, np.float64, np.uint8]
    for p in place:
        paddle.set_device(p)
        print("++++++ use device {} ++++++".format(p))
        for o_t in type_list:
            for x_t in type_list:
                print("input dtype is {}".format(x_t))
                print("output dtype is {}".format(o_t))
                x = np.array([-1.2, -2, 0, 4.5]).astype(x_t)
                exp = paddle.to_tensor(x.astype(o_t))
                res = paddle.cast(paddle.to_tensor(x), o_t)
                assert exp.dtype == res.dtype
                print("exp is {}".format(exp))
                print("res is {}".format(res))
                if o_t == np.float16:
                    delta = 1e-2
                    rtol = 1e-2
                else:
                    delta = 1e-6
                    rtol = 1e-7
                assert np.allclose(res.numpy(), exp.numpy(), delta, rtol)


# @pytest.mark.api_base_cast_parameters
# def test_cast1():
#     """
#     default
#     """
#     paddle.disable_static()
#     if paddle.device.is_compiled_with_cuda() is True:
#         place = ['cpu', 'gpu']
#         type_list = [np.int32, np.int64, np.bool, np.float32, np.float64, np.uint8, np.float16]
#     else:
#         place = ['cpu']
#         type_list = [np.int32, np.int64, np.bool, np.float32, np.float64, np.uint8]
#     for p in place:
#         paddle.set_device(p)
#         print('++++++ use device {} ++++++'.format(p))
#         for o_t in type_list:
#             for x_t in type_list:
#                 print('input dtype is {}'.format(x_t))
#                 print('output dtype is {}'.format(o_t))
#                 x = np.array([-1.2, -2, 0, 4.5]).astype(x_t)
#                 exp = paddle.to_tensor(x.astype(o_t))
#                 res = paddle.cast(paddle.to_tensor(x), o_t)
#                 assert exp.dtype == res.dtype
#                 print('exp is {}'.format(exp))
#                 print('res is {}'.format(res))
#                 if o_t == np.float16:
#                     delta = 1e-2
#                     rtol = 1e-2
#                 else:
#                     delta = 1e-6
#                     rtol = 1e-7
#                 assert np.allclose(res.numpy(), exp.numpy(), delta, rtol)
