#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
aprac2
"""
import numpy as np
import paddle

np.random.seed(33)
paddle.seed(33)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


def naive_func(a, in_params, func):
    """用于动转静的方法"""
    layer = eval(func)(**a, **in_params)
    return layer


np.random.seed(33)
paddle.seed(33)
func = "paddle.randint_like"
inputs = [randtool("float", -1, 1, shape=[2, 3, 4, 4])]

in_tensor = {"x": paddle.to_tensor(inputs[0], dtype="float32")}

in_params = {"low": 3, "high": 5}

np.random.seed(33)
paddle.seed(33)
obj = naive_func

np.random.seed(33)
paddle.seed(33)
obj_ = paddle.jit.to_static(obj)

# obj.eval()

np.random.seed(33)
paddle.seed(33)
dy_out = obj(in_tensor, in_params, func)
print("dy_out is: ", dy_out)


# jit_obj = paddle.jit.to_static(obj)
# print('jit_obj is created successfully !!!')

st_out = obj_(in_tensor, in_params, func)
print("st_out is: ", st_out)

paddle.jit.save(obj_, "debug_path/randint_like_0")
