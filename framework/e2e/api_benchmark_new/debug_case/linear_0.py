#!/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test linear_0
"""
import timeit
from inspect import isclass
import time
import numpy as np
import paddle

paddle.set_device("cpu")


def _randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)
    elif dtype == "int32":
        return np.random.randint(low, high, shape).astype("int32")
    elif dtype == "int64":
        return np.random.randint(low, high, shape).astype("int64")
    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)
    elif dtype == "float16":
        return low + (high - low) * np.random.random(shape).astype("float16")
    elif dtype == "float32":
        return low + (high - low) * np.random.random(shape).astype("float32")
    elif dtype == "float64":
        return low + (high - low) * np.random.random(shape).astype("float64")
    elif dtype == "bfloat16":
        return low + (high - low) * np.random.random(shape).astype("bfloat16")
    elif dtype in ["complex", "complex64", "complex128"]:
        data = low + (high - low) * np.random.random(shape) + (low + (high - low) * np.random.random(shape)) * 1j
        return data if dtype == "complex" or "complex128" else data.astype(np.complex64)
    elif dtype == "bool":
        data = np.random.randint(0, 2, shape).astype("bool")
        return data
    else:
        assert False, "dtype is not supported"


api = "paddle.nn.functional.linear"
all_data = {"x": {"random": True, "type": "Tensor", "dtype": "float32", "shape": [1, 1], "range": [-1, 1]}}
params = {
    "weight": {"random": True, "type": "Tensor", "dtype": "float32", "shape": [1, 1], "range": [-1, 1]},
    "bias": {"random": True, "type": "Tensor", "dtype": "float32", "shape": [1], "range": [-1, 1]},
}

inputs = {}
for data, v in all_data.items():
    if isinstance(v, dict):
        if v.get("random"):
            inputs[data] = paddle.to_tensor(
                _randtool(dtype=v.get("dtype"), low=v.get("range")[0], high=v.get("range")[1], shape=v.get("shape"))
            )
        else:
            inputs[data] = paddle.to_tensor(np.array(v.get("value")), dtype=v.get("dtype"))

for data, v in params.items():
    if isinstance(v, dict):
        if v.get("random"):
            params[data] = paddle.to_tensor(
                _randtool(dtype=v.get("dtype"), low=v.get("range")[0], high=v.get("range")[1], shape=v.get("shape"))
            )
        else:
            params[data] = paddle.to_tensor(np.array(v.get("value")), dtype=v.get("dtype"))


def func_def(api, inputs, params):
    """
    func
    """
    eval(api)(**inputs, **params)


def func_class(api, inputs, params):
    """
    class
    """
    obj = eval(api)(**params)
    obj(*inputs)


all_time = []
loops = 50

for i in range(loops):
    if isclass(eval(api)):
        inputs_list = []
        for k, v in inputs.items():
            inputs_list.append(v)
        forward_time = timeit.timeit(lambda: func_class(api, inputs_list, params), number=1000)
        all_time.append(forward_time)
    else:
        forward_time = timeit.timeit(lambda: func_def(api, inputs, params), number=1000)
        all_time.append(forward_time)

head = int(loops / 5)
tail = int(loops - loops / 5)
result = sum(sorted(all_time)[head:tail]) / (tail - head)
print(result)
