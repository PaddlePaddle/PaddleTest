#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
aprac2
"""
import numpy as np
import paddle

paddle.seed(33)
np.random.seed(33)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


# class BuildClass(paddle.nn.Layer):
#     """
#     用于动转静的nn.Layer
#     """
#
#     def __init__(self, in_params, func):
#         super(BuildClass, self).__init__()
#         self.func = eval(func)(**in_params)
#
#     def forward(self, input):
#         """
#         forward
#         """
#         x = self.func(input)
#         return x


class BuildJitFunc(paddle.nn.Layer):
    """
    用于动转静的nn.Layer
    """

    def __init__(self, in_params, func):
        super(BuildJitFunc, self).__init__()
        paddle.seed(33)
        self.func = eval(func)
        self._params = in_params

    @paddle.jit.to_static
    def forward(self, inputs):
        """
        forward
        """
        x = self.func(inputs, **self._params)
        return x


func = "paddle.multinomial"

in_tensor = {"x": paddle.to_tensor(randtool("float", 0, 1, shape=[2, 5]), dtype="float32")}

in_params = {"num_samples": 1, "replacement": True}


obj = BuildJitFunc(in_params, func)

obj.eval()

paddle.seed(33)
dy_out = obj(in_tensor["x"])
print("dy_out is: ", dy_out)

jit_obj = paddle.jit.to_static(obj)
print("jit_obj is created successfully !!!")

paddle.seed(33)
st_out = jit_obj(in_tensor["x"])
print("st_out is: ", st_out)

# data_i = paddle.static.InputSpec(shape=[None, 3, 5, 5], dtype='float32', name='data')
data_i = paddle.static.InputSpec(shape=[2, 5], dtype="float32", name="x")
# indice_i = paddle.static.InputSpec(shape=[2, 3, 5, 5], dtype='int32', name='indice_i')
spec_obj = paddle.jit.to_static(obj, input_spec=[data_i])
paddle.seed(33)
spec_out = spec_obj(in_tensor["x"])
print("spec_out is: ", spec_out)
