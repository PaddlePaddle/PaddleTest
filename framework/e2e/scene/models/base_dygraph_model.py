#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
dygraph model
"""

import paddle
import numpy as np


class Dygraph(paddle.nn.Layer):
    """model"""

    def __init__(self, dtype=np.float64, in_features=10, out_features=2):
        """__init__"""
        paddle.set_default_dtype(dtype)
        super(Dygraph, self).__init__()
        self.fc = paddle.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
            bias_attr=paddle.nn.initializer.Constant(value=0.33),
        )

    def forward(self, inputs):
        """forward"""
        output = self.fc(inputs)
        return output


# paddle.enable_static()


# model = Dygraph()
# main_program = paddle.static.Program()
# startup_program = paddle.static.Program()
# with paddle.utils.unique_name.guard():
#     # with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
#         reader_static = paddle.static.data(name="reader_static",
#                                                 shape=[1, 1, 10],
#                                                 dtype="float64")
#
#         # logging.info("params_group1 is: {}".format(self.kwargs_dict_static["params_group1"]))
#         # logging.info("self.reader_static is: {}".format(self.reader_static))
#         # logging.info("self.model is: {}".format(self.model))
#         out = model(reader_static)


# reader_static = paddle.static.data(name="reader_static",
#                                         shape=[1, 1, 10],
#                                         dtype="float64")
# # logging.info("params_group1 is: {}".format(self.kwargs_dict_static["params_group1"]))
# # logging.info("self.reader_static is: {}".format(self.reader_static))
# # logging.info("self.model is: {}".format(self.model))
# out = model(reader_static)
