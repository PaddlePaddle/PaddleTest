#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
  rnn test base class
"""

import copy
import logging
import paddle
import paddle.nn.initializer as initializer
import numpy as np


class RnnBase(object):
    """
    rnn base
    """

    def __init__(self, func):
        """
        initializer
        """
        self.places = []
        self.dtype = "float32"
        self.gap = 0.005
        if paddle.device.is_compiled_with_cuda() is True:
            self.places = [paddle.CUDAPlace(0)]
        else:
            self.places = [paddle.CPUPlace()]

        self.func = func
        self.atol = 1e-5
        self.enable_static = True

    def cal_dynamic(self, place):
        """
        calculate dynamic forward and backward result
        """
        paddle.disable_static(place)
        cell = self.func(**self.kwargs)
        r = cell(*self.data)
        loss = paddle.mean(r)
        # loss.backward(retain_graph=True)
        loss.backward()
        return r[0], self.data[0].grad

    def cal_static(self, place):
        """
        calculate static forward and backward result
        """

        static_data = {}
        l_data = len(self.data)
        for i in range(l_data):
            static_data[i] = self.data[i]
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.utils.unique_name.guard():
            with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
                data, feed = [], {}
                for k, v in static_data.items():
                    data.append(paddle.static.data(name=str(k), shape=v.shape, dtype=v.dtype))
                    feed[str(k)] = static_data[k].numpy()
                data[0].stop_gradient = False
                cell = self.func(**self.kwargs)
                out = cell(*data)
                loss = paddle.mean(out[0])
                grad = paddle.static.gradients(loss, data[0])
                exe = paddle.static.Executor(place)
                exe.run(startup_program)
                res = exe.run(main_program, feed=feed, fetch_list=[out[0]] + [grad])
                paddle.disable_static()
                return res

    def numerical_gradients(self):
        """
        calculate numerical gradients
        """
        gradients = []
        shape = self.data[0].numpy().shape
        data_length = len(self.data[0].numpy().flatten())
        for i in range(data_length):
            tmp0, tmp1 = self.data[0].numpy().flatten(), self.data[0].numpy().flatten()
            tmp0[i] += self.gap
            tmp1[i] -= self.gap
            tmp0 = tmp0.reshape(shape)
            tmp1 = tmp1.reshape(shape)
            tmp0, tmp1 = paddle.to_tensor(tmp0), paddle.to_tensor(tmp1)
            r1, r2 = self.solve_loss(tmp0), self.solve_loss(tmp1)
            g = (r1 - r2) / (2 * self.gap)
            gradients.append(g[0])
        return np.array(gradients).reshape(shape)

    def solve_loss(self, inputs):
        """
        solve loss
        """
        loss_inputs = copy.copy(self.data)
        loss_inputs[0] = inputs
        cell = self.func(**self.kwargs)
        res = cell(*loss_inputs)
        loss = paddle.mean(res).numpy()
        return loss

    def run(self, res, *args, **kwargs):
        """
        run test case
        """
        for place in self.places:
            if str(place) == "CPUPlace":
                paddle.set_device("cpu")
            else:
                paddle.set_device("gpu:0")

            self.generate_parameters(*args, **kwargs)

            paddle.set_default_dtype(self.dtype)
            self.process_dtype(self.dtype)
            logging.info("--->>> device: {}; dtype: {}".format(place, self.data[0].dtype))

            dynamic_res, dynamic_grad = self.cal_dynamic(place)
            print("Asdasd")
            numeric_grad = self.numerical_gradients()
            if self.enable_static:
                static_res, static_grad = self.cal_static(place)
                assert np.allclose(dynamic_res.numpy(), static_res)
                assert np.allclose(dynamic_grad.numpy(), static_grad)
            assert np.allclose(res, dynamic_res.numpy())
            assert np.allclose(dynamic_grad.numpy(), numeric_grad, atol=self.atol)

    def generate_parameters(self, *args, **kwargs):
        """
        Input data processing
        """
        self.kwargs = copy.deepcopy(kwargs)
        length = len(args)
        self.data = []
        for i in range(length):
            if isinstance(args[i], np.ndarray):
                tmp = paddle.to_tensor(args[i], stop_gradient=False)
                self.data.append(tmp)
            elif isinstance(args[i], (list, tuple)):

                def solve_tuple(item):
                    if isinstance(item, np.ndarray):
                        return paddle.to_tensor(item, stop_gradient=False)
                    elif isinstance(item, (list, tuple)):
                        item = list(item)
                        item_len = len(item)
                        for j in range(item_len):
                            item[j] = solve_tuple(item[j])
                    else:
                        return item
                    return item

                res = solve_tuple(args[i])
                self.data.append(res)
            else:
                logging.info("rnn_base can not solve inputs size")

    def process_dtype(self, dtype):
        """
        processing inputs dtype
        """
        l_data = len(self.data)
        for i in range(l_data):
            if isinstance(self.data[i], paddle.Tensor):
                if self.data[i].dtype == paddle.float32 or self.data[i].dtype == paddle.float64:
                    self.data[i] = self.data[i].astype(dtype)
            elif isinstance(self.data[i], (list, tuple)):

                def solve_tuple(item):
                    if isinstance(item, paddle.Tensor):
                        if item.dtype == paddle.float32 or item.dtype == paddle.float64:
                            return item.astype(dtype)
                        else:
                            return item
                    elif isinstance(item, (list, tuple)):
                        item = list(item)
                        leng = len(item)
                        for j in range(leng):
                            item[j] = solve_tuple(item[j])
                    else:
                        return item
                    return item

                self.data[i] = solve_tuple(self.data[i])
