#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
api compare test
"""
from inspect import isclass
import random
import numpy as np
import paddle
import torch
from competitor_test.tools import FrontAPIBase, compare, solve_tuple, TORCHDTYPE

# import logging
from utils.logger import Logger


class CompetitorCompareTest(object):
    """
    API test
    compare paddle with competitive product
    """

    def __init__(self, paddle_api, torch_api):
        self.seed = 33
        self.enable_backward = True
        self.debug = True
        self.paddle_api = paddle_api
        self.torch_api = torch_api
        self.compare_dict = None
        self.paddle_inputs = dict()
        self.paddle_param = dict()
        self.torch_inputs = dict()
        self.torch_param = dict()
        self.places = None
        self._set_seed()
        self._set_place()
        self.ignore_var = []
        self.types = None
        self.logger = Logger("CompetitorCompareTest")
        self.hook()
        # 日志等级
        # if self.debug:
        #     logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
        # else:
        #     logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

    def hook(self):
        """
        hook
        """
        pass

    def run(self, data, data_c):
        """
        run paddle and competitor api
        """
        for place in self.places:
            self.logger.get_log().info(
                "[{}]start compare [paddle]{} and [torch]{}".format(
                    place, str(self.paddle_api.__name__), str(self.torch_api.__name__)
                )
            )

            paddle.set_device(place)
            if place == "cpu":
                torch.device("cpu")
            else:
                torch.device(0)
            for dtype in self.types:
                if dtype in ["float16", "float32", "float64"]:
                    paddle.set_default_dtype(dtype)
                    torch.set_default_dtype(TORCHDTYPE.get(dtype))
                if self.enable_backward:
                    paddle_forward_res, paddle_backward_res = self._run_paddle(data, dtype)
                    torch_forward_res, torch_backward_res = self._run_torch(data_c, dtype)
                    paddle_forward_res = self._paddle_to_numpy(paddle_forward_res)
                    paddle_backward_res = self._paddle_to_numpy(paddle_backward_res)
                    torch_forward_res = self._torch_to_numpy(torch_forward_res)
                    torch_backward_res = self._torch_to_numpy(torch_backward_res)
                    compare(paddle_forward_res, torch_forward_res)
                    self.logger.get_log().info("[{}] data type forward result compare success!".format(dtype))
                    compare(paddle_backward_res, torch_backward_res)
                    self.logger.get_log().info("[{}] data backward result compare success!".format(dtype))
                else:
                    paddle_forward_res = self._run_paddle(data, dtype)
                    torch_forward_res = self._run_torch(data_c, dtype)
                    compare(paddle_forward_res, torch_forward_res)
                    self.logger.get_log().info("[{}] data type forward result compare success!".format(dtype))

    def _run_paddle(self, data, dtype):
        """
        run paddle
        """
        self._set_paddle_param(data, dtype)
        self.paddle_api = self._settle_api(self.paddle_api)
        res = self._paddle_forward()
        if self.debug:
            self.logger.get_log().info("[{}] data type [paddle] api forward result:\n ".format(dtype, res))
        backward_res = None
        if self.enable_backward:
            backward_res = self._paddle_backward(res)
            if self.debug:
                self.logger.get_log().info(
                    "[{}] data type [paddle] api backward result:\n {}".format(dtype, backward_res)
                )
        return res, backward_res if self.enable_backward else res

    def _run_torch(self, data_c, dtype):
        """
        run torch
        """
        self._set_torch_param(data_c, dtype)
        self.torch_api = self._settle_api(self.torch_api)
        res = self._torch_forward()
        if self.debug:
            self.logger.get_log().info("[{}] data type [torch] api forward result:\n {}".format(dtype, res))
        backward_res = None
        if self.enable_backward:
            backward_res = self._torch_backward(res)
            if self.debug:
                self.logger.get_log().info("[{}] data type [torch] api backward result:\n {}".format(dtype, res))
        return res, backward_res if self.enable_backward else res

    def _paddle_to_numpy(self, t):
        """
        convert paddle.Tensor to ndarry
        """
        if isinstance(t, paddle.Tensor):
            return paddle.Tensor.numpy(t)
        elif isinstance(t, (list, tuple)):
            return solve_tuple(t, paddle.Tensor, paddle.Tensor.numpy)

    def _torch_to_numpy(self, t):
        """
        convert torch.Tensor to ndarry
        """
        if isinstance(t, torch.Tensor):
            return torch.detach(t).numpy()
        elif isinstance(t, (list, tuple)):
            convert_numpy = lambda x: torch.detach(x).numpy()
            return solve_tuple(t, torch.Tensor, convert_numpy)

    def _set_seed(self):
        """
        init seed
        :return:
        """
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        use_cuda = paddle.is_compiled_with_cuda()
        if use_cuda:
            torch.cuda.manual_seed(self.seed)

    def _set_place(self):
        """
        init place
        :return:
        """
        if paddle.is_compiled_with_cuda() is True:
            if torch.cuda.is_available() is True:
                self.places = ["cpu", "gpu:0"]
            else:
                raise EnvironmentError
        else:
            # default
            self.places = ["cpu"]

    def _set_paddle_param(self, data: dict, dtype):
        """
        set paddle params
        """
        if len(data) == 1:
            data["params"] = {}
        for k, v in data["inputs"].items():
            if isinstance(v, (np.generic, np.ndarray)):
                self.paddle_inputs[k] = paddle.to_tensor(v, stop_gradient=False, dtype=dtype)
            elif isinstance(v, (list, tuple)):
                self.paddle_inputs = []
                for i, j in enumerate(v):
                    if isinstance(j, (np.generic, np.ndarray)):
                        self.paddle_inputs[k][i] = paddle.to_tensor(j, stop_gradient=False, dtype=dtype)
                    else:
                        self.logger.get_log().error("Paddle inputs type cannot convert")
                        raise TypeError
            else:
                self.paddle_inputs[k] = v

        for k, v in data["params"].items():
            if isinstance(v, (np.generic, np.ndarray)):
                self.paddle_param[k] = paddle.to_tensor(v, dtype=dtype)
            else:
                self.paddle_param[k] = v

    def _set_torch_param(self, data: dict, dtype):
        """
        set torch params
        """
        if len(data) == 1:
            data["params"] = {}
        for k, v in data["inputs"].items():
            if isinstance(v, (np.generic, np.ndarray)):
                self.torch_inputs[k] = torch.tensor(v, requires_grad=True, dtype=TORCHDTYPE.get(dtype))
            elif isinstance(v, (list, tuple)):
                self.torch_inputs = []
                for i, j in enumerate(v):
                    if isinstance(j, (np.generic, np.ndarray)):
                        self.torch_inputs[k][i] = torch.tensor(j, requires_grad=True, dtype=TORCHDTYPE.get(dtype))
                    else:
                        self.logger.get_log().error("torch inputs type cannot convert")
                        raise TypeError
            else:
                self.torch_inputs[k] = v

        for k, v in data["params"].items():
            if isinstance(v, (np.generic, np.ndarray)):
                self.torch_param[k] = paddle.to_tensor(v, dtype=TORCHDTYPE.get(dtype))
            else:
                self.torch_param[k] = v

    def _settle_api(self, api):
        """
        set api
        """
        if not isclass(api):
            return api
        else:
            obj = FrontAPIBase(api)
            return obj.encapsulation

    def _paddle_forward(self):
        """
        paddle forward calculate
        """
        inputs = [v for k, v in self.paddle_inputs.items()]
        # print(inputs)
        return self.paddle_api(*inputs, **self.paddle_param)

    def _torch_forward(self):
        """
        torch forward calculate
        """
        inputs = [v for k, v in self.torch_inputs.items()]
        # print(inputs)
        return self.torch_api(*inputs, **self.torch_param)

    def _paddle_backward(self, res):
        """
        paddle backward calculate
        """
        # loss = paddle.mean(res)
        loss = self._paddle_loss(res)
        loss.backward()
        grad = {}
        for k, v in self.paddle_inputs.items():
            if isinstance(v, paddle.Tensor) and k not in self.ignore_var:
                grad[k] = v.gradient()
            elif isinstance(v, (list, tuple)):
                grad[k] = []
                for i, j in enumerate(v):
                    if isinstance(j, paddle.Tensor):
                        grad[k].append(v[i].gradient())
        return grad

    def _torch_backward(self, res):
        """
        torch backward calculate
        """
        loss = self._torch_loss(res)
        loss.backward()
        grad = {}

        for k, v in self.torch_inputs.items():
            if isinstance(v, torch.Tensor) and k not in self.ignore_var:
                grad[k] = v.grad
            elif isinstance(v, (list, tuple)):
                grad[k] = []
                for i, j in enumerate(v):
                    if isinstance(j, paddle.Tensor):
                        grad[k].append(v[i].grad)
        return grad

    def _paddle_loss(self, res):
        """
        calculate paddle loss
        """
        if isinstance(res, paddle.Tensor):
            return paddle.sum(res)
        elif isinstance(res, (list, tuple)):
            res = solve_tuple(res, paddle.Tensor, paddle.sum)
            return sum(res)
        else:
            raise ValueError

    def _torch_loss(self, res):
        """
        calculate torch loss
        """
        if isinstance(res, torch.Tensor):
            return torch.sum(res)
        elif isinstance(res, (list, tuple)):
            res = solve_tuple(res, torch.Tensor, torch.sum)
            return sum(res)
        else:
            raise ValueError


# "params": {"in_channels":4, "out_channels":6, "kernel_size": (3, 3),}
# "params": {"in_channels":4, "out_channels":6, "kernel_size": (3, 3),}
