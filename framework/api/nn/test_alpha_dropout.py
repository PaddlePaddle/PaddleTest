#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
    test AlphaDropout
"""
import logging
from apibase import APIBase, compare, compare_grad
from apibase import randtool
import paddle
import pytest
import numpy as np

RANDOM_SEED = 100


class TestAlphaDropout(APIBase):
    """
    test
    """

    def hook(self):
        self.types = [np.float32, np.float64]
        self.seed = RANDOM_SEED
        self.enable_backward = False

    # 重写base run，确保动态图和静态图的随机数产出相同
    def _baserun(self, res, data=None, **kwargs):
        """
        baserun
        Args:
            res: expect result
            **kwargs: kwargs

        Returns:
            Assertion
        """
        if self.debug:
            # start run paddle dygraph
            if self.dygraph:
                paddle.disable_static(self.place)
                if str(self.place) == "CPUPlace":
                    paddle.set_device("cpu")
                else:
                    paddle.set_device("gpu:0")
                paddle.seed(self.seed)
                logging.info("[start] run " + self.__class__.__name__ + " dygraph")
                self._check_params(res, data, **kwargs)
                dygraph_forward_res = self._dygraph_forward()
                logging.info("dygraph forward result is :")
                if isinstance(dygraph_forward_res, (list)):
                    compare(dygraph_forward_res, res, self.delta, self.rtol)
                    logging.info(dygraph_forward_res)
                else:
                    compare(dygraph_forward_res.numpy(), res, self.delta, self.rtol)
                    logging.info(dygraph_forward_res.numpy())
                if self.enable_backward:
                    dygraph_backward_res = self._dygraph_backward(dygraph_forward_res)
                    logging.info("[dygraph grad]")
                    logging.info(dygraph_backward_res)
                paddle.enable_static()
            if self.static:
                # start run paddle static
                logging.info("[start] run " + self.__class__.__name__ + " static")
                if self.enable_backward:
                    static_forward_res, static_backward_res = self._static_forward(res, data, **kwargs)
                    logging.info("static forward result is :")
                    logging.info(static_forward_res)
                    logging.info("[static grad]")
                    logging.info(static_backward_res)
                else:
                    static_forward_res = self._static_forward(res, data, **kwargs)
                    logging.info("static forward result is :")
                    logging.info(static_forward_res)
                compare(static_forward_res, res, self.delta, self.rtol)
                # start run torch
            if self.enable_backward:
                grad = self.compute_grad(res, data, **kwargs)
                logging.info("[numeric grad]")
                logging.info(grad)
                if self.static and self.dygraph:
                    compare_grad(static_backward_res, dygraph_backward_res, mode="both", no_grad_var=self.no_grad_var)
                if self.dygraph:
                    compare_grad(
                        dygraph_backward_res,
                        grad,
                        mode="dygraph",
                        delta=self.delta,
                        rtol=self.rtol,
                        no_grad_var=self.no_grad_var,
                    )
                if self.static:
                    compare_grad(
                        static_backward_res,
                        grad,
                        mode="static",
                        delta=self.delta,
                        rtol=self.rtol,
                        no_grad_var=self.no_grad_var,
                    )
        else:
            # start run paddle dygraph
            logging.info("[start] run " + self.__class__.__name__ + " dygraph")
            paddle.disable_static(self.place)
            if str(self.place) == "CPUPlace":
                paddle.set_device("cpu")
            else:
                paddle.set_device("gpu:0")
            paddle.seed(self.seed)
            self._check_params(res, data, **kwargs)
            dygraph_forward_res = self._dygraph_forward()
            if isinstance(dygraph_forward_res, (list)):
                compare(dygraph_forward_res, res, self.delta, self.rtol)
            else:
                compare(dygraph_forward_res.numpy(), res, self.delta, self.rtol)
            if self.enable_backward:
                dygraph_backward_res = self._dygraph_backward(dygraph_forward_res)
            paddle.enable_static()
            # start run paddle static
            paddle.seed(self.seed)
            logging.info("[start] run " + self.__class__.__name__ + " static")
            if self.enable_backward:
                static_forward_res, static_backward_res = self._static_forward(res, data, **kwargs)
            else:
                static_forward_res = self._static_forward(res, data, **kwargs)
            compare(static_forward_res, res, self.delta, self.rtol)
            # start run torch
            if self.enable_backward:
                grad = self.compute_grad(res, data, **kwargs)
                compare_grad(static_backward_res, dygraph_backward_res, mode="both", no_grad_var=self.no_grad_var)
                compare_grad(
                    dygraph_backward_res,
                    grad,
                    mode="dygraph",
                    delta=self.delta,
                    rtol=self.rtol,
                    no_grad_var=self.no_grad_var,
                )
                compare_grad(
                    static_backward_res,
                    grad,
                    mode="static",
                    delta=self.delta,
                    rtol=self.rtol,
                    no_grad_var=self.no_grad_var,
                )

    def run(self, res, data=None, **kwargs):
        """
        run
        Args:
            res: expect result
            **kwargs: kwargs

        Returns:
            Assertion
        """
        # 取默认type
        if self.dtype is None:
            if np.float64 in self.types:
                self.dtype = np.float64
            else:
                self.dtype = self.types[0]
        if self.debug:
            for place in self.places:
                self.place = place
                logging.info("[Place] is ===============================>>>>>>>>" + str(self.place))
                # start run paddle dygraph
                if self.dygraph:
                    paddle.disable_static(self.place)
                    if str(self.place) == "CPUPlace":
                        paddle.set_device("cpu")
                    else:
                        paddle.set_device("gpu:0")
                    logging.info("[start] run " + self.__class__.__name__ + " dygraph")
                    paddle.seed(self.seed)
                    self._check_params(res, data, **kwargs)
                    dygraph_forward_res = self._dygraph_forward()
                    logging.info("dygraph forward result is :")
                    if isinstance(dygraph_forward_res, (list)):
                        compare(dygraph_forward_res, res, self.delta, self.rtol)
                        logging.info(dygraph_forward_res)
                    else:
                        compare(dygraph_forward_res.numpy(), res, self.delta, self.rtol)
                        logging.info(dygraph_forward_res.numpy())
                    if self.enable_backward:
                        dygraph_backward_res = self._dygraph_backward(dygraph_forward_res)
                        logging.info("[dygraph grad]")
                        logging.info(dygraph_backward_res)
                    paddle.enable_static()
                if self.static:

                    # start run paddle static
                    logging.info("[start] run " + self.__class__.__name__ + " static")
                    if self.enable_backward:
                        static_forward_res, static_backward_res = self._static_forward(res, data, **kwargs)
                        logging.info("static forward result is :")
                        logging.info(static_forward_res)
                        logging.info("[static grad]")
                        logging.info(static_backward_res)
                    else:
                        static_forward_res = self._static_forward(res, data, **kwargs)
                        logging.info("static forward result is :")
                        logging.info(static_forward_res)
                    compare(static_forward_res, res, self.delta, self.rtol)
                    # start run torch
                if self.enable_backward:
                    paddle.seed(100)
                    grad = self.compute_grad(res, data, **kwargs)
                    logging.info("[numeric grad]")
                    logging.info(grad)
                    if self.static and self.dygraph:
                        compare_grad(
                            static_backward_res, dygraph_backward_res, mode="both", no_grad_var=self.no_grad_var
                        )
                    if self.dygraph:
                        compare_grad(
                            dygraph_backward_res,
                            grad,
                            mode="dygraph",
                            delta=self.delta,
                            rtol=self.rtol,
                            no_grad_var=self.no_grad_var,
                        )
                    if self.static:
                        compare_grad(
                            static_backward_res,
                            grad,
                            mode="static",
                            delta=self.delta,
                            rtol=self.rtol,
                            no_grad_var=self.no_grad_var,
                        )
        else:
            for place in self.places:
                self.place = place
                paddle.disable_static(self.place)
                if str(self.place) == "CPUPlace":
                    paddle.set_device("cpu")
                else:
                    paddle.set_device("gpu:0")
                logging.info("[Place] is ===============================>>>>>>>>" + str(self.place))
                # start run paddle dygraph
                logging.info("[start] run " + self.__class__.__name__ + " dygraph")
                paddle.disable_static(self.place)
                paddle.seed(self.seed)
                self._check_params(res, data, **kwargs)
                dygraph_forward_res = self._dygraph_forward()
                if isinstance(dygraph_forward_res, (list)):
                    compare(dygraph_forward_res, res, self.delta, self.rtol)
                else:
                    compare(dygraph_forward_res.numpy(), res, self.delta, self.rtol)
                if self.enable_backward:
                    dygraph_backward_res = self._dygraph_backward(dygraph_forward_res)
                paddle.enable_static()
                # start run paddle static
                paddle.seed(100)
                logging.info("[start] run " + self.__class__.__name__ + " static")
                if self.enable_backward:
                    static_forward_res, static_backward_res = self._static_forward(res, data, **kwargs)
                else:
                    static_forward_res = self._static_forward(res, data, **kwargs)
                compare(static_forward_res, res, self.delta, self.rtol)
                # start run torch
                if self.enable_backward:
                    grad = self.compute_grad(res, data, **kwargs)
                    compare_grad(static_backward_res, dygraph_backward_res, mode="both", no_grad_var=self.no_grad_var)
                    compare_grad(
                        dygraph_backward_res,
                        grad,
                        mode="dygraph",
                        delta=self.delta,
                        rtol=self.rtol,
                        no_grad_var=self.no_grad_var,
                    )
                    compare_grad(
                        static_backward_res,
                        grad,
                        mode="static",
                        delta=self.delta,
                        rtol=self.rtol,
                        no_grad_var=self.no_grad_var,
                    )


obj = TestAlphaDropout(paddle.nn.AlphaDropout)
np_random_tensor = np.array([[0.55355287, 0.20714243, 0.01162981], [0.51577556, 0.36369765, 0.26091650]])


def numpy_alpha_dropout(x, p, random_tensor, training=True):
    """
    numpy version alpha dropout
    """

    def f_scale(x, scale=1.0, bias=0.0):
        out = scale * x + bias
        return out

    if training:
        if p == 1:
            return f_scale(x, scale=0.0)
        # get transformation params
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale
        a = ((1 - p) * (1 + p * alpha_p ** 2)) ** -0.5
        b = -a * alpha_p * p

        dtype = x.dtype
        input_shape = x.shape

        p = np.ones(input_shape, dtype="float32") * p
        keep_mask = np.greater_equal(random_tensor, p)
        keep_mask = keep_mask.astype(dtype)
        drop_mask = np.subtract(np.ones(shape=input_shape), keep_mask)

        b = np.ones(input_shape, dtype=dtype) * b
        y = x * keep_mask + f_scale(drop_mask, scale=alpha_p)
        res = f_scale(y, scale=a) + b
        return res
    else:
        return x


@pytest.mark.api_nn_AlphaDropout_vartype
def test_alpha_dropout_base():
    """
    base
    """
    x = randtool("float", 0, 2, [2, 3])
    p = 0.5
    paddle.seed(100)
    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    obj.base(res, data=x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout1():
    """
    default
    """
    x = randtool("float", 0, 2, [2, 3])
    paddle.seed(100)
    p = 0.5  # defult is 0.5
    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    obj.run(res, x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p=1
    """
    x = randtool("float", 0, 2, [2, 3])
    paddle.seed(100)
    p = 1.0  # defult is 0.5
    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    obj.run(res, x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p=0
    """
    x = randtool("float", 0, 2, [2, 3])
    paddle.seed(100)
    p = 0.0  # defult is 0.5
    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    obj.run(res, x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p = -1
    """
    x = randtool("float", 0, 2, [2, 3])
    obj.exception(etype=ValueError, mode="python", data=x, p=-1)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p = 2, 使用exception接口
    """
    x = randtool("float", 0, 2, [2, 3])
    obj.exception(etype=ValueError, mode="python", data=x, p=-2)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    p = '1'
    """
    x = randtool("float", 0, 2, [2, 3])
    obj.exception(etype=TypeError, mode="python", data=x, p="1")
