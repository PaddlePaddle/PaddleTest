#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.AlphaDropout
"""
import logging
import numpy as np
import pytest
import paddle
from apibase import APIBase
from apibase import randtool, compare, compare_grad


class TestAlphaDropout(APIBase):
    """
    test
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.enable_backward = False
        # self.enable_backward = True
        self.debug = False

    def _baserun(self, res, data=None, **kwargs):
        """
         Compared with the parent method, add one line of code

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

            # After adding the following line, the random_tensor generated
            # in AlphaDropout under static mode will be the same as that
            # under dynamic mode.
            # Compared with the parent method, add the following line
            paddle.seed(self.seed)
            # start run paddle static
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
        Compared with the parent method, add one line of code
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
                # After adding the following line, the random_tensor generated
                # in AlphaDropout under static mode will be the same as that
                # under dynamic mode.
                # Compared with the parent method, add the following line
                paddle.seed(self.seed)
                # start run paddle static
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


def alpha_dropout_using_numpy(x, p=0.5, training=True):
    """
    implementation of alpha dropout using numpy
    """
    if not isinstance(p, (float, int)):
        raise TypeError("p argument should be a float or int")
    if p < 0 or p > 1:
        raise ValueError("p argument should between 0 and 1")

    def scale_func(x, scale=1.0, bias=0.0):
        return x * scale + bias

    if training:
        if p == 1:
            return scale_func(x, scale=0.0)
        # get transformation params
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale
        a = ((1 - p) * (1 + p * alpha_p ** 2)) ** -0.5
        b = -a * alpha_p * p

        dtype = x.dtype
        input_shape = x.shape

        random_tensor = paddle.uniform(shape=input_shape, min=0.0, max=1.0)
        p = np.full(shape=[1], fill_value=p, dtype="float32")
        keep_mask = np.greater_equal(random_tensor.numpy(), p)
        keep_mask = keep_mask.astype(dtype)
        drop_mask = np.subtract(np.ones(shape=input_shape, dtype=dtype), keep_mask)
        # apply mask
        b = np.full(shape=[1], fill_value=b, dtype=dtype)
        y = np.add(np.multiply(x, keep_mask), scale_func(drop_mask, scale=alpha_p))
        res = np.add(scale_func(y, scale=a), b)
        return res
    else:  # test
        return x


@pytest.mark.api_nn_AlphaDropout_vartype
def test_alpha_dropout_base():
    """
    base
    """
    x = randtool("float", low=-10, high=10, shape=[3, 4])
    p = 0.5
    # Adding the following 2 lines of code ensures that
    # random_tensor generated in alpha_dropout_using_numpy will
    # be the same as random_tensor generated in AlphaDropout
    # which will be called in obj.base
    paddle.disable_static()
    paddle.seed(obj.seed)
    res = alpha_dropout_using_numpy(x, p=p, training=True)
    obj.base(res=res, data=x, p=p)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout1():
    """
    default p = 0.5
    """
    x = randtool("float", low=-10, high=10, shape=[3, 4])
    p = 0.5
    paddle.disable_static()
    paddle.seed(obj.seed)
    res = alpha_dropout_using_numpy(x, p=p, training=True)
    obj.run(res=res, data=x, p=p)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
     p = 1
    """
    x = randtool("float", low=-10, high=10, shape=[3, 4])
    p = 1
    paddle.disable_static()
    paddle.seed(obj.seed)
    res = alpha_dropout_using_numpy(x, p=p, training=True)
    obj.run(res=res, data=x, p=p)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout3():
    """
     p = 0
    """
    x = randtool("float", low=-10, high=10, shape=[3, 4])
    p = 0
    paddle.disable_static()
    paddle.seed(obj.seed)
    res = alpha_dropout_using_numpy(x, p=p, training=True)
    obj.run(res=res, data=x, p=p)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout4():
    """
     p = 0.7, input value in [-1000.0, 1000.0),
     input shape is [3, 4, 5, 6, 7]
    """
    x = randtool("float", low=-1000, high=1000, shape=[3, 4, 5, 6, 7])
    p = 0.7
    paddle.disable_static()
    paddle.seed(obj.seed)
    res = alpha_dropout_using_numpy(x, p=p, training=True)
    obj.run(res=res, data=x, p=p)


@pytest.mark.api_nn_AlphaDropout_exception
def test_alpha_dropout5():
    """
    check invalid p=-0.5
    """
    x = randtool("float", low=-10, high=10, shape=[3, 4, 5])
    obj.exception(etype=ValueError, mode="python", data=x, p=-0.5)


@pytest.mark.api_nn_AlphaDropout_exception
def test_alpha_dropout6():
    """
    check invalid p=1.1
    """
    x = randtool("float", low=-10, high=10, shape=[3, 4, 5])
    obj.exception(etype=ValueError, mode="python", data=x, p=1.1)


@pytest.mark.api_nn_AlphaDropout_exception
def test_alpha_dropout7():
    """
    check invalid p='0.7', p is a string
    """
    x = randtool("float", low=-10, high=10, shape=[3, 4, 5])
    obj.exception(etype=TypeError, mode="python", data=x, p="0.7")
