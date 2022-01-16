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
        self.types = [np.float32]
        self.seed = RANDOM_SEED
        self.enable_backward = False

    def _check_dtype(self, res, data, **kwargs):
        """
        check dtype
        Args:
            res: res[0] result of cpu, res[1] result of gpu
            **kwargs: kwargs

        Returns:
            Assertion
        """
        # check whether dtype is wrong, but it won't stop test cases behind, it will report at last
        # remember user enable_backward
        backward_tag = self.enable_backward
        assert len(res) == 2

        for place in self.places:
            self.place = place
            logging.info("[Place] is ===============================>>>>>>>>" + str(self.place))
            tag = True
            for dtype in self.types:
                # 判断是否应该做反向计算，只有float类型的需要反向，同时如果api明确没有反向，需要根据配置进行反向截断。
                if dtype in self.backward_dtype and backward_tag:
                    self.enable_backward = True
                else:
                    self.enable_backward = False
                logging.info("[test dtype] " + self.__class__.__name__ + str(dtype))
                try:
                    self.dtype = dtype
                    if str(place) in ["Place(cpu)", "CPUPlace"]:
                        self._baserun(res[0], data, **kwargs)
                    else:
                        self._baserun(res[1], data, **kwargs)
                except Exception as e:
                    logging.error("[test dtype] " + self.__class__.__name__ + str(dtype) + " failed!!!")
                    tag = False
                    # assume(tag, "[Place {}] type check Error {}".format(str(self.place), str(dtype)))
                    assert tag, "[Place {}] type check Error {}".format(str(self.place), str(dtype))
                    if self.debug:
                        logging.error(e)
        self.dtype = None
        self.enable_backward = backward_tag

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
                if str(self.place) in ["Place(cpu)", "CPUPlace"]:
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
            if str(self.place) in ["Place(cpu)", "CPUPlace"]:
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

    def run(self, res_list, data=None, **kwargs):
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

        for place in self.places:
            self.place = place
            paddle.disable_static(self.place)
            if str(self.place) in ["Place(cpu)", "CPUPlace"]:
                paddle.set_device("cpu")
                res = res_list[0]
            else:
                paddle.set_device("gpu:0")
                res = res_list[1]
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
np_random_tensor_gpu = np.array([[0.00224779, 0.50324494, 0.13526054], [0.16112770, 0.79557019, 0.96897715]])


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
    基础测试， 包括：
    1. 数据类型测试
    2. cpu/gpu测试
    3. 动态图静态图结果正确性验证
    """
    x = randtool("float", 0, 2, [2, 3])
    p = 0.5
    paddle.seed(100)
    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    gpu_res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor_gpu)
    obj.base([res, gpu_res], data=x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout1():
    """
    默认参数结果测试
    """
    x = randtool("float", 0, 2, [2, 3])
    paddle.seed(100)
    p = 0.5  # defult is 0.5
    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    gpu_res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor_gpu)
    obj.run([res, gpu_res], x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout2():
    """
    右边界测试，p=1
    """
    paddle.disable_static()
    x = randtool("float", 0, 2, [2, 3])
    paddle.seed(100)
    p = 1.0  # defult is 0.5
    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    gpu_res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor_gpu)
    obj.run([res, gpu_res], x, p=p)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout3():
    """
    左边界测试，p=0
    """
    x = randtool("float", 0, 2, [2, 3])
    paddle.seed(100)
    p = 0
    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    gpu_res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor_gpu)
    obj.run([res, gpu_res], x, p=p)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout4():
    """
    p为负值异常测试, p=-1
    """
    x = randtool("float", 0, 2, [2, 3])
    obj.exception(etype=ValueError, mode="python", data=x, p=-1)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout5():
    """
    p大于1异常测试, p=2
    """
    x = randtool("float", 0, 2, [2, 3])
    obj.exception(etype=ValueError, mode="python", data=x, p=-2)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout6():
    """
    p值类型错误测试，p = '1' string类型
    """
    x = randtool("float", 0, 2, [2, 3])
    obj.exception(etype=TypeError, mode="python", data=x, p="1")


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout7():
    """
    多维度测试
    """
    _shape = (5, 10, 15, 20)
    paddle.disable_static()
    x = randtool("float", -5, 10, _shape)
    paddle.seed(100)
    paddle.set_device("cpu")
    np_random_tensor = paddle.uniform(_shape, dtype="float32", min=0.0, max=1.0)
    np_random_tensor = np.array(np_random_tensor)

    p = 0.5
    gpu_res = None
    if paddle.device.is_compiled_with_cuda() is True:
        paddle.set_device("gpu")
        paddle.seed(100)
        np_random_tensor_gpu = paddle.uniform(_shape, dtype="float32", min=0.0, max=1.0)
        np_random_tensor_gpu = np.array(np_random_tensor_gpu)
        gpu_res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor_gpu)

    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    obj.run([res, gpu_res], x)


@pytest.mark.api_nn_AlphaDropout_parameters
def test_alpha_dropout8():
    """
    x正负测试范围测试
    """
    x = randtool("float", -100, 100, [2, 3])
    paddle.seed(100)
    p = 0.5
    res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor)
    gpu_res = numpy_alpha_dropout(x, p, random_tensor=np_random_tensor_gpu)
    obj.run([res, gpu_res], x, p=p)
