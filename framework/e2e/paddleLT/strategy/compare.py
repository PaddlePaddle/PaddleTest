#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
runner
"""

import logging
import numpy as np
import paddle


def base_compare(result, expect, res_name, exp_name, logger, delta=1e-10, rtol=1e-10):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :param rtol: 相对误差
    :return:
    """
    if isinstance(result, str):
        raise Exception("result is exception !!!")
    if isinstance(expect, str):
        raise Exception("expect is exception !!!")

    if isinstance(expect, paddle.Tensor) or isinstance(expect, np.ndarray):
        if isinstance(result, paddle.Tensor):
            result = result.numpy()
        if isinstance(expect, paddle.Tensor):
            expect = expect.numpy()
        # res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # # 出错打印错误数据
        # if res is False:
        #     diff = abs(result - expect)
        #     # logger.error("{} is: {}".format(exp_name, expect))
        #     # logger.error("{} is: {}".format(res_name, result))
        #     logger.error("{} and {} has diff! max diff: {}".format(exp_name, res_name, np.amax(diff)))

        np.testing.assert_allclose(actual=result, desired=expect, atol=delta, rtol=rtol, equal_nan=True)

        if result.dtype != expect.dtype:
            logger.error(
                "Different output data types! res type is: {}, and expect type is: {}".format(
                    result.dtype, expect.dtype
                )
            )
        # assert res
        assert result.shape == expect.shape
        assert result.dtype == expect.dtype
    elif isinstance(expect, list) or isinstance(expect, tuple):
        for i, element in enumerate(expect):
            if isinstance(result, (np.generic, np.ndarray)) or isinstance(result, paddle.Tensor):
                if i > 0:
                    break
                base_compare(
                    result=result,
                    expect=expect[i],
                    res_name=res_name + "[{}]".format(str(i)),
                    exp_name=exp_name + "[{}]".format(str(i)),
                    logger=logger,
                    delta=delta,
                    rtol=rtol,
                )
            else:
                base_compare(
                    result=result[i],
                    expect=expect[i],
                    res_name=res_name + "[{}]".format(str(i)),
                    exp_name=exp_name + "[{}]".format(str(i)),
                    logger=logger,
                    delta=delta,
                    rtol=rtol,
                )
    elif isinstance(expect, (bool, int, float)):
        assert expect == result
    elif expect is None:
        pass
    else:
        raise Exception("expect is unknown data struction in compare_tool!!!")
