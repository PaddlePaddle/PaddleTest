#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
jittrans.py
"""
import os
import logging
from inspect import isclass
import shutil
import numpy as np
import paddle
import paddle.inference as paddle_infer
from paddle.static import InputSpec
from utils.weaktrans import WeakTrans


def naive_func(a, in_params, func):
    """用于动转静的方法"""
    layer = eval(func)(**a, **in_params)
    return layer


class BuildClass(paddle.nn.Layer):
    """
    用于动转静的nn.Layer
    """

    def __init__(self, in_params, func):
        super(BuildClass, self).__init__()
        self.func = eval(func)(**in_params)

    def forward(self, inputs):
        """
        forward
        """
        x = self.func(*inputs)
        return x


class BuildFunc(paddle.nn.Layer):
    """
    用于动转静的nn.Layer
    """

    def __init__(self, in_params, func):
        super(BuildFunc, self).__init__()
        self.func = eval(func)
        self._params = in_params

    def forward(self, inputs):
        """
        forward
        """
        x = self.func(**inputs, **self._params)
        return x


class BuildJitClass(paddle.nn.Layer):
    """
    用于动转静的nn.Layer
    """

    def __init__(self, in_params, func):
        super(BuildJitClass, self).__init__()
        self.func = eval(func)(**in_params)

    @paddle.jit.to_static
    def forward(self, inputs):
        """
        forward
        """
        x = self.func(*inputs)
        return x


class BuildJitFunc(paddle.nn.Layer):
    """
    用于动转静的nn.Layer
    """

    def __init__(self, in_params, func):
        super(BuildJitFunc, self).__init__()
        self.func = eval(func)
        self._params = in_params

    @paddle.jit.to_static
    def forward(self, inputs):
        """
        forward
        """
        x = self.func(**inputs, **self._params)
        return x


class Framework(object):
    """framework"""

    PADDLE = "paddle"
    TORCH = "pytorch"


class JitTrans(WeakTrans):
    """base jit test framework"""

    def __init__(self, case, default_type=np.float32, seed=None):
        """init"""
        super(JitTrans, self).__init__(case, default_type=np.float32, seed=None)
        if paddle.is_compiled_with_cuda():
            paddle.set_device("gpu")
        else:
            paddle.set_device("cpu")
        self.atol = 1e-5
        self.rtol = 1e-6
        # 忽略全部测试的api
        self.ignore_api = [
            "paddle.add_n",
            "paddle.all",
            "paddle.is_tensor",
            "paddle.to_tensor",
            "paddle.tolist",
            "paddle.where",
            "paddle.element_size",
        ]
        # 忽略全部测试的某些api的某些case
        self.ignore_case = []
        # 忽略GPU测试的api
        self.ignore_gpu = []
        # 忽略CPU测试的api
        self.ignore_cpu = []
        # 忽略jit_save测试的api
        self.ignore_api_jit_save = ["paddle.broadcast_shape", "paddle.rank"]
        # 忽略jit_save测试的case
        self.ignore_case_jit_save = []
        # 忽略jit_load测试的api
        self.ignore_api_jit_load = [
            "paddle.allclose",
            "paddle.any",
            "paddle.equal",
            "paddle.equal_all",
            "paddle.greater_equal",
            "paddle.greater_than",
            "paddle.less_equal",
            "paddle.less_than",
            "paddle.logical_and",
            "paddle.logical_not",
            "paddle.logical_or",
            "paddle.logical_xor",
            "paddle.not_equal",
            "paddle.is_empty",
            "paddle.isfinite",
            "paddle.isinf",
            "paddle.isnan",
            "paddle.isclose",
        ]
        # 忽略jit_load测试的某些api的某些case
        self.ignore_case_jit_load = ["conj0"]
        # InputSpec测试时, 设定输入Tensor[0].shape=None, 但以下api的Tensor[0].shape保持原数值
        self.ignore_none_shape_api = [
            "paddle.addmm",
            "paddle.nn.GRU",
            "paddle.nn.SimpleRNN",
            "paddle.scatter_nd",
            "paddle.linalg.cholesky_solve",
            "paddle.gcd",
            "paddle.take_along_axis",
        ]
        # InputSpec测试时, 设定输入Tensor[0].shape=None, 但以下case的Tensor[0].shape保持原数值
        self.ignore_none_shape_case = ["unbind", "unstack"]
        # self.in_tensor, self.in_params, self.func = self.get_func_params("paddle")
        self.in_tensor = self.get_inputs("paddle")
        self.none_shape_tensor_list = []
        for k, v in self.in_tensor.items():
            if isinstance(v, (np.generic, np.ndarray)):
                self.in_tensor[k] = paddle.to_tensor(v, dtype=v.dtype)
                self.none_shape_tensor_list.append(k)

        self.in_params = self.get_params("paddle")
        for k in list(self.in_params.keys()):
            if isinstance(self.in_params[k], (np.generic, np.ndarray)):
                self.in_tensor[k] = paddle.to_tensor(self.in_params[k], dtype=self.in_params[k].dtype)
                del self.in_params[k]

        self.func = self.get_func("paddle")

        if isclass(eval(self.func)):
            self.func_type = "class"
        else:
            self.func_type = "func"

        self.jit_save_path = os.path.join(os.getcwd(), "jit_save", self.case_name)

        if not os.path.exists(os.path.join(os.getcwd(), "jit_save")):
            os.mkdir(os.path.join(os.getcwd(), "jit_save"))

        if os.path.exists(self.jit_save_path):
            shutil.rmtree(self.jit_save_path)
        os.mkdir(self.jit_save_path)

    def sort_intensor(self):
        """对输入进行排序，构建一个新的输入list"""
        tensor_dict = self.in_tensor
        inputs_key = sorted(tensor_dict.keys())

        inputs_value = []
        for k in inputs_key:
            inputs_value.append(tensor_dict[k])
        return inputs_value

    def mk_dict_spec(self):
        """根据输入的dict, 生成InputSpec"""
        input_spec_dict = {}
        for k, v in self.in_tensor.items():
            v_shape = v.shape
            v_dtype = v.dtype
            if (
                k in self.none_shape_tensor_list
                and self.func not in self.ignore_none_shape_api
                and self.case_name not in self.ignore_none_shape_case
            ):
                v_shape[0] = None
            input_spec_dict[k] = InputSpec(shape=v_shape, dtype=v_dtype, name=k)
        return input_spec_dict

    def mk_list_spec(self):
        """根据输入的dict, 生成InputSpec"""
        input_spec_list = []
        for k, v in self.in_tensor.items():
            v_shape = v.shape
            v_dtype = v.dtype
            if (
                k in self.none_shape_tensor_list
                and self.func not in self.ignore_none_shape_api
                and self.case_name not in self.ignore_none_shape_case
            ):
                v_shape[0] = None
            input_spec_list.append(InputSpec(shape=v_shape, dtype=v_dtype, name=k))
        return input_spec_list

    def init_test_object(self, method):
        """init jit test obj"""
        if method == "BuildClass" or method == "BuildClassWithInputSpec":
            # 仅实例化一次，防止多次实例化后，因为随机种子不固定导致多个结果值res不相等
            obj = BuildClass(self.in_params, self.func)
            obj.eval()
        elif method == "BuildFunc" or method == "BuildFuncWithInputSpec":
            obj = BuildFunc(self.in_params, self.func)
            obj.eval()
        elif method == "naive_func":
            obj = naive_func

        return obj

    def mk_exp(self, obj, method):
        """获取动态图结果"""
        if method == "BuildClass" or method == "BuildClassWithInputSpec":
            inputs_value = self.sort_intensor()
            exp = obj(inputs_value)
        elif method == "BuildFunc" or method == "BuildFuncWithInputSpec":
            exp = obj(self.in_tensor)
        elif method == "naive_func":
            exp = obj(self.in_tensor, self.in_params, self.func)
        return exp

    def mk_res(self, obj, method):
        """获取静态图结果"""
        if method == "BuildClass":
            inputs_value = self.sort_intensor()
            jit_obj = paddle.jit.to_static(obj)
            res = jit_obj(inputs_value)
        elif method == "BuildClassWithInputSpec":
            inputs_value = self.sort_intensor()
            input_spec = self.mk_list_spec()
            jit_obj = paddle.jit.to_static(obj, input_spec=[input_spec])
            res = jit_obj(inputs_value)
        elif method == "BuildFunc":
            jit_obj = paddle.jit.to_static(obj)
            res = jit_obj(self.in_tensor)
        elif method == "BuildFuncWithInputSpec":
            input_spec = self.mk_dict_spec()
            jit_obj = paddle.jit.to_static(obj, input_spec=[input_spec])
            res = jit_obj(self.in_tensor)
        elif method == "naive_func":
            jit_obj = paddle.jit.to_static(obj)
            res = jit_obj(self.in_tensor, self.in_params, self.func)
        return res

    def jit_save(self, obj, method):
        """
        动转静保存模型，两种情况:
        1. 当self.func为class时，继承nn.Layer构建BuildClass类，
        此时paddle.jit.save将输出 api.pdiparams, api.pdiparams.info, api.pdmodel三个文件
        后续既会比对测试paddle.jit.load加载api.pdmodel的输出结果，也会比对预测库推理部署api.pdiparams和api.pdmodel的结果。
        2. 当self.func为func时，直接构建def函数方法，
        此时paddle.jit.save只输出 api.pdmodel一个文件
        后续只会比对测试paddle.jit.load加载api.pdmodel的输出结果
        """
        if method == "BuildClass":
            inputs_value = self.sort_intensor()
            jit_obj = paddle.jit.to_static(obj)
            exp = jit_obj(inputs_value)  # 此行用于构建inputSpec,不可删除
            paddle.jit.save(jit_obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")))
            print("jit_save use exp: ", exp)
        elif method == "BuildClassWithInputSpec":
            input_spec = self.mk_list_spec()
            paddle.jit.save(
                obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")), input_spec=[input_spec]
            )
        elif method == "BuildFunc":
            jit_obj = paddle.jit.to_static(obj)
            exp = jit_obj(self.in_tensor)  # 此行用于构建inputSpec,不可删除
            paddle.jit.save(jit_obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")))
            print("jit_save use exp: ", exp)
        elif method == "BuildFuncWithInputSpec":
            input_spec = self.mk_dict_spec()
            # jit_obj = paddle.jit.to_static(obj, input_spec=[input_spec])
            paddle.jit.save(
                obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")), input_spec=[input_spec]
            )
        elif method == "naive_func":
            jit_obj = paddle.jit.to_static(obj)
            exp = jit_obj(self.in_tensor, self.in_params, self.func)  # 此行用于构建inputSpec,不可删除
            paddle.jit.save(jit_obj, path=os.path.join(self.jit_save_path, self.get_func("paddle")))
            print("jit_save use exp: ", exp)

    def jit_load(self, method=None):
        """paddle.jit.load加载"""
        jit = paddle.jit.load(os.path.join(self.jit_save_path, self.get_func("paddle")))
        inputs_value = self.sort_intensor()
        res = jit(*inputs_value)
        return res

    def infer_load(self):
        """paddle预测库加载，只会用于测试nn.Layer"""
        config = paddle_infer.Config(
            os.path.join(self.jit_save_path, self.get_func("paddle") + ".pdmodel"),
            os.path.join(self.jit_save_path, self.get_func("paddle") + ".pdiparams"),
        )
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        input_list = self.sort_intensor()

        for i, name in enumerate(input_names):
            input_handle = predictor.get_input_handle(name)
            input_handle.copy_from_cpu(input_list[i].numpy())

        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        infer_res = output_handle.copy_to_cpu()
        return infer_res

    def jit_run(self):
        """测试运行流程"""
        if self.func not in self.ignore_api and self.case_name not in self.ignore_case:
            if self.func_type == "class":
                self.test_method(method="BuildClass")
                self.test_method(method="BuildClassWithInputSpec")  # 需要进一步排查，涉及到某些有问题的api
            else:
                self.test_method(method="naive_func")
                self.test_method(method="BuildFunc")
                self.test_method(method="BuildFuncWithInputSpec")
        else:
            self.logger.get_log().info("(api: {}) (case: {}) ignore all test...".format(self.func, self.case_name))

    def test_method(self, method):
        """jit test method"""
        # self.logger.get_log().info("self.in_tensor is: {}".format(self.in_tensor))
        # self.logger.get_log().info("self.in_params is: {}".format(self.in_params))
        self.logger.get_log().info(
            "start init testing obj ==========> case: {} test_method: {}".format(self.case_name, method)
        )
        obj = self.init_test_object(method)
        self.logger.get_log().info("start make exp ==========> case: {} test_method: {}".format(self.case_name, method))
        exp = self.mk_exp(obj=obj, method=method)
        self.logger.get_log().info("end make exp ==========> case: {} test_method: {}".format(self.case_name, method))

        # self.logger.get_log().info("exp is: {}".format(exp))

        self.logger.get_log().info(
            "start make to_static_res ==========> case: {} test_method: {}".format(self.case_name, method)
        )
        to_static_res = self.mk_res(obj=obj, method=method)

        # self.logger.get_log().info("to_static_res is: {}".format(to_static_res))

        self.logger.get_log().info(
            "start acc comparing ==========> case: {} test_method: {} compare: {} ".format(
                self.case_name, method, "to_static_res and exp"
            )
        )
        compare(to_static_res, exp, self.atol, self.rtol)

        if self.func not in self.ignore_api_jit_save and self.case_name not in self.ignore_case_jit_save:
            self.logger.get_log().info(
                "start jit save ==========> case: {} test_method: {}".format(self.case_name, method)
            )
            self.jit_save(obj=obj, method=method)
        else:
            self.logger.get_log().info("(api: {}) (case: {}) ignore jit_save test...".format(self.func, self.case_name))
            self.logger.get_log().info("nice!! test case: {} method: {} test complete~~".format(self.case_name, method))
            return

        if self.func not in self.ignore_api_jit_load and self.case_name not in self.ignore_case_jit_load:
            self.logger.get_log().info(
                "start jit load ==========> case: {} test_method: {}".format(self.case_name, method)
            )
            load_res = self.jit_load(method=method)
            # self.logger.get_log().info("load_res is: {}".format(load_res))
            self.logger.get_log().info(
                "start acc comparing ==========> case: {} test_method: {} compare: {} ".format(
                    self.case_name, method, "load_res and exp"
                )
            )
            compare(load_res, exp, self.atol, self.rtol)
        else:
            self.logger.get_log().info("(api: {}) (case: {}) ignore jit_load test...".format(self.func, self.case_name))
            self.logger.get_log().info("nice!! test case: {} method: {} test complete~~".format(self.case_name, method))
            return

        # 若是nn.Layer组网且有参数pdiparams的情况，则需要进一步测试推理部署结果
        if self.func_type == "class" and os.path.exists(
            os.path.join(self.jit_save_path, self.get_func("paddle") + ".pdiparams")
        ):
            self.logger.get_log().info(
                "start infer load ==========> case: {} test_method: {}".format(self.case_name, method)
            )
            infer_res = self.infer_load()
            # self.logger.get_log().info("infer_res is: {}".format(infer_res))
            if isinstance(exp, (list, tuple)):
                exp = exp[0]
            self.logger.get_log().info(
                "start acc comparing ==========> case: {} test_method: {} compare: {} ".format(
                    self.case_name, method, "infer_res and exp"
                )
            )
            compare(infer_res, exp, self.atol, self.rtol)
        self.logger.get_log().info("nice!! test case: {} method: {} test complete~~".format(self.case_name, method))


def compare(result, expect, delta=1e-10, rtol=1e-10):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :param rtol: 相对误差
    :return:
    """
    if isinstance(expect, paddle.Tensor) or isinstance(expect, np.ndarray):
        if isinstance(result, paddle.Tensor):
            result = result.numpy()
        if isinstance(expect, paddle.Tensor):
            expect = expect.numpy()
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            diff = abs(result - expect)
            logging.error("expect is: {}".format(expect))
            logging.error("result is: {}".format(result))
            logging.error("Output has diff! max diff: {}".format(np.amax(diff)))
        if result.dtype != expect.dtype:
            logging.error(
                "Different output data types! res type is: {}, and expect type is: {}".format(
                    result.dtype, expect.dtype
                )
            )
        assert res
        assert result.shape == expect.shape
        assert result.dtype == expect.dtype
    elif isinstance(expect, list) or isinstance(expect, tuple):
        for i, element in enumerate(expect):
            if isinstance(result, (np.generic, np.ndarray)) or isinstance(result, paddle.Tensor):
                if i > 0:
                    break
                compare(result, expect[i], delta, rtol)

            else:
                compare(result[i], expect[i], delta, rtol)
    elif isinstance(expect, (bool, int, float)):
        assert expect == result
    else:
        raise Exception("expect is unknown data struction in compare_tool!!!")
