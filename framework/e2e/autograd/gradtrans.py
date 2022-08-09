#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
gradtrans
"""
import os
import sys

curPath = os.path.abspath(os.path.dirname("utils"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.weaktrans import WeakTrans, Framework
from utils.yaml_loader import YamlLoader
import numpy as np
import paddle
import jax
from gradtest import JaxTest


class GradTrans(WeakTrans):
    """
    Competitive transform
    """

    def __init__(self, case):
        """initialize"""
        super().__init__(case)
        self.eval_str = None
        self.framework = Framework()
        self.stop = False
        if not self.case.get(self.framework.JAX, None):
            self.logger.get_log().info("no competitor test !!!")
            self.stop = True
            return
        self.mapping = self.case[self.framework.JAX].get("mapping", None)
        self.ins = None
        self._run()

    def _run(self):
        """run"""
        self._generate_ins()

    def get_function(self):
        """get function"""
        paddle_api = super(GradTrans, self).get_func(self.framework.PADDLE)
        jax_api = super(GradTrans, self).get_func(self.framework.JAX)
        return eval(paddle_api), eval(jax_api)

    def get_jax_ins(self):
        """get torch ins"""
        ins_mapping = self.mapping.get("ins", None)
        self._check_ins_info(ins_mapping)
        jax_inputs = self._convert_keys(self.ins["inputs"], ins_mapping)
        jax_params = self._convert_keys(self.ins["params"], ins_mapping)
        return {"inputs": jax_inputs, "params": jax_params}

    def get_paddle_ins(self):
        """get paddle ins"""
        return self.ins

    def get_init_actor(self):
        """get init """
        return self.mapping.get("init")

    def _generate_ins(self):
        """generate ins"""
        inputs = super(GradTrans, self).get_inputs(self.framework.PADDLE)
        params = super(GradTrans, self).get_params(self.framework.PADDLE)
        self.ins = {"inputs": inputs, "params": params}

    def _convert_keys(self, map1, map2):
        """
        convert torch keys to paddle keys
        """
        rslt = dict()
        for k, v in map1.items():
            if not map2.get(k):  # 跳过paddle有的，但是torch没有的参数
                continue
            rslt[map2.get(k)] = v
        return rslt

    def _check_ins_info(self, ins_mapping):
        """
        check ins format
        """
        if ins_mapping:
            paddle_ins = self.get_paddle_ins()
            inputs_keys = list(paddle_ins["inputs"].keys())
            params_keys = list(paddle_ins["params"].keys())
            origin_keys = inputs_keys + params_keys
            mapping_keys = list(ins_mapping.keys())
            # 判断mapping的ins中输入的key(paddle的)是否输入有误
            for k in mapping_keys:
                if k not in origin_keys:
                    self.logger.get_log().error("mapping ins key: {} entered incorrectly!".format(k))
                    assert False
            if len(origin_keys) != len(mapping_keys):
                self.logger.get_log().warning("the number of keys is difference!!!")

        else:
            self.logger.get_log().error("The mapping of inputs and params from paddle to pytorch is not set")
            assert False

    def compare_condation(self):
        """
        get compare condations
        """
        condations = self.mapping.get("condations", None)
        return condations

    def get_dtype(self):
        """
        get types
        """
        if self.case[self.framework.PADDLE].get("inputs"):
            types_info = self.case[self.framework.PADDLE]["inputs"]
        elif self.case[self.framework.PADDLE].get("params"):
            types_info = self.case[self.framework.PADDLE].get("params")
        else:
            self.logger.get_log().error("Either inputs or params must be set !!!")
            assert False

        types_info = list(types_info.values())[0]

        if isinstance(types_info, dict):
            dtype = types_info.get("dtype", "float")
            if dtype in ["int", "int32", "int64"]:
                return ["int32", "int64"]
            elif dtype in ["float", "float32", "float64"]:
                return ["float32", "float64"]
            elif dtype == "float16":
                return ["float16"]
            elif dtype in ["complex", "complex64", "complex128"]:
                return ["complex64", "complex128"]
            elif dtype == "bool":
                return ["bool"]
        else:
            return ["int32", "int64"]


if __name__ == "__main__":
    # obj = YamlLoader("../yaml/test0.yml")
    # cases_name = obj.get_all_case_name()
    # for case_name in cases_name:
    #     case = obj.get_case_info(case_name)
    #     tran = GradTrans(case)
    #     apis = tran.get_function()
    #     print(apis[0].__name__, apis[1].__name__)
    #     paddle_ins = tran.get_paddle_ins()
    #     jax_ins = tran.get_jax_ins()
    #     print(paddle_ins)
    #     print(jax_ins)
    #     init_actor = tran.get_init_actor()
    #     print(init_actor)
    # jt = JaxTest(apis)
    # jt.run(paddle_ins, jax_ins)
    pass
