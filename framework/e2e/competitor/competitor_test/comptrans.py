#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
compatrans
"""

from utils.weaktrans import WeakTrans, Framework
from utils.yaml_loader import YamlLoader
import numpy as np
import paddle
import torch


class CompeTrans(WeakTrans):
    """
    Competitive transform
    """

    def __init__(self, case, k1, k2):
        """initialize"""
        super().__init__(case)
        self.eval_str = None
        self.framework = Framework()
        self.mapping = self.case[self.framework.TORCH].get("mapping", None)
        self.ins = None
        self._run()

    def _run(self):
        """run"""
        self._generate_ins()

    def get_function(self):
        """get function"""
        paddle_api = super(CompeTrans, self).get_func(self.framework.PADDLE)
        torch_api = super(CompeTrans, self).get_func(self.framework.TORCH)
        return eval(paddle_api), eval(torch_api)

    def get_torch_ins(self):
        """get torch ins"""
        ins_mapping = self.mapping.get("ins", None)
        self._check_ins_info(ins_mapping)
        torch_inputs = self._convert_keys(self.ins["inputs"], ins_mapping)
        torch_params = self._convert_keys(self.ins["params"], ins_mapping)
        return {"inputs": torch_inputs, "params": torch_params}

    def get_paddle_ins(self):
        """get paddle ins"""
        return self.ins

    def _generate_ins(self):
        """generate ins"""
        inputs = super(CompeTrans, self).get_inputs(self.framework.PADDLE)
        params = super(CompeTrans, self).get_params(self.framework.PADDLE)
        self.ins = {"inputs": inputs, "params": params}

    def _convert_keys(self, map1, map2):
        """
        convert torch keys to paddle keys
        """
        rslt = dict()
        for k, v in map1.items():
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
        inputs_info = self.case[self.framework.PADDLE]["inputs"].values()
        inputs_info = list(inputs_info)[0]
        dtype = inputs_info.get("dtype", "float")
        if dtype in ["int", "int32", "int64"]:
            return ["int32", "int64"]
        elif dtype in ["float", "float32", "float64"]:
            return ["float32", "float64"]
        elif dtype in ["complex", "complex64", "complex128"]:
            return ["complex64", "complex128"]


if __name__ == "__main__":
    obj = YamlLoader("eg0.yml")
    cases_name = obj.get_all_case_name()
    for case_name in cases_name:
        case = obj.get_case_info(case_name)
        # print(case)

        w = CompeTrans(case, 0, 0)
        api = w.get_function()

        ins0 = w.get_paddle_ins()
        # print(ins0)

        ins = w.get_torch_ins()
        print(ins)
