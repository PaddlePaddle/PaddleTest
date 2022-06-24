#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
benchmark trans extend weaktrans
"""
import torch

from utils.weaktrans import WeakTrans, Framework
import numpy as np


class BenchTrans(WeakTrans):
    """BenchTrans"""

    def __init__(self, case, default_type=np.float32, seed=None):
        super().__init__(case, default_type=np.float32, seed=None)
        self._check_exists_torch()
        self.paddle_inputs = None
        self.paddle_param = None

    def _check_exists_torch(self):
        """
        检查是否存在torch配置信息
        """
        if self.case.get(Framework.TORCH) is None:
            self.check_torch = False
        else:
            self.check_torch = True

    def get_paddle_api(self):
        """get paddle api str"""
        return self.get_func(Framework.PADDLE)

    def get_torch_api(self):
        """get torch api str"""
        if self.check_torch:
            return self.get_func(Framework.TORCH)
        else:
            RuntimeError("No torch yaml settings")

    def get_torch_inputs(self):
        """get torch inputs"""
        if self.check_torch:
            torch_inputs = dict()
            mapping = self.case[Framework.TORCH]["mapping"].get("ins")
            if self.paddle_inputs is None:
                self.paddle_inputs = self.get_inputs(Framework.PADDLE)
            for key, value in mapping.items():
                if self.paddle_inputs.get(key) is not None:
                    torch_inputs[value] = self.paddle_inputs.get(key)
            return torch_inputs
        else:
            RuntimeError("No torch yaml settings")

    def get_paddle_inputs(self):
        """
        获取paddle输入，只初始化一次
        """
        if self.paddle_inputs is None:
            self.paddle_inputs = self.get_inputs(Framework.PADDLE)
        return self.paddle_inputs

    def get_torch_param(self):
        """
        get torch param
        """
        if self.check_torch:
            torch_param = dict()
            mapping = self.case[Framework.TORCH]["mapping"].get("ins")
            if self.paddle_param is None:
                self.paddle_param = self.get_params(Framework.PADDLE)
            for key, value in mapping.items():
                if self.paddle_param.get(key) is not None:
                    torch_param[value] = self.paddle_param.get(key)
            # for excess
            excess = self.case[Framework.TORCH]["mapping"].get("excess")
            if excess is not None:
                for key, value in excess.items():
                    torch_param[key] = value
            return torch_param
        else:
            RuntimeError("No torch yaml settings")

    def get_paddle_param(self):
        """
        获取paddle参数，只初始化一次
        """
        if self.paddle_param is None:
            self.paddle_param = self.get_params(Framework.PADDLE)
        return self.paddle_param
