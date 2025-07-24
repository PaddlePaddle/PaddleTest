#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ServeTest
"""
import os
from abc import ABC, abstractmethod


class ResponseHandler(ABC):
    """
    abc
    """
    def __init__(self, name):
        """
        初始化函数。
        Args:
            name (str): 测试名称。
        Attributes:
            failed_cases (list): 存储失败的测试用例列表。
            result_log_path (str): 存储测试结果日志的路径。
        """
        self.failed_cases = []
        self.result_log_path = os.path.join(".", f"{name}_result.log")

    @abstractmethod
    def save(self, data, save_path):
        """保存结构化数据"""
        pass

    @abstractmethod
    def compare(self, data, baseline_path):
        """与 baseline 数据进行对比"""
        pass

    @abstractmethod
    def run(self):
        """执行整个流程"""
        pass

