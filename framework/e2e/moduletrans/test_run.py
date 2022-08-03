#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yml test
"""
import os
import pytest
import allure
from yaml_loader import YamlLoader
import controller


@allure.story("e2e_Layer")
def test_module_layer(yaml, case):
    """pytest case"""
    allure.dynamic.title(case)
    allure.dynamic.description("Layer 测试")
    yml = YamlLoader(yaml)
    case_ = yml.get_case_info(case)
    test_ = controller.ControlModuleTrans(case=case_)
    test_.run_test()
