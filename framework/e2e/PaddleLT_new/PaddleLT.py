#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yml test
"""
import os
import pytest
import allure
import layertest


# @allure.feature
# @allure.story("e2e_Layer")
# @allure.dynamic.title("{case}")
# @pytest.mark.parametrize
def test_module_layer(title, layerfile, testing):
    """pytest case"""
    # last_dir = os.path.basename(all_dir)
    # base_dir = all_dir.replace(last_dir, "")
    # title = yaml.replace(base_dir, "").replace(".yml", ".{}".format(case)).replace("/", ".")
    allure.dynamic.title(title)
    # allure.dynamic.feature(case)
    allure.dynamic.feature("case")

    allure.dynamic.description("Layer 测试")
    single_test = layertest.LayerTest(title=title, layerfile=layerfile, testing=testing)
    single_test._case_run()
