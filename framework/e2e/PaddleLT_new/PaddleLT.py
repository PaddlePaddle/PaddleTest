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
def test_module_layer(title, layerfile, testing, device_place_id):
    """pytest case"""
    # last_dir = os.path.basename(all_dir)
    # base_dir = all_dir.replace(last_dir, "")
    # title = yaml.replace(base_dir, "").replace(".yml", ".{}".format(case)).replace("/", ".")
    allure.dynamic.title(title)
    # allure.dynamic.feature(case)
    allure.dynamic.feature("case")

    flags_str = ""
    # if os.environ.get("FRAMEWORK") == "paddle":
    #     import paddle

    #     flags_str += "paddle_commit=" + paddle.__git_commit__ + ";"

    for key, value in os.environ.items():
        if key.startswith("FLAGS_"):
            flags_str = flags_str + key + "=" + value + ";"

    flags_str = f"paddle_commit={os.environ.get('paddle_commit')};"
    flags_str += f"TESTING={os.environ.get('TESTING')};"
    flags_str += f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')};"
    flags_str += f"FRAMEWORK={os.environ.get('FRAMEWORK')};"
    flags_str += f"USE_PADDLE_MODEL={os.environ.get('USE_PADDLE_MODEL')};"
    flags_str += f"wheel_url={os.environ.get('wheel_url')};"
    flags_str += f"docker_image={os.environ.get('docker_image')};"

    for key, value in os.environ.items():
        if key.startswith("PLT_"):
            flags_str = flags_str + key + "=" + value + ";"

    allure.dynamic.description(flags_str)
    # allure.dynamic.description("Layer Test")
    single_test = layertest.LayerTest(
        title=title, layerfile=layerfile, testing=testing, device_place_id=device_place_id
    )
    single_test._case_run()
