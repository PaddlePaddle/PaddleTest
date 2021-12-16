#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
拦截器工具函数直接复制粘贴
或者通过 sys.path.append 函数作为包导入
"""

import os
import platform
import paddle
import pytest


# 检测CPU / GPU
skip_not_compile_gpu = pytest.mark.skipif(
    paddle.is_compiled_with_cuda() is not True, reason="skip cases because paddle is not compiled with CUDA"
)

# 检测分支
skip_branch_not_develop = pytest.mark.skipif(
    os.getenv("AGILE_COMPILE_BRANCH") != "develop", reason="skip cases because branch!=develop"
)
skip_branch_is_2_2 = pytest.mark.skipif(
    os.getenv("AGILE_COMPILE_BRANCH") == "release/2.2", reason="skip cases because branch==release/2.2"
)

# 检测平台
skip_platform_not_linux = pytest.mark.skipif(
    platform.system() != "Linux", reason="skip cases because system is not Linux"
)
skip_platform_not_windows = pytest.mark.skipif(
    platform.system() != "Windows", reason="skip cases because system is not Windows"
)
skip_platform_not_mac = pytest.mark.skipif(
    platform.system() != "Darwin", reason="skip cases because system is not Darwin"
)
skip_platform_is_linux = pytest.mark.skipif(platform.system() == "Linux", reason="skip cases because system is Linux")
skip_platform_is_windows = pytest.mark.skipif(
    platform.system() == "Windows", reason="skip cases because system is Windows"
)
skip_platform_is_mac = pytest.mark.skipif(platform.system() == "Darwin", reason="skip cases because system is Darwin")


@skip_branch_not_develop
def test_check_dev_branch():
    """
    test
    :return:
    """
    pass


# check platform
@skip_not_compile_gpu
def test_check_platform():
    """
    test
    :return:
    """
    pass
