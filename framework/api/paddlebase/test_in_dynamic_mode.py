#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_in_dynamic_mode

"""

import paddle
import pytest


@pytest.mark.api_base_in_dynamic_mode_parameters
def test_in_dynamic_mode():
    """
    default mode is dynamic, True
    """
    check = paddle.in_dynamic_mode()
    assert check is True


@pytest.mark.api_base_in_dynamic_mode_parameters
def test_in_dynamic_mode1():
    """
    change to static mode, False
    """
    paddle.enable_static()
    check = paddle.in_dynamic_mode()
    assert check is False


@pytest.mark.api_base_in_dynamic_mode_parameters
def test_in_dynamic_mode2():
    """
    change back to dynamic mode, True
    """
    paddle.enable_static()
    paddle.disable_static()
    check = paddle.in_dynamic_mode()
    assert check is True
