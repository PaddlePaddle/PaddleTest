#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
  * @file dist_fetch_envs.py
  * @author lvkunpeng@baidu.com
  * @date 2025-07-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import os
from unittest import mock
from paddle.distributed.launch.context import fetch_envs
from utils import run_priority


@run_priority(level="P0")
def test_fetch_envs():
    """test_fetch_envs"""
    fake_env = {
        'http_proxy': 'http://example.com',
        'https_proxy': 'https://example.com',
        'OTHER_ENV': 'some_value'
    }

    with mock.patch.dict(os.environ, fake_env, clear=True):
        envs = fetch_envs()

        assert 'http_proxy' not in envs
        assert 'https_proxy' not in envs

        assert envs.get('OTHER_ENV') == 'some_value'

        assert 'http_proxy' not in os.environ
        assert 'https_proxy' not in os.environ
        print("test_fetch_envs ... ok")


if __name__ == "__main__":
    test_fetch_envs()