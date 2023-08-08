#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file dist_fleet_utils_localfs.py
  * @author liujie44@baidu.com
  * @date 2021-11-15 11:15
  * @brief
  *
  **************************************************************************/
"""
import os
import sys
import time
from paddle.distributed.fleet.utils import LocalFS
from utils import run_priority

os.system("rm -rf LocalFS_*")
client = LocalFS()


@run_priority(level="P0")
def test_ls_dir():
    """test_ls_dir"""
    subdirs, files = client.ls_dir("./")
    print(subdirs, files)
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_mkdirs():
    """test_mkdirs"""
    client.mkdirs("LocalFS_mkdirs")
    time.sleep(1)
    assert os.path.exists("LocalFS_mkdirs") is True
    assert os.path.isdir("LocalFS_mkdirs") is True
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_delete():
    """test_delete"""
    client.mkdirs("LocalFS_delete")
    time.sleep(1)
    client.delete("LocalFS_delete")
    assert os.path.exists("LocalFS_delete") is False
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_touch():
    """test_touch"""
    client.touch("LocalFS_touch")
    time.sleep(1)
    assert os.path.isfile("LocalFS_touch") is True
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_rename():
    """test_rename"""
    client.touch("LocalFS_rename_src")
    time.sleep(5)
    client.rename("LocalFS_rename_src", "LocalFS_rename_dst")
    assert os.path.isfile("LocalFS_rename_dst")
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_is_file():
    """test_is_file"""
    client.touch("LocalFS_is_file")
    time.sleep(1)
    client.is_file("LocalFS_is_file")
    print("{} ... ok".format(sys._getframe().f_code.co_name))


def test_is_dir():
    """test_is_dir"""
    client.mkdirs("LocalFS_is_dir")
    time.sleep(1)
    assert client.is_dir("LocalFS_is_dir") is True
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_is_exist():
    """test_is_exist"""
    client.touch("LocalFS_exist")
    time.sleep(1)
    assert client.is_exist("LocalFS_exist") is True
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_local_mv():
    """test_local_mv"""
    client.touch("LocalFS_mv_src")
    time.sleep(1)
    client.mv("LocalFS_mv_src", "LocalFS_mv_dst")
    assert client.is_exist("LocalFS_mv_dst") is True
    print("{} ... ok".format(sys._getframe().f_code.co_name))


@run_priority(level="P0")
def test_local_list_dirs():
    """test_local_list_dirs"""
    subdirs = client.list_dirs("./")
    print(subdirs)
    print("{} ... ok".format(sys._getframe().f_code.co_name))


if __name__ == "__main__":
    test_ls_dir()
    test_mkdirs()
    test_delete()
    test_touch()
    test_rename()
    test_is_file()
    test_is_dir()
    test_is_exist()
    test_local_mv()
    test_local_list_dirs()
