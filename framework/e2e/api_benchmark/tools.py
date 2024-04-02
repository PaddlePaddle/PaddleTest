#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
utils
"""
import os
import shutil


def delete(path):
    """
    删除一个文件/文件夹
    :param path: 待删除的文件路径
    :return:
    """
    if not os.path.exists(path):
        print("[*] {} not exists".format(path))
        return

    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)
    elif os.path.islink(path):
        os.remove(path)
    else:
        print("[*] unknow type for: " + path)
