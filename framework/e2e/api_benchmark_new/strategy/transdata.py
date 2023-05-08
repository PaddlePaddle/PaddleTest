#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
transdata 用于数据的格式结构转换
"""

import json


def data_list_to_dict(data_list):
    """
    将list[{...}]转换成dict{'case_name': {...}}
    :param data_list:
    :return:
    """
    data_dict = {}
    for i in data_list:
        data_dict[i["case_name"]] = i
    return data_dict
