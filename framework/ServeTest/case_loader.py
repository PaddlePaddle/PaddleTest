#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ServeTest
"""
import os
import yaml


def load_yaml_case(path):
    """
    从指定路径加载 YAML 文件，并返回解析后的数据结构。

    Args:
        path (str): YAML 文件的路径。

    Returns:
        dict: 解析后的 YAML 数据结构。

    Raises:
        FileNotFoundError: 如果指定的路径不存在或不是一个文件。
        ValueError: 如果 YAML 文件解析失败或内容不符合预期格式。
        KeyError: 如果 YAML 文件中缺少必要的 'global' 或 'cases' 字段。
        TypeError: 如果 'cases' 字段不是一个列表。

    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[ERROR] YAML 文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"[ERROR] YAML 解析失败: {e}")

    # 检查字段
    if not isinstance(data, dict):
        raise ValueError(f"[ERROR] YAML 顶层应为字典结构，但实际为: {type(data).__name__}")

    if "global" not in data or "cases" not in data:
        raise KeyError("[ERROR] YAML 中必须包含 'global' 和 'cases' 字段")

    if not isinstance(data["cases"], list):
        raise TypeError("[ERROR] 'cases' 字段必须是一个列表")

    return data
