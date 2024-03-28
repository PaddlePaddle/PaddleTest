#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
保存工具
"""

import os
import pandas as pd


def xlsx_save(sublayer_dict):
    """
    子图保存到excel
    """
    # data = [
    #     {"sub_layer": key, "time(s)": value}
    #     for key, sublayer_dict in sublayer_dict.items()
    #     for value in sublayer_dict.values()
    # ]
    # # 创建 DataFrame
    # df = pd.DataFrame(data)

    # # 将数据写入 Excel 文件
    # excel_file = "output.xlsx"  # 输出的 Excel 文件名
    # df.to_excel(excel_file, index=False)

    data = []

    # 遍历嵌套字典，提取数据并添加到列表中
    for key, sub_dict in sublayer_dict.items():
        row = {"sub_layer": key}
        for subkey, value in sub_dict.items():
            row[subkey] = value
        data.append(row)

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 将数据写入 Excel 文件
    excel_file = os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx"  # 输出的 Excel 文件名
    df.to_excel(excel_file, index=False)
