"""
自定义解析入口
"""
# encoding: utf-8
import re

import numpy as np


def paddlelas_imagenet_parse(log_content, kpi_name):
    """
    从log中解析出想要的kpi
    """
    # print('###log_content', log_content)
    # print('###kpi_name', kpi_name)
    kpi_value_all = []
    f = open(log_content, encoding="utf-8", errors="ignore")
    for line in f.readlines():
        if kpi_name + ":" in line:
            regexp = r"%s:(\s*\d+(?:\.\d+)?)" % kpi_name
            r = re.findall(regexp, line)
            # 如果解析不到符合格式到指标，默认值设置为-1
            kpi_value = float(r[0].strip()) if len(r) > 0 else -1
            kpi_value_all.append(kpi_value)
    f.close()
    # print('###kpi_value_all', kpi_value_all)
    if "-1" in kpi_value_all:
        kpi_value = -1
    else:
        kpi_value = float(np.average(np.array(kpi_value_all)))
    # check 逻辑
    # print('###kpi_value', kpi_value)
    return kpi_value
