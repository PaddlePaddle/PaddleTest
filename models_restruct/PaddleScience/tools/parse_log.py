"""
自定义解析入口
"""
# encoding: utf-8
import re
import ast
import json
import logging
import numpy as np

logger = logging.getLogger("ce")


def paddlelas_imagenet_parse(log_content, kpi_name):
    # 定义一个空列表来存储提取到的KPI值
    kpi_value_all = []

    # 打开日志文件
    with open(log_content, 'r') as f:
        # 逐行读取日志内容
        for line in f.readlines():
            # 如果需要提取class_ids指标
            if kpi_name == "class_ids":
                if "class_ids:" in line:
                    class_ids_str = line.split(":")[1]
                    class_ids = list(map(int, class_ids_str.strip().split()))
                    kpi_value_all.append(class_ids)
            else:
                # 使用正则表达式匹配KPI名称和对应的数值，将提取到的数值添加到kpi_value_all中
                if kpi_name + ":" in line and "[Train]" in line:
                    regexp = r"%s:\s*([0-9.]+)" % re.escape(kpi_name)  # 修改正则表达式以支持带有"."的KPI名称
                    r = re.findall(regexp, line)
                    if len(r) > 0:
                        kpi_value_all.append(float(r[0].strip()))
    
    # 如果没有提取到任何KPI值，则将最终的KPI值设置为-1
    if len(kpi_value_all) == 0:
        kpi_value = -1
    # 否则，根据不同的KPI名称选择最后一个值作为最终的KPI值
    else:
        if kpi_name == "bbox":
            kpi_value = kpi_value_all[-1]
        elif kpi_name == "class id(s)":
            kpi_value = kpi_value_all[-1]
        elif kpi_name == "attributes":
            kpi_value = kpi_value_all[-1]
        else:
            kpi_value = kpi_value_all[-1]

    # 返回最终的KPI值
    return kpi_value


if __name__ == "__main__":

    logger.info("###")
    log_content = "out_type"
    kpi_name = "class_ids"
    paddlelas_imagenet_parse(log_content, kpi_name)
