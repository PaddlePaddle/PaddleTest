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


def paddleslim_quat_ptq_parse(log_content, kpi_name):
    """
    从log中解析出想要的kpi
    """
    f = open(log_content, encoding="utf-8", errors="ignore")
    for line in f.readlines():
        if "PTQ with mse/mse_channel_wise:" in line:
            try:
                values = line.split("=")[-1]
                values = values.strip()
                values = values.split("/")
                if kpi_name + "/" in line:
                    kpi_value = values[0].strip("%")
                elif "/" + kpi_name + " =" in line:
                    kpi_value = values[1].strip("%")
            except:
                return -1
    f.close()

    return kpi_value


if __name__ == "__main__":
    logger.info("###")
    log_content = "./tets_log.txt"
    kpi_name = "top5"
    res = paddleslim_quat_ptq_parse(log_content, kpi_name)
    print(res)
