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
    """
    从log中解析出想要的kpi
    """
    # logger.info("###log_content: {}".format(log_content))
    # logger.info("###kpi_name: {}".format(kpi_name))
    kpi_value_all = []
    f = open(log_content, encoding="utf-8", errors="ignore")
    for line in f.readlines():
        if kpi_name == "class_ids":
            if "class_ids" in line and ": [" in line:
                # 增加对nan的处理
                line = line.replace("nan", "'nan'")
                if line.count("class_ids") > 1:  # 存在多个标签时
                    kpi_value_all.append(
                        ast.literal_eval(line.replace("[{", "{").replace("}]", "}").strip())[0]["class_ids"]
                    )
                else:
                    kpi_value_all.append(
                        ast.literal_eval(line.replace("[{", "{").replace("}]", "}").strip())["class_ids"]
                    )
            elif "bbox" in line and ": [" in line:
                if line.count("bbox") > 1:
                    kpi_value_all.append(
                        ast.literal_eval(line.replace("[{", "{").replace("}]", "}").strip())[0]["bbox"]
                    )
                else:
                    kpi_value_all.append(ast.literal_eval(line.replace("[{", "{").replace("}]", "}").strip())["bbox"])
            elif "class id(s)" in line and ": [" in line:
                line = line[line.rfind("class id(s): ") : line.rfind(", score(s):")]
                kpi_value_all.append(line[line.rfind("[") :])
            elif "attributes" in line and ": [" in line:
                line = line[line.rfind("'output': ") : line.rfind("}")]
                kpi_value_all.append(line[line.rfind("[") :])
        else:
            if kpi_name + ":" in line:
                regexp = r"%s:(\s*\d+(?:\.\d+)?)" % kpi_name
                # 如果解析不到符合格式到指标，默认值设置为-1
                r = re.findall(regexp, line)
                # kpi_value = float(r[0].strip()) if len(r) > 0 else -1
                # kpi_value_all.append(kpi_value)
                if len(re.findall(regexp, line)) > 0:  # 只要有值就行，部分模型可能因为学习率的问题导致后几轮的loss为nan, 不在保存-1
                    kpi_value_all.append(float(r[0].strip()))
    f.close()

    # logger.info("###kpi_value_all: {}".format(kpi_value_all))
    # if "-1" in kpi_value_all or kpi_value_all == []: #前几轮是正常后面loss出nan的情况暂时不考虑，后续变化能直接感知
    if kpi_value_all == []:
        kpi_value = float(-1)
    else:
        if kpi_name == "class_ids":
            kpi_value = str(kpi_value_all[-1])
        elif kpi_name == "scores":
            kpi_value = str(kpi_value_all[-1])
        else:
            # kpi_value = float(np.average(np.array(kpi_value_all)))
            kpi_value = float(kpi_value_all[-1])  # 使用最终的loss值  230208修改
    # check 逻辑
    # logger.info("###kpi_value: {}".format(kpi_value))
    # print("###kpi_value: {}".format(kpi_value))
    # logger.info("###kpi_value: {}".format(type(kpi_value)))
    return kpi_value


if __name__ == "__main__":

    logger.info("###")
    log_content = "out_type"
    kpi_name = "class_ids"
    paddlelas_imagenet_parse(log_content, kpi_name)
