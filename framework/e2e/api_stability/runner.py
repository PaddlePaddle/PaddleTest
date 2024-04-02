#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import paddle
import sys

sys.path.append("..")
from utils.yaml_loader import YamlLoader
from utils.logger import Logger
from utils.weaktrans import WeakTrans, Framework
from core import Core
from stability import check_all_arrays_equal

log = Logger("stability", "channel")
logger = log.get_log()
# yaml_file = "../yaml/op_correctness_and_stability_phase_one.yml"
yaml_file = "../yaml/op_correctness_phase_one.yml"
yaml_loader = YamlLoader(yaml_file)
cases_name = yaml_loader.get_all_case_name()

error_list = []

for case_name in cases_name:
    wk = WeakTrans(yaml_loader.get_case_info(case_name), logger=log)
    api_name = wk.get_func(Framework.PADDLE)
    c = Core(api_name, dtype="float32")
    c.set_paddle_param(wk.get_inputs(Framework.PADDLE), wk.get_params(Framework.PADDLE))
    forward, grad = c.paddle_run()
    if check_all_arrays_equal(forward):
        logger.info(wk.get_func(Framework.PADDLE) + " 前向值全部相同")
    else:
        # Todo: 报错api记录
        error_list.append(api_name + "前向稳定性测试失败")
    if check_all_arrays_equal(grad):
        logger.info(wk.get_func(Framework.PADDLE) + " 反向值全部相同")
    else:
        # Todo: 报错api记录
        error_list.append(api_name + "反向稳定性测试失败")
if len(error_list) == 0:
    logger.info("测试全部通过")
else:
    logger.info("============ 测试失败，api如下：=============")
    for err in error_list:
        logger.info(err)


