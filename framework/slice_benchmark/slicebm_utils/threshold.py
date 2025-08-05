#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
case性能阈值
"""

threshold_map = {
    'Getitem - forward - Scalar - Integer - float16 - paddle' : [-0.25, -0.15, 0.15],
    'Getitem - forward - Scalar - Tuple of Integers - float16 - paddle': [-0.25, -0.15, 0.15],
    'basic_threshold': [-0.2, -0.1, 0.1],
}

def perf_grade(res, threshold):
    """
    评分标准
    :param res: 性能对比结果
    :return:
    """
    grade = ""
    if isinstance(res, str):
        grade = res
    else:
        if res <= threshold[0]:
            grade = "worse"
        elif threshold[0] < res <= threshold[1]:
            grade = "doubt"
        elif threshold[1] < res <= threshold[2]:
            grade = "equal"
        elif res > threshold[2]:
            grade = "better"
    return grade


def perf_compare(baseline, latest, case_name):
    """
    比较函数
    :param latest: 待测值
    :param baseline: 基线值
    :return: 比例值
    """
    if case_name in threshold_map:
        threshold = threshold_map[case_name]
    else:
        threshold = threshold_map['basic_threshold']

    if isinstance(baseline, str) or isinstance(baseline, str):
        res = "error"
        return res
    else:
        if baseline == 0 or latest == 0:
            res = 0
        else:
            if latest > baseline:
                res = (latest - baseline) / baseline * -1
            else:
                res = (baseline - latest) / latest
    grade = perf_grade(res, threshold)
    return res, grade
