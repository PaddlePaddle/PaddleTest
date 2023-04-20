#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
compare
"""

import json


def base_compare(baseline, latest):
    """
    用于api benchmark 的基本对比方法
    :param baseline:
    :param latest:
    :return:
    """
    if isinstance(baseline, str) or isinstance(baseline, str):
        res = "error"
    else:
        if baseline == 0 or latest == 0:
            res = 0
        else:
            if latest > baseline:
                res = (latest / baseline) * -1
            else:
                res = baseline / latest
    return res


def data_dict_compare(baseline_data, latest_data):
    """
    用于api benchmark 的 一次job的所有case性能数 据对比方法
    :param baseline_data: 基线
    :param latest_data: 待测
    :return:
    """
    res = {}
    # baseline_dict = {}
    # latest_dict = {}
    # for i in baseline_data:
    #     baseline_dict[i["case_name"]] = i
    # for i in latest_data:
    #     latest_dict[i["case_name"]] = i

    # print('baseline_data is: ', baseline_data)
    for k, i in baseline_data.items():
        case_name = i.get("case_name")
        if case_name not in latest_data.keys():
            continue
        else:
            j = latest_data.get(case_name)
            # baseline_api = i.get("result").get("api")
            # latest_api = j.get("result").get("api")
            if isinstance(i.get("result"), str):
                baseline_api = json.loads(i.get("result")).get("api")
                forward_base = float(json.loads(i.get("result")).get("forward"))
                backward_base = float(json.loads(i.get("result")).get("backward"))
                total_base = float(json.loads(i.get("result")).get("total"))
            else:
                baseline_api = i.get("result").get("api")
                forward_base = i.get("result").get("forward")
                backward_base = i.get("result").get("backward")
                total_base = i.get("result").get("total")

            if isinstance(j.get("result"), str):
                latest_api = json.loads(j.get("result")).get("api")
                forward_latest = float(json.loads(j.get("result")).get("forward"))
                backward_latest = float(json.loads(j.get("result")).get("backward"))
                total_latest = float(json.loads(j.get("result")).get("total"))
            else:
                latest_api = j.get("result").get("api")
                forward_latest = j.get("result").get("forward")
                backward_latest = j.get("result").get("backward")
                total_latest = j.get("result").get("total")

        forward = base_compare(baseline=forward_base, latest=forward_latest)
        backward = base_compare(baseline=backward_base, latest=backward_latest)
        total = base_compare(baseline=total_base, latest=total_latest)

        res[case_name] = {
            "baseline_api": baseline_api,
            "latest_api": latest_api,
            "forward": forward,
            "backward": backward,
            "total": total,
        }
        # print('res is: ', res)
        # res[case_name] = {
        #     'forward': forward, 'forward_grade': performance_grade(forward),
        #     'backward': backward, 'backward_grade': performance_grade(backward),
        #     'total': total, 'total_grade': performance_grade(total)
        # }
    return res


def data_list_compare(baseline_list, latest_list):
    """
    用于api benchmark 的 一次job的所有case性能数 据对比方法
    :param baseline_data: 基线
    :param latest_data: 待测
    :return:
    """
    res = {}
    baseline_dict = {}
    latest_dict = {}
    for i in baseline_list:
        baseline_dict[i["case_name"]] = i
    for i in latest_list:
        latest_dict[i["case_name"]] = i

    for i in baseline_list:
        case_name = i.get("case_name")
        if case_name not in latest_dict.keys():
            continue
        else:
            j = latest_dict.get(case_name)
            forward_base = float(json.loads(i.get("result")).get("forward"))
            backward_base = float(json.loads(i.get("result")).get("backward"))
            total_base = float(json.loads(i.get("result")).get("total"))
            forward_latest = float(json.loads(j.get("result")).get("forward"))
            backward_latest = float(json.loads(j.get("result")).get("backward"))
            total_latest = float(json.loads(j.get("result")).get("total"))

        forward = base_compare(baseline=forward_base, latest=forward_latest)
        backward = base_compare(baseline=backward_base, latest=backward_latest)
        total = base_compare(baseline=total_base, latest=total_latest)

        res[case_name] = {
            "forward": forward,
            "forward_grade": performance_grade(forward),
            "backward": backward,
            "backward_grade": performance_grade(backward),
            "total": total,
            "total_grade": performance_grade(total),
        }
    return res


def double_check(res):
    """
    获取需要 double check 的 api list
    :param res: data_compare函数输出的结果
    :return:
    """
    case_list = []
    for case_name, grade_dict in res.items():
        if performance_grade(grade_dict["forward"]) == "doubt":
            case_list.append(case_name)
    return case_list


def performance_grade(res):
    """
    评分标准
    :param res: compare_res对比
    :return:
    """
    grade = ""
    if isinstance(res, str):
        grade = res
    else:
        if res <= -1.3:
            grade = "worse"
        elif -1.3 < res <= -1.15:
            grade = "doubt"
        elif -1.15 < res <= 1.15:
            grade = "equal"
        elif res > 1.15:
            grade = "better"
    return grade


def ci_level_reveal(compare_res):
    """
    等级分类
    :param compare_res:
    :return:
    """
    grade_dict = {}
    grade_dict["worse"] = []
    grade_dict["doubt"] = []
    grade_dict["equal"] = []
    grade_dict["better"] = []

    for case_name, compare_dict in compare_res.items():
        tmp = {}
        grade = performance_grade(res=compare_dict["forward"])
        tmp[compare_dict["latest_api"]] = compare_dict["forward"]
        grade_dict[grade].append(tmp)

    return grade_dict
