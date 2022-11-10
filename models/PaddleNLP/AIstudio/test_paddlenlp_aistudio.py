# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @brief Test PaddleNLP AIstudio
  * case from https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995
  **************************************************************************/
"""

import io
import os
import sys
import subprocess
import logging
import allure
import pytest
import json


def exit_check(exit_code, file_name,project_name):
    """
    check exit_code
    """
    assert exit_code == 0, "ProjectID: %s %s Failed!" % (file_name,project_name)


def save_log(exit_code, output, file_name):
    """
    Save log for disply
    """
    if exit_code == 0:
        log_file = os.getcwd() + "/log/" + os.path.join(file_name + "_success.log")
        with open(log_file, "a") as flog:
            flog.write("%s" % (output))
            allure.attach.file(log_file, '执行日志', allure.attachment_type.TEXT)
    else:
        log_file = os.getcwd() + "/log/" + os.path.join(file_name + "_err.log")
        with open(log_file, "a") as flog:
            flog.write("%s" % (output))
            allure.attach.file(log_file, '执行日志', allure.attachment_type.TEXT)


def download_project_files():
    """
    Auto download PaddleNLP AIstudio Project
    """
    log_path = os.getcwd() + "/log/"
    output = subprocess.getstatusoutput("cd AIstudio_Download && python ./aistudio_client.py")
    with open(log_path + "download.log", "a") as flog:
        flog.write("%s" % (output[1]))


def get_project_list():
    download_project_files()
    project_list = []
    path = os.listdir(os.getcwd() + "/AIstudio_Download/aistudio_projects_files/")
    for file in path:
        file_name = os.path.splitext(file)[0]
        project_list.append(file_name)
    return project_list


@pytest.mark.parametrize('file_name', get_project_list())
def test_aistudio_case(file_name):
    """
    EXEC AIstudio main.ipynb
    """
    work_path = os.getcwd() 
    file_path = work_path + "/AIstudio_Download/aistudio_projects_files/" + file_name
    project_info= json.load(open('project_info.json','r',encoding="utf-8"))
    project_name = project_info[file_name]

    # 模拟aistudio 环境,解决绝对路径问题
    aistudio_path = '/home/aistudio/'
    os.system("cp -r %s %s && cd %s" % (file_path,aistudio_path,aistudio_path))
    if os.path.exists(os.path.join(aistudio_path + "/main.ipynb")):
        exec_name = "main"
    else:
        exec_name = file_name
    if not os.path.exists(os.path.join(aistudio_path + "/data")):
        os.system("cd %s && mkdir data " % (aistudio_path))
    os.system("cd %s && jupyter nbconvert --to python %s.ipynb" % (aistudio_path, exec_name))
    output = subprocess.getstatusoutput("cd %s && ipython %s.py" % (aistudio_path, exec_name))

    os.system("cd {}".format(work_path))

    # TODO: add download failure case
    # origin_project_list = project_info.keys()
    # result_project_list = os.listdir(os.getcwd() + "/aistudio_projects_files/")
    # failure_list = (set(origin_project_list) ^ set(result_project_list))

    with open('results_project_info.txt',"a+",encoding='utf-8') as result:
        result.write("{}\t{}\t{}".format(0, file_name, project_name) + "\r\n")

    allure.dynamic.parent_suite("PaddleNLP AIstudio")
    allure.dynamic.title("{}".format(file_name))
    allure.dynamic.feature(project_name)
    allure.dynamic.description(
    "启动命令: ipython {}.py".format(exec_name))
    allure.attach.file("{}.py".format(exec_name), '执行脚本', allure.attachment_type.TEXT)

    save_log(output[0], output[1], file_name)
    exit_check(output[0], file_name,project_name)
    os.system("rm -rf {}/*".format(aistudio_path))