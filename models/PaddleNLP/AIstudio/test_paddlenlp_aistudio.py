"""
Test PaddleNLP AIstudio
case from https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995
"""

import io
import os
import sys
import subprocess
import logging
import allure
import pytest
import json


def exit_check(exit_code, file_name, project_name):
    """
    check exit_code
    """
    assert exit_code == 0, "ProjectID: %s %s Failed!" % (file_name, project_name)


def save_log(exit_code, output, file_name, log_dir=""):
    """
    Save log for disply
    """
    if exit_code == 0:
        log_dir = os.getcwd() + "/log/" + os.path.join(file_name + "_success.log")
        with open(log_dir, "a") as flog:
            flog.write("%s" % (output))
            allure.attach.file(log_dir, "执行日志", allure.attachment_type.TEXT)
    else:
        log_dir = os.getcwd() + "/log/" + os.path.join(file_name + "_err.log")
        with open(log_dir, "a") as flog:
            flog.write("%s" % (output))
            allure.attach.file(log_dir, "执行日志", allure.attachment_type.TEXT)


def download_project_files():
    """
    Auto download PaddleNLP AIstudio Project
    """
    log_path = os.getcwd() + "/log/"
    output = subprocess.getstatusoutput("cd AIstudio_Download && python ./aistudio_client.py")
    with open(log_path + "download.log", "a") as flog:
        flog.write("%s" % (output[1]))
    assert output[0] == 0, "download failed !"


def get_project_list():
    """
    get aistudio_projects_files
    """
    download_project_files()
    project_list = []
    path = os.listdir(os.getcwd() + "/aistudio_projects_files/")
    for file in path:
        file_name = os.path.splitext(file)[0]
        project_list.append(file_name)
    return project_list


@pytest.mark.parametrize("file_name", get_project_list())
def test_aistudio_case(file_name):
    """
    EXEC AIstudio main.ipynb
    """
    file_path = os.getcwd() + "/aistudio_projects_files/" + file_name
    project_name = json.load(open("project_info.json", "r", encoding="utf-8"))[file_name]
    os.system("cd {}".format(file_path))
    if os.path.exists(os.path.join(file_path + "/main.ipynb")):
        exec_name = "main"
    else:
        exec_name = file_name
    if not os.path.exists(os.path.join(file_path + "/data")):
        os.system("cd %s && mkdir data " % (file_path))
    os.system("cd %s && jupyter nbconvert --to python %s.ipynb" % (file_path, exec_name))
    output = subprocess.getstatusoutput("cd %s && ipython %s.py" % (file_path, exec_name))

    allure.dynamic.parent_suite("PaddleNLP AIstudio")
    allure.dynamic.title("{}".format(file_name))
    allure.dynamic.feature(project_name)
    allure.dynamic.description("启动命令: ipython {}.py".format(exec_name))

    save_log(output[0], output[1], file_name)
    exit_check(output[0], file_name, project_name)
