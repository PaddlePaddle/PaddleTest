"""
Test PaddleNLP AIstudio
case from https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995
"""

import io
import os
import sys
import subprocess
import logging


def save_log(exit_code, output, model, log_dir=""):
    """
    Save log for disply
    """
    if exit_code == 0:
        logging.info("%s passed" % (model))
        log_dir = os.getcwd() + "/log/" + os.path.join(model + ".success")
        with open(log_dir, "a") as flog:
            flog.write("%s" % (output))
    else:
        logging.info("%s failed" % (model))
        log_dir = os.getcwd() + "/log/" + os.path.join(model + ".err")
        with open(log_dir, "a") as flog:
            flog.write("%s" % (output))


def download_project_files():
    """
    Auto download PaddleNLP AIstudio Project
    """
    log_path = os.getcwd() + "/log/"
    output = subprocess.getstatusoutput("cd AIstudio_Download && python ./aistudio_client.py")
    with open(log_path + "download.log", "a") as flog:
        flog.write("%s" % (output[1]))
    assert output[0] == 0, "download failed !"


def test_aistudio_case():
    """
    EXEC AIstudio main.ipynb
    """
    download_project_files()
    path = os.listdir(os.getcwd() + "/aistudio_projects_files/")
    for file in path:
        file_name = os.path.splitext(file)[0]
        file_path = os.getcwd() + "/aistudio_projects_files/" + file
        os.system("cd {}".format(file_path))
        if os.path.exists(os.path.join(file_path + "/main.ipynb")):
            exec_name = "main"
        else:
            exec_name = file_name
        if not os.path.exists(os.path.join(file_path + "/data")):
            os.system("cd %s && mkdir data " % (file_path))
        os.system("cd %s && jupyter nbconvert --to python %s.ipynb" % (file_path, exec_name))
        output = subprocess.getstatusoutput("cd %s && ipython %s.py" % (file_path, exec_name))
        excode = output[0]
        message = output[1]
        save_log(excode, message, file_name)


if __name__ == "__main__":
    test_aistudio_case()
