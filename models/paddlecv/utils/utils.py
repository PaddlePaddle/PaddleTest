# encoding: utf-8
"""
paddlecv utils
"""

import subprocess
import os
import os.path
import shutil
import logging
import yaml
import allure
import paddle

__all__ = ["exit_check_fucntion", "allure_step"]


def exit_check_fucntion(exit_code, output, output_vis, output_json, input_image):
    """
    exit_check_fucntion
    """
    print(output)
    assert exit_code == 0, "model predict failed!   log information:%s" % (output)
    assert "Error" not in output, "model predict failed!   log information:%s" % (output)
    logging.info("train model sucessfuly!")
    allure.attach(output, "output.log", allure.attachment_type.TEXT)
    allure_attach(output_vis)
    allure_attach(output_json)
    # allure_attach(input_image)


def allure_step(cmd):
    """
    allure_step
    """
    with allure.step("运行指令：{}".format(cmd)):
        pass


def allure_attach(filepath):
    """
    allure_attach
    """
    if os.path.exists("models/paddlecv/" + filepath):
        postfix = os.path.splitext(filepath)[-1]
        if postfix == ".png":
            with open("models/paddlecv/" + filepath, mode="rb") as f:
                file_content = f.read()
            allure.attach(file_content, filepath, allure.attachment_type.PNG)
        elif postfix == ".jpeg" or postfix == ".jpg":
            with open("models/paddlecv/" + filepath, mode="rb") as f:
                file_content = f.read()
            allure.attach(file_content, filepath, allure.attachment_type.JPG)
        elif postfix == ".json" or postfix == ".txt":
            with open("models/paddlecv/" + filepath, mode="rb") as f:
                file_content = f.read()
            allure.attach(file_content, filepath, allure.attachment_type.TEXT)
    else:
        pass
