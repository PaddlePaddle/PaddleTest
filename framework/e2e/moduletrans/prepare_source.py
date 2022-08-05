#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yml test
"""
import os


class PrepareResource(object):
    """prepare resource"""

    def __init__(self, env, repo):
        self.env = env
        self.cur_path = os.getcwd()

        if not os.path.exists(os.path.join(self.cur_path, "ground_truth.tar")):
            os.system(
                "wget https://paddle-qa.bj.bcebos.com/luozeyu01/framework_e2e_LayerTest/{}/ground_truth.tar".format(env)
            )
            os.system("tar -xzf ground_truth.tar")

        if not os.path.exists(os.path.join(self.cur_path, "ppcls")):
            os.system("git clone -b develop https://github.com/PaddlePaddle/PaddleClas.git")
            os.system("cp -r PaddleClas/ppcls .")
            os.system("cd PaddleClas && python -m pip install -r requirements.txt")
            os.system("cd PaddleClas && python setup.py install")

        if not os.path.exists(os.path.join(self.cur_path, "ppdet")):
            os.system("git clone -b develop https://github.com/PaddlePaddle/PaddleDetection.git")
            os.system("cp -r PaddleDetection/ppdet .")
            os.system("cd PaddleDetection && python -m pip install -r requirements.txt")
            os.system("cd PaddleDetection && python setup.py install")


if __name__ == "__main__":
    env = "Mac_CPU"
    PrepareResource(env)
