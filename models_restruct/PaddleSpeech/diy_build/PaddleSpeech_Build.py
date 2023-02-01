# encoding: utf-8
"""
自定义环境准备
"""
import os
import sys
import logging
import tarfile
import argparse
import subprocess
import platform
import shutil
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleSpeech_Build(Model_Build):
    """
    自定义环境准备
    """

    def __init__(self, args):
        """
        初始化变量
        """
        self.paddle_whl = args.paddle_whl
        self.get_repo = args.get_repo
        self.branch = args.branch
        self.system = args.system
        self.set_cuda = args.set_cuda
        self.dataset_org = args.dataset_org
        self.dataset_target = args.dataset_target

        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame)  # 所有和yaml相关的变量与此拼接
        self.reponame = args.reponame
        self.models_list = args.models_list
        self.models_file = args.models_file
        self.test_model_list = []
        if str(self.models_list) != "None":
            for line in self.models_list.split(","):
                if ".yaml" or ".yml" in line:
                    self.test_model_list.append(line.strip().replace(":", "/"))
                    print("self.test_model_list:{}".format(self.test_model_list))
        elif str(self.models_file) != "None":  # 获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                for line in open(file_name):
                    if ".yaml" or ".yml" in line:
                        self.test_model_list.append(line.strip().replace(":", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" or ".yml" in file_name:
                    self.test_model_list.append(file_name.strip().replace(":", "/"))

    def build_wheel(self):
        """
        build paddlespeech wheel
        """
        if os.path.exists("/etc/lsb-release"):
            os.system("apt-get update")
            os.system("apt-get install -y libsndfile1")

        if os.path.exists("/etc/redhat-release"):
            os.system("yum update")
            os.system("yum install -y libsndfile")

        if platform.machine() == "arm64":
            print("mac M1")
            os.system("conda install -y scikit-learn")
            os.system("conda install -y onnx")

        if os.path.exists(self.reponame):
            path_now = os.getcwd()
            os.chdir(self.reponame)
            os.system("python -m pip install . --ignore-installed")
            
            # mac intel install paddlespeech_ctcdecoders
            sysstr = platform.system()
            if sysstr == "Darwin" and platform.machine() == "x86_64":
                os.system("python -m pip install -U protobuf==3.19.6")
            os.chdir(path_now)
            print("build paddlespeech wheel!")

    def build_cli_data(self):
        """
        build_cli_data
        """
        if os.path.exists(self.reponame):
            path_now = os.getcwd()
            os.chdir(self.reponame)
            wget.download("https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav")
            wget.download("https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav")
            wget.download("https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav")
            wget.download("https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav")
            wget.download("https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/test_long_audio_01.wav")
            wget.download("https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav")
            os.system('echo "demo1 85236145389.wav \n demo2 85236145389.wav" > vec.job')
            # asr tiny data
            os.chdir("dataset")

            if os.path.exists("librispeech"):
                shutil.rmtree("librispeech")
                os.symlink("/ssd2/ce_data/PaddleSpeech_t2s/preprocess_data/deepspeech/librispeech", "librispeech")
            os.chdir(path_now)

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleSpeech_Build, self).build_env()
        ret = 0
        ret = self.build_wheel()
        self.build_cli_data()
        if ret:
            logger.info("build env dataset failed")
            return ret
        return ret
