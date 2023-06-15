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
        self.mount_path = str(os.getenv("mount_path"))
        self.use_data_cfs = str(args.use_data_cfs)

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
            os.system("apt-get install -y python3-tk")
            os.system("apt install -y tk-dev")

        if os.path.exists("/etc/redhat-release"):
            os.system("yum -y update")
            os.system("yum install -y libsndfile")
            os.system("yum install -y python3-tk")
            os.system("yum install -y tk-devel")

        if platform.machine() == "arm64":
            print("mac M1")
            os.system("conda install -y scikit-learn")
            os.system("conda install -y onnx")

        if os.path.exists(self.reponame):
            path_now = os.getcwd()
            os.chdir(self.reponame)

            # mac from numba.np.ufunc import _internal
            # os.system("python -m pip install -U numpy<1.24.0")
            # os.system("python -m pip install -U setuptools")
            # mac intel install paddlespeech_ctcdecoders
            sysstr = platform.system()
            if sysstr == "Darwin" and platform.machine() == "x86_64":
                # os.system("python -m pip install -U protobuf==3.19.6")
                # mac interl: installed in '/var/root/.local/bin' which is not on PATH.
                os.environ["PATH"] += os.pathsep + "/var/root/.local/bin"

            os.system("python -m pip uninstall -y paddlespeech")
            if sysstr == "Linux":
                # linux：paddlespeech are installed in '/root/.local/bin' which is not on PATH
                os.environ["PATH"] += os.pathsep + "/root/.local/bin"  # 注意修改你的路径

                os.system("python -m pip install numba")
                os.system("python -m pip install jsonlines")
            # M1: cant not add --ignore-installed"
            # windows install add --user
            os.system("python -m pip install -U setuptools")
            os.system("python -m pip install .  -i https://mirror.baidu.com/pypi/simple")
            # mac intel from numba.np.ufunc import _internal
            if sysstr == "Darwin" and platform.machine() == "x86_64":
                os.system("python -m pip install -U numpy<1.24.0")
            # bug: bce-python-sdk==0.8.79
            os.system("python -m pip install  bce-python-sdk==0.8.74")
            os.system("python -m pip install -U protobuf==3.20.0")
            # librosa
            os.system("python -m pip install -U librosa")
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
            wget.download("https://paddlespeech.bj.bcebos.com/PaddleAudio/ch_zh_mix.wav")
            wget.download("https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/test_long_audio_01.wav")
            wget.download("https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav")
            os.system('echo "demo1 85236145389.wav \n demo2 85236145389.wav" > vec.job')

            # asr tiny data
            sysstr = platform.system()
            if sysstr == "Linux":
                os.chdir("dataset")
                if os.path.exists(self.mount_path) and self.use_data_cfs == "True":
                    src_path = os.path.join(self.mount_path, "asr")
                else:
                    if os.path.exists("/ssd2/ce_data/PaddleSpeech_t2s/asr"):
                        src_path = "/ssd2/ce_data/PaddleSpeech_t2s/asr"
                    else:
                        src_path = "/home/data/cfs/models_ce/PaddleSpeech_t2s/asr"

                # asr librispeech
                if os.path.exists("librispeech"):
                    os.chdir("librispeech")
                    if os.path.exists("dev-clean"):
                        os.unlink("dev-clean")
                        os.unlink("test-clean")
                    os.symlink(os.path.join(src_path, "librispeech/dev-clean"), "dev-clean")
                    os.symlink(os.path.join(src_path, "librispeech/test-clean"), "test-clean")
                    os.chdir("..")

                # asr tal_cs
                os.chdir("tal_cs")
                if os.path.exists("TALCS_corpus"):
                    if os.path.islink("TALCS_corpu") is False:
                        shutil.rmtree("TALCS_corpu")
                    else:
                        os.unlink("TALCS_corpu")
                    shutil.rmtree("TALCS_corpus")
                os.symlink(os.path.join(src_path, "TALCS_corpus"), "TALCS_corpus")
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
