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
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleOCR_Build(Model_Build):
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
                    self.test_model_list.append(line.strip().replace("^", "/"))
                    print("self.test_model_list:{}".format(self.test_model_list))
        elif str(self.models_file) != "None":  # 获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                for line in open(file_name):
                    if ".yaml" or ".yml" in line:
                        self.test_model_list.append(line.strip().replace("^", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" or ".yml" in file_name:
                    self.test_model_list.append(file_name.strip().replace("^", "/"))

    def build_dataset(self):
        """
        make datalink
        """
        if os.path.exists(self.reponame):
            path_now = os.getcwd()
            os.chdir(self.reponame)

            sysstr = platform.system()
            if sysstr == "Linux":
                src_path = "/ssd2/ce_data/PaddleOCR"
            elif sysstr == "Windows":
                src_path = "F:\\ce_data\\PaddleOCR"
            elif sysstr == "Darwin":
                src_path = "/Users/paddle/PaddleTest/ce_data/PaddleOCR"

            if not os.path.exists("train_data"):
                os.symlink(os.path.join(src_path, "train_data"), "train_data")
            if not os.path.exists("pretrain_models"):
                os.symlink(os.path.join(src_path, "pretrain_models"), "pretrain_models")

            # configs/rec/rec_resnet_stn_bilstm_att.yml
            os.system("python -m pip install fasttext")

            for filename in self.test_model_list:
                print("filename:{}".format(filename))
                if "rec" in filename:
                    cmd = "sed -i s!data_lmdb_release/training!data_lmdb_release/validation!g %s" % filename
                    subprocess.getstatusoutput(cmd)
            os.chdir(path_now)
            print("build dataset!")

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleOCR_Build, self).build_env()
        ret = 0
        ret = self.build_dataset()
        if ret:
            logger.info("build env dataset failed")
            return ret
        return ret
