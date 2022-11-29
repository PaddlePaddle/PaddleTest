# encoding: utf-8
"""
自定义环境准备
"""

import os
import sys
import logging
import tarfile
import argparse
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleSeg_Build(Model_Build):
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
        self.seg_model_list = []
        if str(self.models_list) != "None":
            for line in self.models_list.split(","):
                if ".yaml" in line:
                    self.seg_model_list.append(line.strip().replace("-", "/"))
        elif str(self.models_file) != "None":  # 获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                for line in open(file_name):
                    if ".yaml" in line:
                        self.seg_model_list.append(line.strip().replace("-", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" in file_name:
                    self.seg_model_list.append(file_name.strip().replace("-", "/"))

    def build_paddleseg(self):
        """
        安装依赖包
        """
        os.system("python -m pip install --upgrade pip --ignore-installed")
        os.system("pip install -r requirements.txt --ignore-installed")
        os.system("pip install zip --ignore-installed")
        path_now = os.getcwd()
        os.chdir(self.reponame)
        path_repo = os.getcwd()
        wget.download("https://paddle-qa.bj.bcebos.com/PaddleSeg/demo.tar")
        os.system("tar xvf demo.tar")
        if os.path.exists("seg_dynamic_pretrain"):
            os.system("rm -rf seg_dynamic_pretrain")
        os.system("ln -s {}/seg_dynamic_pretrain seg_dynamic_pretrain".format("/ssd2/ce_data/PaddleSeg"))
        mkdir data
        os.chdir("data")
        if os.path.exists("cityscapes"):
            os.system("rm -rf cityscapes")
        logger.info("***start download data")
        wget.download("https://paddle-qa.bj.bcebos.com/PaddleSeg/cityscapes.zip")
        os.system("unzip cityscapes.zip")
        wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/voc.zip")
        os.system("unzip voc.zip")
        os.system("mv voc/VOCdevkit .")
        wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/mot.zip")
        os.system("unzip mot.zip")
        wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/visdrone.zip")
        os.system("unzip visdrone.zip")
        wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/mainbody.zip")
        os.system("unzip mainbody.zip")
        wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/voc.zip")
        os.system("unzip voc.zip")
        logger.info("***download data ended")
        #os.chdir(path_repo + "/deploy/cpp")
        #wget.download(
        #    "https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-Centos"
        #    "-Gcc82-Cuda102-Cudnn76-Trt6018-Py38-Compile/latest/paddle_inference.tgz"
        #)
        #os.system("tar xvf paddle_inference.tgz")
        #os.system('sed -i "s|WITH_GPU=OFF|WITH_GPU=ON|g" scripts/build.sh')
        #os.system('sed -i "s|CUDA_LIB=/path/to/cuda/lib|CUDA_LIB=/usr/local/cuda/lib64|g" scripts/build.sh')
        #os.system('sed -i "s|/path/to/paddle_inference|../paddle_inference|g" scripts/build.sh')
        #os.system('sed -i "s|CUDNN_LIB=/path/to/cudnn/lib|CUDNN_LIB=/usr/lib/x86_64-linux-gnu|g" scripts/build.sh')
        #os.system("sh scripts/build.sh")
        #os.chdir(path_now)
        return 0

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleSeg_Build, self).build_env()
        ret = 0
        ret = self.build_paddleseg()
        if ret:
            logger.info("build env failed")
            return ret
        return ret


if __name__ == "__main__":

    def parse_args():
        """
        接收和解析命令传入的参数
            最好尽可能减少输入给一些默认参数就能跑的示例!
        """
        parser = argparse.ArgumentParser("Tool for running CE task")
        parser.add_argument("--models_file", help="模型列表文件", type=str, default=None)
        parser.add_argument("--reponame", help="输入repo", type=str, default=None)
        args = parser.parse_args()
        return args

    args = parse_args()
    print("args:{}".format(args))
    # logger.info('###args {}'.format(args.models_file))
    model = PaddleSeg_Build(args)
    model.build_paddleseg()
