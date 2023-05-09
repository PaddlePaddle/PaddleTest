# encoding: utf-8
"""
自定义环境准备
"""
import os
import sys
import logging
import tarfile
import platform
import argparse
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleSlim_Build(Model_Build):
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
        self.data_path_endswith = "datasets"

        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame)
        # 所有和yaml相关的变量与此拼接
        self.reponame = args.reponame
        self.models_list = args.models_list
        self.models_file = args.models_file
        self.clas_model_list = []
        if str(self.models_list) != "None":
            for line in self.models_list.split(","):
                if ".yaml" in line:
                    self.clas_model_list.append(line.strip().replace(":", "/"))
        elif str(self.models_file) != "None":  # 获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                for line in open(file_name):
                    if ".yaml" in line:
                        self.clas_model_list.append(line.strip().replace(":", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" in file_name:
                    self.clas_model_list.append(file_name.strip().replace(":", "/"))

        # 数据原存储路径
        self.dataset_org = str(args.dataset_org)
        # 数据软链的路径
        self.dataset_target = str(args.dataset_target)
        # 使用数据服务时，将被用于更新dataset_org参数
        self.mount_path = str(os.getenv("mount_path"))
        # linux 环境通过use_data_cfs控制是否使用数据服务（True：使用）
        self.use_data_cfs = str(args.use_data_cfs)
        # windows、mac、liunx系统下使用数据服务
        if (
            ("Windows" in platform.system() or "Darwin" in platform.system())
            and os.path.exists(self.mount_path)
            or (os.path.exists(self.mount_path) and self.use_data_cfs == "True")
        ):
            logger.info("#### mount_path diy_build is {}".format(self.mount_path))
            # 设置dataset_org为mount_path
            if os.listdir(self.mount_path) != []:
                self.dataset_org = self.mount_path
                os.environ["dataset_org"] = self.mount_path
                self.dataset_target = os.path.join(os.getcwd(), self.reponame, self.data_path_endswith)
                os.environ["dataset_target"] = self.dataset_target

            # 将dataset_org 软链到dataset_targ
            exit_code = os.symlink(self.dataset_org, os.path.join(self.reponame, self.dataset_target))
            if exit_code:
                logger.info("#### link_dataset failed")

    def build_paddleslim(self):
        """
        安装依赖包
        """
        path_now = os.getcwd()
        os.chdir("PaddleSlim")
        logger.info("install required whl")
        os.system("python -m pip uninstall paddleslim -y")
        os.system("python -m pip uninstall paddlepaddle -y")
        os.system("python -m pip uninstall paddlepaddle-gpu -y")
        os.system("python -m pip install --user setuptools")
        os.system("python -m pip install --user pycocotools")
        os.system("python -m pip install --user sklearn==0.0")
        os.system("python -m pip install --user numpy==1.23.1")
        os.system("python -m pip install --user -U paddleclas")
        os.system("python -m pip install --user -U paddleseg==2.5.0")
        os.system("python -m pip install --user -U paddledet")
        # os.system("python -m pip install --user -U paddlenlp")
        os.system("python -m pip install paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html")
        os.system("python -m pip install --user -U x2paddle==1.3.9")
        cmd_return = os.system("python -m pip install {}".format(self.paddle_whl))
        if cmd_return:
            logger.info("repo {} python -m pip install paddle failed".format(self.reponame))
        os.system("python -m pip install -r requirements.txt")
        os.system("python -m pip install faiss")
        os.system("python -m pip install wheel")
        os.system("python -m pip install twine")
        os.system("python setup.py bdist_wheel")
        os.system("python -m pip uninstall paddleslim -y")
        cmd_return = os.system("python -m pip install -U dist/paddleslim*.whl")
        if cmd_return:
            logger.info("repo {} python -m pip install paddleslim failed with dist".format(self.reponame))
            cmd_setup = os.system("python setup.py install")
            if cmd_setup:
                logger.info("repo {} python -m pip install paddleslim failed with setup".format(self.reponame))
        os.chdir(path_now)
        return 0

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleSlim_Build, self).build_env()
        ret = 0
        ret = self.build_paddleslim()
        if ret:
            logger.info("build env whl failed")
            return ret
        return ret


if __name__ == "__main__":

    def parse_args():
        """
        接收和解析命令传入的参数最好尽可能减少输入给一些默认参数就能跑的示例!
        """
        parser = argparse.ArgumentParser("Tool for running CE task")
        parser.add_argument("--models_file", help="模型列表文件", type=str, default=None)
        parser.add_argument("--reponame", help="输入repo", type=str, default=None)
        args = parser.parse_args()
        return args

    args = parse_args()
    model = PaddleSlim_Build(args)
    model.build_paddleslim()
