# encoding: utf-8
"""
自定义环境准备
"""
import os
import re
from platform import platform
import sys
import logging
import tarfile
import argparse
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleNLP_Build(Model_Build):
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

    def build_paddlenlp(self):
        """
        安装依赖包
        """
        path_now = os.getcwd()
        platform = self.system
        os.environ["no_proxy"] = "bcebos.com,huggingface.co,baidu.com"
        print(os.environ["no_proxy"])

        if platform == "linux" or platform == "linux_convergence":
            os.system("python -m pip install -U setuptools -i https://mirror.baidu.com/pypi/simple")
            os.system("python -m pip install --user -r requirements_nlp.txt -i https://mirror.baidu.com/pypi/simple")
            os.system("python -m pip uninstall paddlepaddle -y")
            os.system(
                "python -m pip install -U {}".format(self.paddle_whl)
            )  # install paddle for lac requirement paddle>=1.6
        else:
            os.system("python -m pip install  --user -r requirements_win.txt -i https://mirror.baidu.com/pypi/simple")
            os.system("python -m pip uninstall paddlepaddle -y")
            os.system(
                "python -m pip install -U {}".format(self.paddle_whl)
            )  # install paddle for lac requirement paddle>=1.6

        if re.compile("elease").findall(self.paddle_whl):
            os.system("python -m pip install -U  paddleslim -i https://mirror.baidu.com/pypi/simple")
        else:
            os.system(
                "python -m pip install \
                 https://paddle-qa.bj.bcebos.com/PaddleSlim/paddleslim-0.0.0.dev0-py3-none-any.whl"
            )

        import nltk

        nltk.download("punkt")
        from visualdl import LogWriter

        if re.compile("37").findall(self.paddle_whl) or re.compile("38").findall(self.paddle_whl):
            os.system("python -m pip install pgl==2.2.4 -i https://mirror.baidu.com/pypi/simple")

        if os.path.exists(self.reponame):
            os.chdir(self.reponame)
            logger.info("installing develop PaddleNLP")
            os.system("python setup.py bdist_wheel")
            cmd_return = os.system(" python -m pip install -U dist/p****.whl")

            if cmd_return:
                logger.info("repo {} python -m pip install-failed".format(self.reponame))

            logger.info("installing develop ppdiffusers")
            os.system("python -m pip install ppdiffusers==0.14.0 -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html")
        os.chdir(path_now)

        os.system("python -m pip list")
        import paddle

        print("paddle-commit:", paddle.version.commit)

        return 0

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleNLP_Build, self).build_env()
        ret = 0
        ret = self.build_paddlenlp()
        if ret:
            logger.info("build env whl failed")
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
    model = PaddleNLP_Build(args)
    model.build_paddlenlp()
