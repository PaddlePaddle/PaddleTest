# encoding: utf-8
"""
自定义环境准备
"""
import os
import re
import time
from platform import platform
import logging
import argparse
import numpy as np
import yaml
from datetime import datetime
from Model_Build import Model_Build
import requests


logger = logging.getLogger("ce")


class PaddleLLM_Build(Model_Build):
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
        paddle_whl = self.paddle_whl
        os.environ["no_proxy"] = "bcebos.com,baidu.com,baidu-int.com,org.cn"
        print("set timeout as:", os.environ["timeout"])
        print("set no_proxy as:", os.environ["no_proxy"])

        os.system("python -m pip install -U setuptools -i https://mirror.baidu.com/pypi/simple")
        os.system("python -m pip install --user -r requirements_nlp.txt -i https://mirror.baidu.com/pypi/simple")
        os.system("python -m pip uninstall protobuf -y")
        os.system("python -m pip install protobuf==3.20.2")

        os.system("python -m pip install \
                 https://paddle-qa.bj.bcebos.com/PaddleSlim/paddleslim-0.0.0.dev0-py3-none-any.whl")
            
        if os.path.exists(self.reponame):
            os.chdir(self.reponame)
            logger.info("### installing develop paddlenlp")
            os.system("python setup.py bdist_wheel")
            cmd_return = os.system("python -m pip install -U dist/p****.whl --force-reinstall")

            logger.info("### installing develop paddlenlp_ops")
            today = datetime.today().strftime("%Y%m%d")
            # today = "20250516"
            import paddle
            cuda_version=float(paddle.version.cuda())
            prop = paddle.device.cuda.get_device_properties()
            sm_version = prop.major * 10 + prop.minor
            paddlenlp_ops_whl = (
                f"paddlenlp_ops-3.0.0b4.post{today}+cuda{cuda_version}sm{sm_version}paddle3b5fe1f-py3-none-any.whl"
            )
            paddlenlp_ops_url = ("https://paddlenlp.bj.bcebos.com/wheels/{}".format(paddlenlp_ops_whl))
            if os.path.exists(paddlenlp_ops_whl):
                logger.info("paddlenlp_ops_whl has been downloaded, skip")
                cmd_ops_return = 0
            else:
                response = requests.head(paddlenlp_ops_url, timeout=10)
                if response.status_code == 200:
                    os.system("wget -q {}".format(paddlenlp_ops_url))
                    cmd_ops_return = os.system("python -m pip install -U {} --force-reinstall".format(paddlenlp_ops_whl))
                else:
                    logger.info("ipipe had biuld paddlenlp_ops, but maybe failed, use history version")
                    # os.chdir("csrc")
                    # cmd_ops_return = os.system("bash tools/build_wheel.sh")
                    cmd_ops_return = 0
                      
            if cmd_return:
                logger.info("repo {} python -m pip install-failed".format("paddlenlp"))
            if cmd_ops_return:
                logger.info("repo {} python -m pip install-failed".format("paddlenlp_ops"))

            logger.info("installing develop ppdiffusers")
            os.system("python -m pip install ppdiffusers==0.14.0 -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html")

        os.chdir(path_now)
        import paddle
        logger.info("paddle final commit: {}".format(paddle.version.commit))
        import paddlenlp
        logger.info("paddlenlp commit: {}".format(paddlenlp.version.commit))
        os.system("python -m pip list")

        return 0

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleLLM_Build, self).build_env()
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
    model = PaddleLLM_Build(args)
    model.build_paddlenlp()
