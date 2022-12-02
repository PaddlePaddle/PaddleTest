"""
start before model running
"""
import os
import sys
import json
import shutil
import logging
import wget

logger = logging.getLogger("ce")


class PaddleSeg_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        init
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("###self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)
        self.env_dict = {}
        self.model = self.qa_yaml_name.split("^")[-1]
        logger.info("###self.model_name: {}".format(self.model))
        self.env_dict["model"] = self.model
        os.environ["model"] = self.model

    def prepare_env(self):
        """
        环境变量设置
        """
        if "cpu" in self.system or "mac" in self.system:
            self.env_dict["set_cuda_flag"] = "cpu"  # 根据操作系统判断
        else:
            self.env_dict["set_cuda_flag"] = "gpu"  # 根据操作系统判断
        if "voc12" in self.model:
            os.environ["image"] = "2007_000033.jpg"
        else:
            os.environ["image"] = "leverkusen_000029_000019_leftImg8bit.png" 
        return 0

    def build_prepare(self):
        """
        build prepare
        """
        ret = 0
        ret = self.prepare_env()
        if ret:
            logger.info("build prepare_gpu_env failed")
            return ret


def run():
    """
    执行入口
    """
    model = PaddleSeg_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    run()
