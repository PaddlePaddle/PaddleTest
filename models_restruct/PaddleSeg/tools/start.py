"""
start before model running
"""
import os
import sys
import json
import shutil
import logging
import platform
import subprocess
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
        self.env_dict = {}
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("###self.qa_yaml_name: {}".format(self.qa_yaml_name))
        logger.info("###self.rd_yaml_path: {}".format(self.rd_yaml_path))
        if "prim" in self.rd_yaml_path:
            logger.info("###prim mode")
            os.environ["FLAGS_prim_all"] = "True"
            self.env_dict["FLAGS_prim_all"] = "True"
        if "static" in self.rd_yaml_path:
            logger.info("###no prim mode")
            os.environ["FLAGS_prim_all"] = "False"
            self.env_dict["FLAGS_prim_all"] = "False"
        logger.info("###self.rd_yaml_path: {}".format(self.rd_yaml_path))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)
        self.model = self.qa_yaml_name.split("^")[-1]
        logger.info("###self.model_name: {}".format(self.model))
        self.env_dict["model"] = self.model
        os.environ["model"] = self.model

    def prepare_env(self):
        """
        环境变量设置
        """
        if "cityscapes" in self.model:
            if not os.path.exists("PaddleSeg/data/seg_dynamic_pretrain/{}/model.pdparams".format(self.model)):
                cmd = (
                    "wget -P PaddleSeg/data/seg_dynamic_pretrain/{} https://bj.bcebos.com/paddleseg/dygraph"
                    "/cityscapes/{}/model.pdparams".format(self.model, self.model)
                )
                if platform.system() == "Windows":
                    subprocess.run(cmd)
                else:
                    subprocess.run(cmd, shell=True)
        if "voc12" in self.model:
            if not os.path.exists("PaddleSeg/data/seg_dynamic_pretrain/{}/model.pdparams".format(self.model)):
                cmd = (
                    "wget -P PaddleSeg/data/seg_dynamic_pretrain/{} https://bj.bcebos.com/paddleseg/dygraph"
                    "/pascal_voc12/{}/model.pdparams".format(self.model, self.model)
                )
                if platform.system() == "Windows":
                    subprocess.run(cmd)
                else:
                    subprocess.run(cmd, shell=True)
        if "ade20k" in self.model:
            if not os.path.exists("PaddleSeg/data/seg_dynamic_pretrain/{}/model.pdparams".format(self.model)):
                cmd = (
                    "wget -P PaddleSeg/data/seg_dynamic_pretrain/{} https://bj.bcebos.com/paddleseg/dygraph"
                    "/ade20k/{}/model.pdparams".format(self.model, self.model)
                )
                if platform.system() == "Windows":
                    subprocess.run(cmd)
                else:
                    subprocess.run(cmd, shell=True)
        if "camvid" in self.model:
            if not os.path.exists("PaddleSeg/data/seg_dynamic_pretrain/{}/model.pdparams".format(self.model)):
                cmd = (
                    "wget -P PaddleSeg/data/seg_dynamic_pretrain/{} https://bj.bcebos.com/paddleseg/dygraph"
                    "/camvid/{}/model.pdparams".format(self.model, self.model)
                )
                if platform.system() == "Windows":
                    subprocess.run(cmd)
                else:
                    subprocess.run(cmd, shell=True)
        if "cpu" in self.system or "mac" in self.system:
            self.env_dict["set_cuda_flag"] = "cpu"  # 根据操作系统判断
        else:
            self.env_dict["set_cuda_flag"] = "gpu"  # 根据操作系统判断
        if "voc12" in self.model:
            os.environ["image"] = "2007_000033.jpg"
            self.env_dict["image"] = "2007_000033.jpg"
        else:
            os.environ["image"] = "leverkusen_000029_000019_leftImg8bit.png"
            self.env_dict["image"] = "leverkusen_000029_000019_leftImg8bit.png"
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
        os.environ[self.reponame] = json.dumps(self.env_dict)
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
