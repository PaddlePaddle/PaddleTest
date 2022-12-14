# encoding: utf-8
"""
执行case前: 生成yaml, 设置特殊参数, 改变监控指标
"""
import os
import sys
import json
import shutil
import logging
import tarfile
import argparse
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")


class PaddleNLP_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("### self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接

        self.env_dict = {}
        self.base_yaml_dict = {
            "ImageNet": "ppcls^configs^ImageNet^ResNet^ResNet50.yaml",
            "slim": "ppcls^configs^slim^PPLCNet_x1_0_quantization.yaml",
            "DeepHash": "ppcls^configs^DeepHash^DCH.yaml",
            "GeneralRecognition": "ppcls^configs^GeneralRecognition^GeneralRecognition_PPLCNet_x2_5.yaml",
            "GeneralRecognitionV2": "ppcls^configs^GeneralRecognitionV2^GeneralRecognitionV2_PPLCNetV2_base.yaml",
            "Cartoonface": "ppcls^configs^Cartoonface^ResNet50_icartoon.yaml",
            "Logo": "ppcls^configs^Logo^ResNet50_ReID.yaml",
            "Products": "ppcls^configs^Products^ResNet50_vd_Inshop.yaml",
            "Vehicle": "ppcls^configs^Vehicle^ResNet50.yaml",
            "PULC": "ppcls^configs^PULC^car_exists^PPLCNet_x1_0.yaml",
            "reid": "ppcls^configs^reid^strong_baseline^baseline.yaml",
            "metric_learning": "ppcls^configs^metric_learning^adaface_ir18",
        }
        self.model_type = self.qa_yaml_name.split("^")[2]  # 固定格式为 ppcls^config^model_type
        self.env_dict["clas_model_type"] = self.model_type
        if "^PULC^" in self.qa_yaml_name:
            self.model_type_PULC = self.qa_yaml_name.split("^")[3]  # 固定格式为 ppcls^config^model_type^PULC_type
            self.env_dict["model_type_PULC"] = self.model_type_PULC

    def prepare_data(self, value=None):
        """
        下载模型数据
        """
        # 调用函数路径已切换至PaddleNLP/

        tar_name = value.split("/")[-1]
        # if os.path.exists(tar_name) and os.path.exists(tar_name.replace(".tar", "")):
        # 有end回收数据, 只判断文件夹
        # UIE-X

        # gpt
        # bert
        # transfomer
        # ernie-1.0
        os.char()

        if os.path.exists(tar_name.replace(".tar", "")):
            logger.info("#### already download {}".format(tar_name))
        else:
            logger.info("#### value: {}".format(value.replace(" ", "")))
            try:
                logger.info("#### start download {}".format(tar_name))
                wget.download(value.replace(" ", ""))
                logger.info("#### end download {}".format(tar_name))
                tf = tarfile.open(tar_name)
                tf.extractall(os.getcwd())
            except:
                logger.info("#### start download failed {} failed".format(value.replace(" ", "")))
        return 0

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        ret = 0
        ret = self.prepare_env()
        if ret:
            logger.info("download data failed")
            return ret
        os.environ[self.reponame] = json.dumps(self.env_dict)
        return ret


def run():
    """
    执行入口
    """
    model = PaddleNLP_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    run()
