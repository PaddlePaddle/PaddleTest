# encoding: utf-8
"""
执行case前：生成不同模型的配置参数，例如算法、字典等
"""

import os
import sys
import re
import json
import shutil
import logging
import tarfile
import argparse
import platform
import time
import yaml
import wget
import paddle
import numpy as np

logger = logging.getLogger("ce")


class PaddleOCR_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("###self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接
        self.env_dict = {}
        self.model = os.path.splitext(os.path.basename(self.rd_yaml_path))[0]
        self.category = re.search("/(.*?)/", self.rd_yaml_path).group(1)

    def prepare_config_params(self):
        """
        准备配置参数
        """
        print("start prepare_config_params!")
        yaml_absolute_path = os.path.join(self.REPO_PATH, self.rd_yaml_path)
        self.rd_config = yaml.load(open(yaml_absolute_path, "rb"), Loader=yaml.Loader)
        algorithm = self.rd_config["Architecture"]["algorithm"]
        self.env_dict["algorithm"] = algorithm
        # os.environ['algorithm'] = algorithm
        if "character_dict_path" in self.rd_config["Global"].keys():
            rec_dict = self.rd_config["Global"]["character_dict_path"]
            if not rec_dict:
                rec_dict = "ppocr/utils/ic15_dict.txt"
            self.env_dict["rec_dict"] = rec_dict
            with open(yaml_absolute_path) as f:
                lines = f.readlines()
                for line in lines:
                    if "image_shape" in line:
                        # sar: image_shape: [3, 48, 48, 160] # h:48 w:[48,160]
                        # image_shape_list = line.strip("\n").split(":")[-1]
                        image_shape_list = line.strip("\n").split(":")[1].split("#")[0]
                        print(image_shape_list)
                        image_shape_list = image_shape_list.replace(" ", "")
                        image_shape = re.findall(r"\[(.*?)\]", image_shape_list)
                        if not image_shape:
                            image_shape = "2,32,320"
                        else:
                            image_shape = image_shape[0]
                            print("len(image_shape)={}".format(len(image_shape.split(","))))

                            #  image_shape: [100, 32] # W H
                            if algorithm == "NRTR":
                                image_shape = "32,100"
                            if len(image_shape.split(",")) == 2:
                                image_shape = "1," + image_shape

                        print(image_shape)
                        break
                    else:
                        image_shape = "3,32,128"
            self.env_dict["image_shape"] = image_shape
        # kie
        if self.category == "kie":
            if "ser" in self.model:
                self.env_dict["kie_token"] = "kie_token_ser"
            else:
                self.env_dict["kie_token"] = "kie_token_ser_re"
        # use_gpu
        if paddle.is_compiled_with_cuda():
            self.env_dict["use_gpu"] = "True"
        else:
            self.env_dict["use_gpu"] = "False"

        if self.mode == "precision" and "train" in self.step:
            # check kpi value
            # self.env_dict["train_base_loss"] = "1"
            with open("tools/train.json", "r") as f:
                content = json.load(f)
                train_base_loss = content[self.model]
                logger.info("#### train_base_loss: {}".format(train_base_loss))
            self.env_dict["train_base_loss"] = str(train_base_loss)
            self.env_dict["train_threshold"] = "1.0"

        pretrained_yaml_path = os.path.join(os.getcwd(), "tools/ocr_pretrained.yaml")
        pretrained_yaml = yaml.load(open(pretrained_yaml_path, "rb"), Loader=yaml.Loader)
        if self.mode == "precision" and "eval" in self.step:
            if self.model in pretrained_yaml[self.category].keys():
                logger.info("#### pretrained model exist! Get eval_base_acc!")
                # check eval kpi value
                with open("tools/eval.json", "r") as f:
                    content = json.load(f)
                    eval_base_acc = content[self.model]
                    logger.info("#### eval_base_acc: {}".format(eval_base_acc))
                self.env_dict["eval_base_acc"] = str(eval_base_acc)
            else:
                logger.info("#### pretrained model not exist! Do not get eval_base_acc!")

    def prepare_pretrained_model(self):
        """
        prepare_pretrained_model
        """
        path_now = os.getcwd()
        pretrained_yaml_path = os.path.join(os.getcwd(), "tools/ocr_pretrained.yaml")
        pretrained_yaml = yaml.load(open(pretrained_yaml_path, "rb"), Loader=yaml.Loader)
        if self.model in pretrained_yaml[self.category].keys():
            print("{} exist in pretrained_yaml!".format(self.model))
            print(pretrained_yaml[self.category][self.model])
            pretrained_model_link = pretrained_yaml[self.category][self.model]
            os.chdir("PaddleOCR")
            tar_name = pretrained_model_link.split("/")[-1]
            if not os.path.exists(tar_name):
                wget.download(pretrained_model_link)
                tf = tarfile.open(tar_name)
                tf.extractall(os.getcwd())
                os.rename(os.path.splitext(tar_name)[0], self.model)
            os.chdir(path_now)
            self.env_dict["model"] = self.model
        else:
            print("{} not exist in pretrained_yaml!".format(self.model))

    def gengrate_test_case(self):
        """
        gengrate_test_case
        """
        # sleep for linux rec hang
        time.sleep(10)
        print(os.path.join("cases", self.qa_yaml_name))
        pretrained_yaml_path = os.path.join(os.getcwd(), "tools/ocr_pretrained.yaml")
        pretrained_yaml = yaml.load(open(pretrained_yaml_path, "rb"), Loader=yaml.Loader)
        if not os.path.exists("cases"):
            os.makedirs("cases")

        case_file = os.path.join("cases", self.qa_yaml_name) + ".yml"
        ocr_distill = ["rec_distillation", "det_distill", "det_dml", "det_cml"]
        if not os.path.exists(case_file):
            if any(item in self.qa_yaml_name for item in ocr_distill):
                with open((os.path.join("cases", self.qa_yaml_name) + ".yml"), "w") as f:
                    f.writelines(
                        (
                            "case:" + os.linesep,
                            "    linux:" + os.linesep,
                            "        base: ./base/ocr_" + self.category + "_base_distill.yaml" + os.linesep,
                            "        train:" + os.linesep,
                            "          -" + os.linesep,
                            "            name: multi" + os.linesep,
                            "          -" + os.linesep,
                            "            name: amp" + os.linesep,
                            "          -" + os.linesep,
                            "            name: static" + os.linesep,
                            "        export:" + os.linesep,
                            "          -" + os.linesep,
                            "            name: trained" + os.linesep,
                            "        predict:" + os.linesep,
                            "          -" + os.linesep,
                            "            name: trained" + os.linesep,
                            "          -" + os.linesep,
                            "            name: trained_mkldnn" + os.linesep,
                            "          -" + os.linesep,
                            "            name: trained_tensorRT" + os.linesep,
                            "    windows:" + os.linesep,
                            "        base: ./base/ocr_" + self.category + "_base_distill.yaml" + os.linesep,
                            "    windows_cpu:" + os.linesep,
                            "        base: ./base/ocr_" + self.category + "_base_distill.yaml" + os.linesep,
                            "    mac:" + os.linesep,
                            "        base: ./base/ocr_" + self.category + "_base_distill.yaml" + os.linesep,
                        )
                    )
            else:
                with open((os.path.join("cases", self.qa_yaml_name) + ".yml"), "w") as f:
                    if self.model in pretrained_yaml[self.category].keys():
                        f.writelines(
                            (
                                "case:" + os.linesep,
                                "    linux:" + os.linesep,
                                "        base: ./base/ocr_" + self.category + "_base_pretrained.yaml" + os.linesep,
                                "        train:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: multi" + os.linesep,
                                "          -" + os.linesep,
                                "            name: amp" + os.linesep,
                                "          -" + os.linesep,
                                "            name: static" + os.linesep,
                                "        export:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained" + os.linesep,
                                "        predict:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_mkldnn" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_tensorRT" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained_mkldnn" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained_tensorRT" + os.linesep,
                                "    windows:" + os.linesep,
                                "        base: ./base/ocr_" + self.category + "_base_pretrained.yaml" + os.linesep,
                                "        train:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: multi" + os.linesep,
                                "        export:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained" + os.linesep,
                                "        predict:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_mkldnn" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_tensorRT" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained_mkldnn" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained_tensorRT" + os.linesep,
                                "    windows_cpu:" + os.linesep,
                                "        base: ./base/ocr_" + self.category + "_base_pretrained.yaml" + os.linesep,
                                "        train:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: multi" + os.linesep,
                                "        export:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained" + os.linesep,
                                "        predict:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_mkldnn" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained_mkldnn" + os.linesep,
                                "    mac:" + os.linesep,
                                "        base: ./base/ocr_" + self.category + "_base_pretrained.yaml" + os.linesep,
                                "        train:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: multi" + os.linesep,
                                "        export:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained" + os.linesep,
                                "        predict:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained" + os.linesep,
                            )
                        )
                    else:
                        f.writelines(
                            (
                                "case:" + os.linesep,
                                "    linux:" + os.linesep,
                                "        base: ./base/ocr_" + self.category + "_base.yaml" + os.linesep,
                                "        train:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: multi" + os.linesep,
                                "          -" + os.linesep,
                                "            name: amp" + os.linesep,
                                "          -" + os.linesep,
                                "            name: static" + os.linesep,
                                "        export:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "        predict:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_mkldnn" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_tensorRT" + os.linesep,
                                "    windows:" + os.linesep,
                                "        base: ./base/ocr_" + self.category + "_base.yaml" + os.linesep,
                                "        train:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: multi" + os.linesep,
                                "        export:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: pretrained" + os.linesep,
                                "        predict:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_mkldnn" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_tensorRT" + os.linesep,
                                "    windows_cpu:" + os.linesep,
                                "        base: ./base/ocr_" + self.category + "_base.yaml" + os.linesep,
                                "        train:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: multi" + os.linesep,
                                "        export:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "        predict:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained_mkldnn" + os.linesep,
                                "    mac:" + os.linesep,
                                "        base: ./base/ocr_" + self.category + "_base.yaml" + os.linesep,
                                "        train:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: multi" + os.linesep,
                                "        export:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                                "          -" + os.linesep,
                                "        predict:" + os.linesep,
                                "          -" + os.linesep,
                                "            name: trained" + os.linesep,
                            )
                        )

    def prepare_dataset(self):
        """
        prepare_dataset
        """
        path_now = os.getcwd()
        os.chdir("PaddleOCR")
        sysstr = platform.system()
        if sysstr == "Linux":
            if os.path.exists("/ssd2/ce_data/PaddleOCR"):
                src_path = "/ssd2/ce_data/PaddleOCR"
            else:
                src_path = "/home/data/cfs/models_ce/PaddleOCR"

            if not os.path.exists("train_data"):
                print("PaddleOCR train_data link:")
                shutil.rmtree("train_data")
                os.symlink(os.path.join(src_path, "train_data"), "train_data")
                os.system("ll")
                if not os.path.exists("train_data"):
                    os.system("ln -s /home/data/cfs/models_ce/PaddleOCR/train_data train_data")
                    os.system("ll")
            if not os.path.exists("pretrain_models"):
                print("PaddleOCR pretrain_models link:")
                shutil.rmtree("pretrain_models")
                os.symlink(os.path.join(src_path, "pretrain_models"), "pretrain_models")
                os.system("ll")
                if not os.path.exists("pretrain_models"):
                    os.system("ln -s /home/data/cfs/models_ce/PaddleOCR/pretrain_models pretrain_models")
                    os.system("ll")

        os.chdir(path_now)

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        if self.category == "benchmark":
            print("PaddleOCR/benchmark")
        elif "convergence" in self.model:
            print("Convergence Test")
        else:
            ret = 0
            ret = self.prepare_config_params()
            if ret:
                logger.info("build prepare_config_params failed")
            self.prepare_pretrained_model()
            self.gengrate_test_case()
            self.prepare_dataset()
            os.environ[self.reponame] = json.dumps(self.env_dict)
            for k, v in self.env_dict.items():
                os.environ[k] = v
            return ret


def run():
    """
    执行入口
    """
    model = PaddleOCR_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    run()
