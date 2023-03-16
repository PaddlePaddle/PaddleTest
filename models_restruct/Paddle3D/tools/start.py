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
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")


class Paddle3D_Start(object):
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
        self.env_dict["model"] = self.model
        self.env_dict["category"] = self.category
        # precision
        if self.mode == "precision" and "train" in self.step:
            # check kpi value
            # self.env_dict["train_base_loss"] = "1"
            with open("tools/train.json", "r") as f:
                content = json.load(f)
                train_base_loss = content[self.model]
                logger.info("#### train_base_loss: {}".format(train_base_loss))
            self.env_dict["train_base_loss"] = str(train_base_loss)
            self.env_dict["train_threshold"] = "0.5"

        if self.mode == "precision" and "eval" in self.step:
            # check eval kpi value
            with open("tools/eval.json", "r") as f:
                content = json.load(f)
                eval_base_acc = content[self.model]
                logger.info("#### eval_base_acc: {}".format(eval_base_acc))
            self.env_dict["eval_base_acc"] = str(eval_base_acc)
            # eval_key
            # self.env_dict["eval_key"] = "AP_R11@25%"
            logger.info("start prepare eval_key")
            speech_map_yaml = os.path.join(os.getcwd(), "tools/3d_map.yaml")
            speech_map = yaml.load(open(speech_map_yaml, "rb"), Loader=yaml.Loader)
            eval_key = speech_map[self.model]
            self.env_dict["eval_key"] = eval_key
            logger.info("#### eval_key: {}".format(eval_key))

    def prepare_pretrained_model(self):
        """
        prepare_pretrained_model
        """
        print("start prepare_pretrained_model!")
        if os.path.exists(self.reponame):
            path_now = os.getcwd()
            os.chdir(self.reponame)

            # change batch_size=1
            cmd = "sed -i 's/batch_size: 4/batch_size: 1/g' %s" % self.rd_yaml_path
            os.system(cmd)

            # delete output
            if os.path.exists("output"):
                shutil.rmtree("output")
            # delete /root/.paddle3d/pretrained/deeplabv3_resnet101
            if os.path.exists("/root/.paddle3d"):
                shutil.rmtree("/root/.paddle3d")

            if not os.path.exists(self.model):
                os.makedirs(self.model)
                os.chdir(self.model)
                if self.category == "smoke":
                    print(
                        "https://paddle3d.bj.bcebos.com/models/{}/{}/model.pdparams".format(self.category, self.model)
                    )
                    wget.download(
                        "https://paddle3d.bj.bcebos.com/models/{}/{}/model.pdparams".format(self.category, self.model)
                    )
                elif self.category == "pointpillars":
                    print("https://bj.bcebos.com/paddle3d/models/pointpillar/{}/model.pdparams".format(self.model))
                    wget.download(
                        "https://bj.bcebos.com/paddle3d/models/pointpillar/{}/model.pdparams".format(self.model)
                    )
                elif self.model == "centerpoint_pillars_02voxel_nuscenes_10sweep":
                    print(
                        "https://bj.bcebos.com/paddle3d/models/centerpoint/\
centerpoint_pillars_02voxel_nuscenes_10_sweep/model.pdparams"
                    )
                    wget.download(
                        "https://bj.bcebos.com/paddle3d/models/centerpoint/\
centerpoint_pillars_02voxel_nuscenes_10_sweep/model.pdparams"
                    )
                else:
                    print(
                        "https://bj.bcebos.com/paddle3d/models/{}/{}/model.pdparams".format(self.category, self.model)
                    )
                    wget.download(
                        "https://bj.bcebos.com/paddle3d/models/{}/{}/model.pdparams".format(self.category, self.model)
                    )

            os.chdir(path_now)

    def gengrate_test_case(self):
        """
        gengrate_test_case
        """
        if not os.path.exists("cases"):
            os.makedirs("cases")
        case_file = os.path.join("cases", self.qa_yaml_name) + ".yml"
        if not os.path.exists(case_file):
            with open(case_file, "w") as f:
                f.writelines(
                    (
                        "case:" + os.linesep,
                        "    linux:" + os.linesep,
                        "        base: ./base/3d_base_pretrained.yaml" + os.linesep,
                    )
                )

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        ret = 0
        ret = self.prepare_config_params()
        if ret:
            logger.info("build prepare_config_params failed")
        self.prepare_pretrained_model()
        self.gengrate_test_case()
        os.environ[self.reponame] = json.dumps(self.env_dict)
        for k, v in self.env_dict.items():
            os.environ[k] = v
        return ret


def run():
    """
    执行入口
    """
    model = Paddle3D_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    run()
