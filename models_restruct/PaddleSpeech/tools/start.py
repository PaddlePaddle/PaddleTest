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
import zipfile
import argparse
import subprocess
import platform
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")


class PaddleSpeech_Start(object):
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
        self.model = self.qa_yaml_name
        self.mount_path = str(os.getenv("mount_path"))
        # self.use_data_cfs = str(args.use_data_cfs)

    def prepare_cli_cmd(self):
        """
        prepare_cli_cmd
        """
        print("start prepare cli cmd!!")
        speech_map_yaml = os.path.join(os.getcwd(), "tools/speech_map.yaml")
        speech_map = yaml.load(open(speech_map_yaml, "rb"), Loader=yaml.Loader)
        self.cli_cmd = speech_map[self.model]
        self.env_dict["cli_cmd"] = self.cli_cmd

    def prepare_config_params(self):
        """
        准备配置参数
        """
        print("start prepare_config_params!")
        self.env_dict["model"] = self.model
        self.env_dict["model_path"] = self.model_path
        self.env_dict["conf_path"] = "conf/default.yaml"
        self.env_dict["train_output_path"] = "exp/default"
        self.env_dict["ckpt_name"] = self.ckpt_name
        self.env_dict["data_path"] = self.data_path

    def download_data(self, value=None):
        """
        download_data
        """

        zip_name = value.split("/")[-1]
        if os.path.exists(zip_name.replace(".tar", "")):
            logger.info("#### already download {}".format(zip_name))
        else:
            logger.info("#### value: {}".format(value.replace(" ", "")))
            try:
                logger.info("#### start download {}".format(zip_name))
                wget.download(value.replace(" ", ""))
                logger.info("#### end download {}".format(zip_name))
                zf = zipfile.ZipFile(zip_name)
                zf.extractall(os.getcwd())
            except:
                logger.info("#### start download failed {} failed".format(value.replace(" ", "")))
        return 0

    def prepare_data(self):
        """
        prepare_pretrained_model
        """
        print("start prepare data for every model!!")
        sysstr = platform.system()
        if sysstr == "Darwin" and platform.machine() == "x86_64":
            if "/var/root/.local/bin" not in os.environ["PATH"]:
                # mac interl: installed in '/var/root/.local/bin' which is not on PATH.
                os.environ["PATH"] += os.pathsep + "/var/root/.local/bin"

        if sysstr == "Linux":
            if "/root/.local/bin" not in os.environ["PATH"]:
                # linux：paddlespeech are installed in '/root/.local/bin' which is not on PATH
                os.environ["PATH"] += os.pathsep + "/root/.local/bin"  # 注意修改你的路径

        speech_map_yaml = os.path.join(os.getcwd(), "tools/speech_map.yaml")
        speech_map = yaml.load(open(speech_map_yaml, "rb"), Loader=yaml.Loader)
        self.data_path = speech_map[self.model]["data_path"]
        self.model_path = speech_map[self.model]["model_path"]
        self.ckpt_name = speech_map[self.model]["ckpt_name"]
        self.category = speech_map[self.model]["category"]
        if os.path.exists(self.reponame):
            path_now = os.getcwd()
            os.chdir(self.reponame)
            os.chdir(self.model_path)
            if self.category == "am":
                # am newTacotron2 speedyspeech
                os.system(
                    'sed -i "s/max_epoch: 200/max_epoch: 1/g;s/batch_size: 64/batch_size: 32/g" ./conf/default.yaml'
                )
                os.system('sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh')
                # fastspeech2
                os.system('sed -i "s/max_epoch: 1000/max_epoch: 1/g" ./conf/default.yaml')
                # transformertts
                os.system(
                    'sed -i "s/max_epoch: 500/max_epoch: 1/g;s/batch_size: 16/batch_size: 4/g"  ./conf/default.yaml'
                )
            elif self.model != "waveflow":
                # voc parallelwavegan
                os.system(
                    'sed -i "s/train_max_steps: 400000/train_max_steps: 10/g; \
                     s/save_interval_steps: 5000/save_interval_steps: 10/g; \
                     s/eval_interval_steps: 1000/eval_interval_steps: 10/g"  ./conf/default.yaml'
                )
                os.system('sed -i "s/python3/python/g;s/ngpu=1/ngpu=2/g" ./local/train.sh')
                # voc MultiBandMelGAN
                os.system(
                    'sed -i "s/train_max_steps: 1000000/train_max_steps: 10/g; \
                     s/save_interval_steps: 5000/save_interval_steps: 10/g; \
                     s/eval_interval_steps: 1000/eval_interval_steps: 10/g"  ./conf/default.yaml'
                )
                # voc StyleMelGAN
                os.system(
                    'sed -i "s/train_max_steps: 1500000/train_max_steps: 10/g; \
                     s/save_interval_steps: 5000/save_interval_steps: 10/g; \
                     s/eval_interval_steps: 1000/eval_interval_steps: 10/g; \
                     s/batch_size: 32/batch_size: 16/g"  ./conf/default.yaml'
                )
                # HiFiGAN
                os.system(
                    'sed -i "s/train_max_steps: 2500000/train_max_steps: 10/g; \
                     s/save_interval_steps: 5000/save_interval_steps: 10/g; \
                     s/eval_interval_steps: 1000/eval_interval_steps: 10/g"  ./conf/default.yaml'
                )
                # waveRNN
                os.system(
                    'sed -i "s/train_max_steps: 400000/train_max_steps: 10/g; \
                     s/save_interval_steps: 5000/save_interval_steps: 10/g; \
                     s/eval_interval_steps: 1000/eval_interval_steps: 10/g; \
                     s/batch_size: 64/batch_size: 32/g"  ./conf/default.yaml'
                )

            # delete exp
            if os.path.exists("exp"):
                shutil.rmtree("exp")

            sysstr = platform.system()
            if sysstr == "Linux":
                if os.path.exists("/ssd2/ce_data/PaddleSpeech_t2s/preprocess_data"):
                    src_path = "/ssd2/ce_data/PaddleSpeech_t2s/preprocess_data"
                else:
                    src_path = "/home/data/cfs/models_ce/PaddleSpeech_t2s/preprocess_data"

            if not os.path.exists("dump") and (self.model != "waveflow"):
                os.symlink(os.path.join(src_path, self.data_path, "dump"), "dump")
            elif not os.path.exists("preprocessed_ljspeech") and (self.model == "waveflow"):
                # waveflow
                os.symlink(os.path.join(src_path, "waveflow/preprocessed_ljspeech"), "preprocessed_ljspeech")
            else:
                pass

            if self.model == "transformer_tts":
                self.download_data(
                    "https://paddlespeech.bj.bcebos.com/Parakeet/released_models/waveflow/\
waveflow_ljspeech_ckpt_0.3.zip"
                )
            else:
                self.download_data(
                    "https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip"
                )
            os.chdir(path_now)

    def gengrate_test_case(self):
        """
        gengrate_test_case
        """
        speech_map_yaml = os.path.join(os.getcwd(), "tools/speech_map.yaml")
        speech_map = yaml.load(open(speech_map_yaml, "rb"), Loader=yaml.Loader)
        self.category = speech_map[self.model]["category"]
        print("self.category:{}".format(self.category))
        if not os.path.exists("cases"):
            os.makedirs("cases")
        case_file = os.path.join("cases", self.qa_yaml_name) + ".yml"
        if not os.path.exists(case_file):
            if self.category == "am":
                with open(case_file, "w") as f:
                    f.writelines(
                        (
                            "case:" + os.linesep,
                            "    linux:" + os.linesep,
                            "        base: ./base/speech_am_base.yaml" + os.linesep,
                        )
                    )
            else:
                with open(case_file, "w") as f:
                    f.writelines(
                        (
                            "case:" + os.linesep,
                            "    linux:" + os.linesep,
                            "        base: ./base/speech_voc_base.yaml" + os.linesep,
                        )
                    )

    def gengrate_test_case_cli(self):
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
                        "        base: ./base/speech_cli_base.yaml" + os.linesep,
                        "    mac:" + os.linesep,
                        "        base: ./base/speech_cli_base.yaml" + os.linesep,
                        "    windows:" + os.linesep,
                        "        base: ./base/speech_cli_base.yaml" + os.linesep,
                        "    windows_cpu:" + os.linesep,
                        "        base: ./base/speech_cli_base.yaml" + os.linesep,
                    )
                )

    def gengrate_test_case_asr(self):
        """
        gengrate_test_case
        """
        print("start prepare data for asr model")
        speech_map_yaml = os.path.join(os.getcwd(), "tools/speech_map.yaml")
        speech_map = yaml.load(open(speech_map_yaml, "rb"), Loader=yaml.Loader)
        self.model_path = speech_map[self.model]["model_path"]
        self.model_yml = speech_map[self.model]["model_yml"]

        self.env_dict["model"] = self.model
        self.env_dict["model_path"] = speech_map[self.model]["model_path"]
        self.env_dict["model_bin"] = speech_map[self.model]["model_bin"]
        self.env_dict["conf_path"] = speech_map[self.model]["model_yml"]

        if os.path.exists(self.reponame):
            path_now = os.getcwd()
            os.chdir(self.reponame)
            os.chdir(self.model_path)
            os.system("sed -i 's/python3/python/g'  `grep -rl python3  ./local/*`")
            # tal_cs
            cmd = "sed -i 's/n_epoch: 100/n_epoch: 1/g' %s" % self.model_yml
            # conformer
            cmd = "sed -i 's/n_epoch: 5/n_epoch: 1/g' %s" % self.model_yml
            os.system(cmd)
            os.chdir(path_now)

        if not os.path.exists("cases"):
            os.makedirs("cases")
        case_file = os.path.join("cases", self.qa_yaml_name) + ".yml"
        if not os.path.exists(case_file):
            with open(case_file, "w") as f:
                f.writelines(
                    (
                        "case:" + os.linesep,
                        "    linux:" + os.linesep,
                        "        base: ./base/speech_asr_base.yaml" + os.linesep,
                    )
                )

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        speech_map_yaml = os.path.join(os.getcwd(), "tools/speech_map.yaml")
        speech_map = yaml.load(open(speech_map_yaml, "rb"), Loader=yaml.Loader)

        if "paddlespeech_cli" not in self.model:
            self.category = speech_map[self.model]["category"]

        if "paddlespeech_cli" in self.model:
            self.prepare_cli_cmd()
            self.gengrate_test_case_cli()

        elif self.category == "asr":
            logger.info("start gengrate_asr_test_case")
            self.gengrate_test_case_asr()
        else:
            logger.info("start prepare_data")
            self.prepare_data()
            logger.info("start prepare_config_params")
            self.prepare_config_params()
            logger.info("start gengrate_test_case")
            self.gengrate_test_case()
        os.environ[self.reponame] = json.dumps(self.env_dict)
        for k, v in self.env_dict.items():
            os.environ[k] = v


def run():
    """
    执行入口
    """
    model = PaddleSpeech_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    run()
