# encoding: utf-8
"""
执行case后：回收数据避免占用太多存储空间
"""
import os
import pstats
import sys
import json
import glob
import shutil
import argparse
import logging
import yaml
import wget
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("ce")


class PaddleNLP_End(object):
    """
    回收类
    """

    def __init__(self):
        """
        初试化
        """
        self.reponame = os.environ["reponame"]
        self.pipeline_name = os.environ["AGILE_PIPELINE_NAME"]

    def drow_picture(self,model_name, actual_mode, desired_mode,actual_y,desired_y):
        """
        bert_baseline
        bert_prim
        bert_cinn
        bert_prim_cinn
        """
        plt.figure()
        plt.plot([i for i in range(len(desired_y))], desired_y, color = 'g', label=desired_mode)
        plt.plot([i for i in range(len(actual_y))], actual_y, color = 'r', label=actual_mode)
        plt.legend()

        plt.title(model_name + actual_mode + desired_mode)
        plt.savefig('{}'.format(self.pipeline_name))
        plt.close()

        pass
    def analysis_log(self):
        """
        analysis log and save it to picture
        """
        if "bert_convergence" in self.qa_yaml_name:
            logger.info("convergence tag is: {}".format(self.case_name.split("train_")[-1]))

            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
            os.environ["FLAGS_cudnn_deterministic"] = "1"

            logger.info("export NVIDIA_TF32_OVERRIDE=1")
            logger.info("export FLAGS_cudnn_deterministic=1")

            if self.case_name.split("train_")[-1] == "dy2st_cinn":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_use_reduce_split_pass"] = "1"
                os.environ["FLAGS_deny_cinn_ops"] = "dropout"
                os.environ["FLAGS_nvrtc_compile_to_cubin"] = "1"
            elif self.case_name.split("train_")[-1] == "dy2st_prim":
                os.environ["FLAGS_prim_all"] = "true"
            elif self.case_name.split("train_")[-1] == "dy2st_prim_cinn":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_use_reduce_split_pass"] = "1"
                os.environ["FLAGS_prim_all"] = "true"
                os.environ["FLAGS_deny_cinn_ops"] = "dropout"
                os.environ["FLAGS_nvrtc_compile_to_cubin"] = "1"

            logger.info("run type is {}".format(self.case_name.split("train_")[-1]))
            logger.info("set FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
            logger.info("set FLAGS_use_reduce_split_pass as {}".format(os.getenv("FLAGS_use_reduce_split_pass")))
            logger.info("set FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))
            logger.info("set FLAGS_deny_cinn_ops as {}".format(os.getenv("FLAGS_deny_cinn_ops")))
            logger.info("set FLAGS_nvrtc_compile_to_cubin as {}".format(os.getenv("FLAGS_nvrtc_compile_to_cubin")))

        elif "gpt_convergence" in self.qa_yaml_name:
            logger.info("convergence tag is: {}".format(self.case_name.split("train_")[-1]))

            if self.case_name.split("train_")[-1] == "dy2st_prim":
                os.environ["FLAGS_prim_all"] = "True"

            elif self.case_name.split("train_")[-1] == "dy2st_baseline":
                os.environ["FLAGS_prim_all"] = "False"

            logger.info("run type is {}".format(self.case_name.split("train_")[-1]))
            logger.info("set FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))

        elif "ernie_convergence" in self.qa_yaml_name:
            logger.info("convergence tag is: {}".format(self.case_name.split("train_")[-1]))

            if self.case_name.split("train_")[-1] == "dy2st_prim":
                os.environ["FLAGS_cudnn_deterministic"] = "1"
                os.environ["FLAGS_prim_all"] = "True"

            elif self.case_name.split("train_")[-1] == "dy2st_baseline":
                os.environ["FLAGS_cudnn_deterministic"] = "1"
                os.environ["FLAGS_prim_all"] = "False"

            logger.info("run type is {}".format(self.case_name.split("train_")[-1]))
            logger.info("export FLAGS_cudnn_deterministic=1")
            logger.info("set FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))

        else:
            return 0

    def build_end(self):
        """
        执行准备过程
        """
        # 进入repo中
        logger.info("build remove_data start")
        ret = 0
        ret = self.remove_data()
        if ret:
            logger.info("build remove_data failed")
            return ret
        logger.info("build remove_data end")
        return ret


def run():
    """
    执行入口
    """
    model = PaddleNLP_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
