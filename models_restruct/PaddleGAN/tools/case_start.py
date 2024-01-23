# encoding: utf-8
"""
执行case前：生成yaml，设置特殊参数，改变监控指标
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


class PaddleClas_Case_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.output = "output"
        self.mode = os.environ["mode"]
        self.reponame = os.environ["reponame"]
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        values_str = os.environ.get(self.reponame, "")
        try:
            self.values_dic = json.loads(values_str)
        except:
            self.values_dic = {}

    def prepare_env(self):
        """
        下载预训练模型, 指定路径
        """
        if str(os.getenv("SOT_EXPORT_FLAG")) == "True":
            os.environ["SOT_EXPORT"] = f"Layer_cases/{self.qa_yaml_name}_{self.case_name}_{self.case_step}"
            logger.info("set org SOT_EXPORT as {}".format(os.getenv("SOT_EXPORT")))

        if "dy2st_convergence" in self.qa_yaml_name:
            logger.info("dy2st_convergence tag is: {}".format(self.case_name.split("train_")[-1]))

            os.environ["FLAGS_prim_all"] = "false"
            logger.info("set org FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))
            os.environ["FLAGS_use_cinn"] = "0"
            logger.info("set org FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
            os.environ["FLAGS_CONVERT_GRAPH_TO_PROGRAM"] = "1"
            logger.info("set FLAGS_CONVERT_GRAPH_TO_PROGRAM {}".format(os.getenv("FLAGS_CONVERT_GRAPH_TO_PROGRAM")))
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
            logger.info("set NVIDIA_TF32_OVERRIDE as {}".format(os.getenv("NVIDIA_TF32_OVERRIDE")))
            logger.info("before set FLAGS_cudnn_deterministic as {}".format(os.getenv("FLAGS_cudnn_deterministic")))
            # os.environ["FLAGS_use_reduce_split_pass"] = "1" #230607 chenzhuo remove
            logger.info("set FLAGS_use_reduce_split_pass {}".format(os.getenv("FLAGS_use_reduce_split_pass")))
            os.environ["FLAGS_deny_cinn_ops"] = "conv2d;conv2d_grad"
            logger.info("set FLAGS_deny_cinn_ops {}".format(os.getenv("FLAGS_deny_cinn_ops")))
            os.environ["FLAGS_conv_workspace_size_limit"] = "400"
            logger.info("set FLAGS_conv_workspace_size_limit {}".format(os.getenv("FLAGS_conv_workspace_size_limit")))
            # os.environ["FLAGS_cudnn_exhaustive_search"] = "1" #设置后无法固定随机量
            os.environ["FLAGS_cudnn_deterministic"] = "1"
            logger.info("set FLAGS_cudnn_exhaustive_search as {}".format(os.getenv("FLAGS_cudnn_exhaustive_search")))

            if self.case_name.split("train_")[-1] == "dy2st_cinn":
                os.environ["FLAGS_use_cinn"] = "1"
            elif self.case_name.split("train_")[-1] == "dy2st_cinn_all":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_cudnn_deterministic"] = "False"
            elif self.case_name.split("train_")[-1] == "dy2st_prim":
                os.environ["FLAGS_prim_all"] = "true"
            elif self.case_name.split("train_")[-1] == "dy2st_prim_all":
                os.environ["FLAGS_prim_all"] = "true"
                os.environ["FLAGS_cudnn_deterministic"] = "False"
            elif self.case_name.split("train_")[-1] == "dy2st_prim_cinn":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_prim_all"] = "true"
            elif self.case_name.split("train_")[-1] == "dy2st_prim_cinn_all":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_prim_all"] = "true"
                os.environ["FLAGS_cudnn_deterministic"] = "False"
                os.environ["FLAGS_nvrtc_compile_to_cubin"] = "1"
            elif self.case_name.split("train_")[-1] == "dy2st_all":
                os.environ["FLAGS_cudnn_deterministic"] = "False"

            logger.info("after set FLAGS_cudnn_deterministic as {}".format(os.getenv("FLAGS_cudnn_deterministic")))
            logger.info("run type is {}".format(self.case_name.split("train_")[-1]))
            logger.info("set FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
            logger.info("set FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))
            logger.info("set FLAGS_nvrtc_compile_to_cubin as {}".format(os.getenv("FLAGS_nvrtc_compile_to_cubin")))

        path_now = os.getcwd()  # 切入路径
        os.chdir(self.reponame)
        if self.case_step == "eval":
            self.values_dic["eval_trained_model"] = None  # 赋初始值
            for name in os.listdir(self.output):
                if self.qa_yaml_name.split("^")[-1].split(".yaml")[0] in name:
                    self.values_dic["eval_trained_model"] = os.path.join(
                        self.output, name, "iter_20_checkpoint.pdparams"
                    )
        os.chdir(path_now)  # 切回路径
        return 0

    def build_prepare(self):
        """
        执行准备过程
        """
        ret = self.prepare_env()
        if ret:
            logger.info("build prepare_env failed")
            return ret

        if self.values_dic != {}:
            os.environ[self.reponame] = json.dumps(self.values_dic)
        return ret


def run():
    """
    执行入口
    """
    model = PaddleClas_Case_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    # args = None
    # model = PaddleClas_Case_Start(args)
    # model.build_prepare()
    run()
