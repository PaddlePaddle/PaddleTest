# encoding: utf-8
"""
执行case前：生成yaml，设置特殊参数，改变监控指标
"""
import os
import logging
import re

logger = logging.getLogger("ce")


class PaddleNLP_Case_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.reponame = os.environ["reponame"]
        self.mode = os.environ["mode"]
        self.case_step = os.environ["case_step"]
        self.case_name = os.environ["case_name"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]

    def build_prepare(self):
        """
        执行准备过程
        """
        if "bert_convergence" in self.qa_yaml_name:
            logger.info("convergence tag is: {}".format(self.case_name.split("train_")[-1]))

            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
            logger.info("export NVIDIA_TF32_OVERRIDE=1")

            if "bert_convergence_daily" in self.qa_yaml_name:
                os.environ["FLAGS_cudnn_deterministic"] = "1"
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


def run():
    """
    执行入口
    """
    platform = os.environ["system"]
    all = re.compile("All").findall(os.environ["AGILE_PIPELINE_NAME"])
    if platform == "linux_convergence" and not all:
        model = PaddleNLP_Case_Start()
        model.build_prepare()
        return 0
    else:
        return 0


if __name__ == "__main__":
    run()
