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
        self.reponame = os.getenv("reponame")
        self.mode = os.getenv("mode")
        self.case_step = os.getenv("case_step")
        self.case_name = os.getenv("case_name")
        self.qa_yaml_name = os.getenv("qa_yaml_name")

    def build_prepare(self):
        """
        执行准备过程
        """
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
            os.environ["FLAGS_use_reduce_split_pass"] = "1"
            logger.info("set FLAGS_use_reduce_split_pass {}".format(os.getenv("FLAGS_use_reduce_split_pass")))
            logger.info("set FLAGS_deny_cinn_ops {}".format(os.getenv("FLAGS_deny_cinn_ops")))
            os.environ["FLAGS_conv_workspace_size_limit"] = "400"
            logger.info("set FLAGS_conv_workspace_size_limit {}".format(os.getenv("FLAGS_conv_workspace_size_limit")))
            os.environ["FLAGS_cudnn_exhaustive_search"] = "0"
            # os.environ["FLAGS_cudnn_exhaustive_search"] = "1" #设置后无法固定随机量
            os.environ["FLAGS_cudnn_deterministic"] = "1"
            logger.info("set FLAGS_cudnn_exhaustive_search as {}".format(os.getenv("FLAGS_cudnn_exhaustive_search")))
            logger.info("set FLAGS_cudnn_deterministic as {}".format(os.getenv("FLAGS_cudnn_deterministic")))

            if self.case_name.split("train_")[-1] == "dy2st_cinn":
                os.environ["FLAGS_use_cinn"] = "1"
                if "^SwinTransformer_tiny_patch4" in self.qa_yaml_name:
                    os.environ["FLAGS_deny_cinn_ops"] = "uniform_random"
                    os.environ["FLAGS_enable_cinn_auto_tune"] = "false"
                if "^CAE^" in self.qa_yaml_name:
                    os.environ["FLAGS_deny_cinn_ops"] = "uniform_random"
                if "^ResNet^" in self.qa_yaml_name:
                    os.environ["FLAGS_cinn_use_cuda_vectorize"] = "1"
                    os.environ["FLAGS_enhance_vertical_fusion_with_recompute"] = "1"

            elif self.case_name.split("train_")[-1] == "dy2st_cinn_all":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_cudnn_deterministic"] = "False"
                if "^SwinTransformer_tiny_patch4" in self.qa_yaml_name:
                    os.environ["FLAGS_deny_cinn_ops"] = "uniform_random"
                    os.environ["FLAGS_enable_cinn_auto_tune"] = "false"
                if "^CAE^" in self.qa_yaml_name:
                    os.environ["FLAGS_deny_cinn_ops"] = "uniform_random"
                if "^ResNet^" in self.qa_yaml_name:
                    os.environ["FLAGS_cinn_use_cuda_vectorize"] = "1"
                    os.environ["FLAGS_enhance_vertical_fusion_with_recompute"] = "1"

            elif self.case_name.split("train_")[-1] == "dy2st_prim":
                os.environ["FLAGS_prim_all"] = "true"

            elif self.case_name.split("train_")[-1] == "dy2st_prim_all":
                os.environ["FLAGS_prim_all"] = "true"
                os.environ["FLAGS_cudnn_deterministic"] = "False"

            elif self.case_name.split("train_")[-1] == "dy2st_prim_cinn":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_prim_all"] = "true"
                if "^SwinTransformer_tiny_patch4" in self.qa_yaml_name:
                    os.environ["FLAGS_deny_cinn_ops"] = "uniform_random"
                    os.environ["FLAGS_enable_cinn_auto_tune"] = "false"
                if "^CAE^" in self.qa_yaml_name:
                    os.environ["FLAGS_deny_cinn_ops"] = "uniform_random"
                if "^ResNet^" in self.qa_yaml_name:
                    os.environ["FLAGS_nvrtc_compile_to_cubin"] = "1"
                    os.environ["FLAGS_cinn_use_cuda_vectorize"] = "1"
                    os.environ["FLAGS_enhance_vertical_fusion_with_recompute"] = "1"

            elif self.case_name.split("train_")[-1] == "dy2st_prim_cinn_all":
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["FLAGS_prim_all"] = "true"
                os.environ["FLAGS_cudnn_deterministic"] = "False"
                if "^SwinTransformer_tiny_patch4" in self.qa_yaml_name:
                    os.environ["FLAGS_deny_cinn_ops"] = "uniform_random"
                    os.environ["FLAGS_enable_cinn_auto_tune"] = "false"
                if "^CAE^" in self.qa_yaml_name:
                    os.environ["FLAGS_deny_cinn_ops"] = "uniform_random"
                if "^ResNet^" in self.qa_yaml_name:
                    os.environ["FLAGS_nvrtc_compile_to_cubin"] = "1"
                    os.environ["FLAGS_cinn_use_cuda_vectorize"] = "1"
                    os.environ["FLAGS_enhance_vertical_fusion_with_recompute"] = "1"
            elif self.case_name.split("train_")[-1] == "dy2st_all":
                os.environ["FLAGS_cudnn_deterministic"] = "False"

            logger.info("after set FLAGS_cudnn_deterministic as {}".format(os.getenv("FLAGS_cudnn_deterministic")))
            logger.info("run type is {}".format(self.case_name.split("train_")[-1]))
            logger.info("set FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
            logger.info("set FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))
            logger.info("set FLAGS_nvrtc_compile_to_cubin as {}".format(os.getenv("FLAGS_nvrtc_compile_to_cubin")))
            logger.info("set FLAGS_deny_cinn_ops as {}".format(os.getenv("FLAGS_deny_cinn_ops")))
            logger.info("set FLAGS_cinn_use_cuda_vectorize as {}".format(os.getenv("FLAGS_cinn_use_cuda_vectorize")))
            logger.info(
                "set FLAGS_enhance_vertical_fusion_with_recompute as {}".format(
                    os.getenv("FLAGS_enhance_vertical_fusion_with_recompute")
                )
            )
            logger.info("set FLAGS_enable_cinn_auto_tune as {}".format(os.getenv("FLAGS_enable_cinn_auto_tune")))

        else:
            return 0


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
