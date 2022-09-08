# encoding: utf-8
"""
执行case前：生成yaml，设置特殊参数，改变监控指标
"""
import os
import sys
import json
import shutil
import tarfile
import argparse
import yaml
import wget
import numpy as np

# TODO wget容易卡死，增加超时计时器 https://blog.csdn.net/weixin_42368421/article/details/101354628


class PaddleClas_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        # self.qa_yaml_name = "ppcls-configs-ImageNet-ResNet-ResNet18.yaml"
        self.reponame = os.environ["reponame"]
        self.mode = os.environ["mode"]  # function or precision

        self.env_dict = {}
        self.base_yaml_dict = {
            "ImageNet": "ppcls-configs-ImageNet-ResNet-ResNet50.yaml",
            "slim": "ppcls-configs-slim-PPLCNet_x1_0_quantization.yaml",
            "DeepHash": "ppcls-configs-DeepHash-DCH.yaml",
            "GeneralRecognition": "ppcls-configs-GeneralRecognition-GeneralRecognition_PPLCNet_x2_5.yaml",
            "Cartoonface": "ppcls-configs-Cartoonface-ResNet50_icartoon.yaml",
            "Logo": "ppcls-configs-Logo-ResNet50_ReID.yaml",
            "Products": "ppcls-configs-Products-ResNet50_vd_Inshop.yaml",
            "Vehicle": "ppcls-configs-Vehicle-ResNet50.yaml",
            "PULC": "ppcls-configs-PULC-car_exists-PPLCNet_x1_0.yaml",
            "reid": "ppcls-configs-reid-strong_baseline-baseline.yaml",
            "metric_learning": "ppcls-configs-metric_learning-adaface_ir18",
        }
        self.model_type = self.qa_yaml_name.split("-")[2]  # 固定格式为 ppcls-config-model_type
        self.env_dict["clas_model_type"] = self.model_type
        if "-PULC-" in self.qa_yaml_name:
            self.clas_model_type_PULC = self.qa_yaml_name.split("-")[3]
            self.env_dict["clas_model_type_PULC"] = self.clas_model_type_PULC

    def download_data(self, value=None):
        """
        下载数据集
        """
        # 调用函数路径已切换至PaddleClas

        tar_name = value.split("/")[-1]
        if os.path.exists(tar_name) and os.path.exists(tar_name.replace(".tar", "")):
            print("#### already download {}".format(tar_name))
        else:
            print("#### start download {}".format(tar_name))
            wget.download(value.replace(" ", ""))
            print("#### end download {}".format(tar_name))
            if os.path.exists(tar_name):
                tf = tarfile.open(tar_name)
                tf.extractall(os.getcwd())
            else:
                return 1
        return 0

    def download_infer_tar(self, value=None):
        """
        下载预测需要的预训练模型
        """
        path_now = os.getcwd()
        os.chdir(self.reponame)
        os.chdir("deploy")
        if os.path.exists("models") is False:
            os.mkdir("models")
        os.chdir("models")

        if os.path.exists(value) and os.path.exists(value.replace(".tar")):
            print("####already download {}".format(value))
        else:
            self.download_data(
                "https://paddle-imagenet-models-name.bj.bcebos.com/\
                dygraph/rec/models/inference/{}.tar".format(
                    value
                )
            )
        os.chdir(path_now)
        return 0

    def prepare_eval_env(self):
        """
        准备评估需要的环境变量
        """
        # print('###', os.path.exists(os.path.join(self.reponame, "output", self.qa_yaml_name)))
        if os.path.exists(os.path.join(self.reponame, "output", self.qa_yaml_name)):
            params_dir = os.listdir(os.path.join(self.reponame, "output", self.qa_yaml_name))[0]
            self.env_dict["eval_pretrained_model"] = os.path.join("output", self.qa_yaml_name, params_dir, "latest")
        else:
            self.env_dict["eval_pretrained_model"] = None
        return 0

    def prepare_predict_env(self):
        """
        准备预测需要的环境变量
        """
        # 下载预训练模型

        if (
            self.model_type == "ImageNet"
            or self.model_type == "slim"
            or self.model_type == "PULC"
            or self.model_type == "DeepHash"
        ):
            # 定义变量
            self.env_dict["save_inference_dir"] = os.path.join("inference", self.qa_yaml_name)
            if os.path.exists(os.path.join(self.reponame, "inference", self.qa_yaml_name)):
                self.env_dict["predict_pretrained_model"] = os.path.join("inference", self.qa_yaml_name)
            else:
                self.env_dict["predict_pretrained_model"] = None
        elif self.model_type == "GeneralRecognition":  # 暂时用训好的模型 220815
            self.download_infer_tar("picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar")
            self.download_infer_tar("general_PPLCNet_x2_5_lite_v1.0_infer.tar")
        elif self.model_type == "Cartoonface":
            self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar")
            self.download_infer_tar("cartoon_rec_ResNet50_iCartoon_v1.0_infer.tar")
        elif self.model_type == "Logo":
            self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar")
            self.download_infer_tar("logo_rec_ResNet50_Logo3K_v1.0_infer.tar")
        elif self.model_type == "Products":
            self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar")
            self.download_infer_tar("product_ResNet50_vd_aliproduct_v1.0_infer.tar")
        elif self.model_type == "Vehicle":
            self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar")
            self.download_infer_tar("vehicle_cls_ResNet50_CompCars_v1.0_infer.tar")
        else:
            print("####{} not sport predict".format(self.qa_yaml_name))
        return 0

    def prepare_creat_yaml(self):
        """
        基于base yaml创造新的yaml
        """
        print("###self.mode", self.mode)
        # 增加 function 和 precision 的选项，只有在precision时才进行复制,function时只用base验证
        if self.mode == "function":
            if os.path.exists(os.path.join("cases", self.qa_yaml_name)) is True:  # cases 是基于最原始的路径的
                os.remove(os.path.join("cases", self.qa_yaml_name))  # 删除已有的 precision 用 base
        else:
            if os.path.exists(os.path.join("cases", self.qa_yaml_name)) is False:  # cases 是基于最原始的路径的
                source_yaml_name = self.base_yaml_dict[self.model_type]
                try:
                    shutil.copy(
                        os.path.join("cases", source_yaml_name), os.path.join("cases", self.qa_yaml_name) + ".yaml"
                    )
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())
        return 0

    def prepare_gpu_env(self):
        """
        根据操作系统获取用gpu还是cpu
        """
        self.env_dict["set_cuda_flag"] = "gpu"  # 根据操作系统判断
        return 0

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        ret = 0
        ret = self.prepare_eval_env()
        if ret:
            print("build prepare_eval_env failed")
            return ret
        ret = self.prepare_predict_env()
        if ret:
            print("build prepare_predict_env failed")
            return ret
        ret = self.prepare_gpu_env()
        if ret:
            print("build prepare_gpu_env failed")
            return ret
        ret = self.prepare_creat_yaml()
        if ret:
            print("build prepare_creat_yaml failed")
            return ret

        # print('####eval_pretrained_model', self.env_dict['eval_pretrained_model'])
        # print('####save_inference_dir', self.env_dict['save_inference_dir'])
        # print('####predict_pretrained_model', self.env_dict['predict_pretrained_model'])
        # print('####set_cuda_flag', self.env_dict['set_cuda_flag'])
        # input()
        os.environ[self.reponame] = json.dumps(self.env_dict)
        return ret


def run():
    """
    执行入口
    """
    model = PaddleClas_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    # args = None
    # model = PaddleClas_Start(args)
    # model.build_prepare()
    run()
