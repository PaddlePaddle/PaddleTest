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
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        print("###self.qa_yaml_name", self.qa_yaml_name)
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接

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
            self.model_type_PULC = self.qa_yaml_name.split("-")[3]  # 固定格式为 ppcls-config-model_type-PULC_type
            self.env_dict["model_type_PULC"] = self.model_type_PULC

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
            print("####value", value.replace(" ", ""))
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
        # os.chdir(self.reponame)
        os.chdir("deploy")
        if os.path.exists("models") is False:
            os.mkdir("models")
        os.chdir("models")
        if os.path.exists(value) and os.path.exists(value.replace(".tar", "")):
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

    def get_params(self):
        """
        获取模型输出路径
        """
        self.eval_trained_params = None  # 初始化
        self.eval_pretrained_params = None
        self.predict_pretrain_params = None

        # 获取训好模型的名称
        with open(os.path.join(self.REPO_PATH, self.rd_yaml_path), "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        self.eval_trained_params = content["Arch"]["name"]

        # 获取下载模型的名称
        if self.eval_trained_params == "RecModel":
            if "Backbone" in str(content):
                self.eval_pretrained_params = content["Arch"]["Backbone"]["name"]
            else:
                self.eval_pretrained_params = content["Arch"]["name"]
        elif self.eval_trained_params == "DistillationModel":
            if "Backbone" in str(content):
                self.eval_pretrained_params = content["Arch"]["Backbone"]["name"]
            else:
                if isinstance(content["Arch"]["models"], list):
                    for i in range(len(content["Arch"]["models"])):
                        if "Student" in content["Arch"]["models"][i].keys():
                            self.eval_pretrained_params = content["Arch"]["models"][i]["Student"]["name"]
                else:
                    assert "do not matched in {}".format(self.rd_yaml_path)
        else:
            self.eval_pretrained_params = self.eval_trained_params
        self.eval_pretrained_params = self.eval_pretrained_params.replace("_Tanh", "").replace(
            "_last_stage_stride1", ""
        )  # 替换多余str
        if self.eval_pretrained_params == "AttentionModel":
            self.eval_pretrained_params = "ResNet18"  # 处理特殊情况

        # 获取预测的预训练模型
        if self.model_type == "PULC":
            self.predict_pretrain_params = self.model_type_PULC
        else:
            self.predict_pretrain_params = self.eval_pretrained_params

        # 获取kpi 的标签
        try:
            if "ATTRMetric" in content["Metric"]["Eval"][0]:
                self.kpi_value_eval = "label_f1"
            elif "Recallk" in content["Metric"]["Eval"][0]:
                self.kpi_value_eval = "recall1"
            elif "TopkAcc" in content["Metric"]["Eval"][0]:
                self.kpi_value_eval = "loss"
            else:
                self.kpi_value_eval = "loss"
        except:
            print("### can not get kpi_value_eval")
        return 0

    def prepare_env(self):
        """
        下载预训练模型，指定路径
        """
        self.get_params()
        path_now = os.getcwd()
        os.chdir(self.reponame)  # 切入路径

        # 准备评估内容
        self.env_dict["kpi_value_eval"] = self.kpi_value_eval
        self.env_dict["eval_trained_model"] = os.path.join(
            "output", self.qa_yaml_name, self.eval_trained_params, "latest"
        )

        # 准备导出模型
        self.env_dict["export_trained_model"] = os.path.join("inference", self.qa_yaml_name)

        # 准备预测内容
        if (
            self.model_type == "ImageNet"
            or self.model_type == "slim"
            or self.model_type == "PULC"
            or self.model_type == "DeepHash"
        ):
            self.env_dict["predict_trained_model"] = os.path.join("../inference", self.qa_yaml_name)
        elif self.model_type == "GeneralRecognition":  # 暂时用训好的模型 220815
            self.download_infer_tar("picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer")
            self.download_infer_tar("general_PPLCNet_x2_5_lite_v1.0_infer")
        elif self.model_type == "Cartoonface":
            self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer")
            self.download_infer_tar("cartoon_rec_ResNet50_iCartoon_v1.0_infer")
        elif self.model_type == "Logo":
            self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer")
            self.download_infer_tar("logo_rec_ResNet50_Logo3K_v1.0_infer")
        elif self.model_type == "Products":
            self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer")
            self.download_infer_tar("product_ResNet50_vd_aliproduct_v1.0_infer")
        elif self.model_type == "Vehicle":
            self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer")
            self.download_infer_tar("vehicle_cls_ResNet50_CompCars_v1.0_infer")
        else:
            print("####{} not sport predict".format(self.qa_yaml_name))

        # 下载预训练模型
        # TODO 路径需要优化，现在在repo外层
        step = self.step.split("+")  # 各个阶段按+分割
        for step_single in step:
            if (
                "eval" in step_single or "infer" in step_single or "export" in step_single
            ) and "pretrained" in step_single:
                if os.path.exists(self.eval_pretrained_params + "_pretrained.pdparams"):
                    print("### already have {}_pretrained.pdparams".format(self.eval_pretrained_params))
                else:
                    if (
                        "-ESNet" in self.qa_yaml_name
                        or "-HRNet" in self.qa_yaml_name
                        or "-InceptionV3" in self.qa_yaml_name
                        or "-MobileNetV1" in self.qa_yaml_name
                        or "-MobileNetV3" in self.qa_yaml_name
                        or "-PPHGNet" in self.qa_yaml_name
                        or "-PPLCNet" in self.qa_yaml_name
                        or "-PPLCNetV2" in self.qa_yaml_name
                        or "-ResNet" in self.qa_yaml_name
                        or "-GeneralRecognition_PPLCNet" in self.qa_yaml_name
                        or "-SwinTransformer" in self.qa_yaml_name
                        or "-VGG" in self.qa_yaml_name
                    ):
                        print("#### use legendary_models pretrain model")
                        print("#### start download {}".format(self.eval_pretrained_params))
                        value = "https://paddle-imagenet-models-name.bj.bcebos.com/\
                            dygraph/legendary_models/{}_pretrained.pdparams".format(
                            self.eval_pretrained_params
                        )
                        wget.download(value.replace(" ", ""))
                        print("#### end download {}".format(self.eval_pretrained_params))
                    else:
                        print("#### start download {}".format(self.eval_pretrained_params))
                        value = "https://paddle-imagenet-models-name.bj.bcebos.com/\
                            dygraph/{}_pretrained.pdparams".format(
                            self.eval_pretrained_params
                        )
                        wget.download(value.replace(" ", ""))
                        print("#### end download {}".format(self.eval_pretrained_params))
                self.env_dict["eval_pretrained_model"] = self.eval_pretrained_params + "_pretrained"
                # 准备导出模型
                self.env_dict["export_pretrained_model"] = self.predict_pretrain_params + "_infer"

            elif "predict" in step_single and "pretrained" in step_single:
                self.env_dict["predict_pretrained_model"] = os.path.join(
                    "../{}_infer".format(self.predict_pretrain_params)
                )
                if (
                    self.model_type == "ImageNet"
                    or self.model_type == "slim"
                    or self.model_type == "PULC"
                    or self.model_type == "DeepHash"
                ):
                    if os.path.exists(self.predict_pretrain_params + "_infer") and os.path.exists(
                        os.path.join(self.predict_pretrain_params + "_infer", "inference.pdiparams")
                    ):
                        print("#### already download {}".format(self.predict_pretrain_params))
                    elif self.model_type == "PULC":
                        self.download_data(
                            "https://paddleclas.bj.bcebos.com/models\
                            /PULC/{}_infer.tar".format(
                                self.predict_pretrain_params
                            )
                        )
                    else:
                        self.download_data(
                            "https://paddle-imagenet-models-name.bj.bcebos.com/\
                            dygraph/inference/{}_infer.tar".format(
                                self.predict_pretrain_params
                            )
                        )
                elif self.model_type == "GeneralRecognition":  # 暂时用训好的模型 220815
                    self.download_infer_tar("picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer")
                    self.download_infer_tar("general_PPLCNet_x2_5_lite_v1.0_infer")
                elif self.model_type == "Cartoonface":
                    self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer")
                    self.download_infer_tar("cartoon_rec_ResNet50_iCartoon_v1.0_infer")
                elif self.model_type == "Logo":
                    self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer")
                    self.download_infer_tar("logo_rec_ResNet50_Logo3K_v1.0_infer")
                elif self.model_type == "Products":
                    self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer")
                    self.download_infer_tar("product_ResNet50_vd_aliproduct_v1.0_infer")
                elif self.model_type == "Vehicle":
                    self.download_infer_tar("ppyolov2_r50vd_dcn_mainbody_v1.0_infer")
                    self.download_infer_tar("vehicle_cls_ResNet50_CompCars_v1.0_infer")
                else:
                    print("####{} not sport predict".format(self.qa_yaml_name))
            else:
                print("### {} do not download pretrained model".format(self.qa_yaml_name))

        os.chdir(path_now)  # 切回路径
        return 0

    def prepare_creat_yaml(self):
        """
        基于base yaml创造新的yaml
        """
        print("###self.mode", self.mode)
        # 增加 function 和 precision 的选项，只有在precision时才进行复制,function时只用base验证
        # if self.mode == "function":
        #     if os.path.exists(os.path.join("cases", self.qa_yaml_name)) is True:  # cases 是基于最原始的路径的
        #         os.remove(os.path.join("cases", self.qa_yaml_name))  # 删除已有的 precision 用 base
        #     try:
        #         shutil.copy(
        #             os.path.join("base", self.model_type + "_base.yaml"), \
        #                 os.path.join("cases", self.qa_yaml_name) + ".yaml"
        #         )
        #     except IOError as e:
        #         print("Unable to copy file. %s" % e)
        #     except:
        #         print("Unexpected error:", sys.exc_info())
        # else:
        if os.path.exists(os.path.join("cases", self.qa_yaml_name) + ".yaml") is False:  # cases 是基于最原始的路径的
            source_yaml_name = self.base_yaml_dict[self.model_type]
            try:
                shutil.copy(os.path.join("cases", source_yaml_name), os.path.join("cases", self.qa_yaml_name) + ".yaml")
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except:
                print("Unexpected error:", sys.exc_info())
        return 0

    def prepare_gpu_env(self):
        """
        根据操作系统获取用gpu还是cpu
        """
        if "cpu" in self.system or "mac" in self.system:
            self.env_dict["set_cuda_flag"] = "cpu"  # 根据操作系统判断
        else:
            self.env_dict["set_cuda_flag"] = "gpu"  # 根据操作系统判断
        return 0

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        ret = 0
        ret = self.prepare_env()
        if ret:
            print("build prepare_env failed")
            return ret
        ret = self.prepare_gpu_env()
        if ret:
            print("build prepare_gpu_env failed")
            return ret
        ret = self.prepare_creat_yaml()
        if ret:
            print("build prepare_creat_yaml failed")
            return ret

        # print('####eval_trained_model', self.env_dict['eval_trained_model'])
        # print('####eval_pretrained_model', self.env_dict['eval_pretrained_model'])
        # print('####export_trained_model', self.env_dict['export_trained_model'])
        # print('####export_pretrained_model', self.env_dict['export_pretrained_model'])
        # print('####predict_trained_model', self.env_dict['predict_trained_model'])
        # print('####predict_pretrained_model', self.env_dict['predict_pretrained_model'])
        # print('####set_cuda_flag', self.env_dict['set_cuda_flag'])
        # print('####kpi_value_eval', self.env_dict['kpi_value_eval'])
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
