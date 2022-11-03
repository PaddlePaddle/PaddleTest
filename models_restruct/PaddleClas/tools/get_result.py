# encoding: utf-8
"""
根据之前执行的结果获取kpi值
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


# 为什么要单独预先作，是因为不能每次执行case时有从一个庞大的tar包中找需要的值，应该生成一个中间状态的yaml文件作为存储
class PaddleClas_Collect(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化参数
        """
        self.repo_name = "PaddleClas"
        self.report_linux_cuda102_py37_develop = {
            "P0": "https://xly.bce.baidu.com/ipipe/ipipe-report/report/19861805/report/result.tar",
            "P1": "https://xly.bce.baidu.com/ipipe/ipipe-report/report/19861798/report/result.tar",
            "P2": "https://xly.bce.baidu.com/ipipe/ipipe-report/report/19861792/report/result.tar",
            "P2_1": "https://xly.bce.baidu.com/ipipe/ipipe-report/report/19861781/report/result.tar",
            "P2_2": "https://xly.bce.baidu.com/ipipe/ipipe-report/report/19894352/report/result.tar",
        }
        # self.report_linux_cuda102_py37_develop = {
        #     "P0": "https://xly.bce.baidu.com/ipipe/ipipe-report/report/19619465/report/result.tar"
        # }

        self.report_linux_cuda102_py37_release = {}

        self.base_yaml_dict = {
            "ImageNet": "ppcls-configs-ImageNet-ResNet-ResNet50.yaml",
            "slim": "ppcls-configs-slim-PPLCNet_x1_0_quantization.yaml",
            "DeepHash": "ppcls-configs-DeepHash-DCH.yaml",
            "GeneralRecognition": "ppcls-configs-GeneralRecognition-GeneralRecognition_PPLCNet_x2_5.yaml",
            "Cartoonface": "ppcls-configs-Cartoonface-ResNet50_icartoon.yaml",
            "GeneralRecognitionV2": "ppcls-configs-GeneralRecognitionV2-GeneralRecognitionV2_PPLCNetV2_base.yaml",
            "Logo": "ppcls-configs-Logo-ResNet50_ReID.yaml",
            "Products": "ppcls-configs-Products-ResNet50_vd_Inshop.yaml",
            "Vehicle": "ppcls-configs-Vehicle-ResNet50.yaml",
            "PULC": "ppcls-configs-PULC-car_exists-PPLCNet_x1_0.yaml",
            "reid": "ppcls-configs-reid-strong_baseline-baseline.yaml",
            "metric_learning": "ppcls-configs-metric_learning-adaface_ir18.yaml",
        }

    def download_data(self, priority="P0", value=None):
        """
        下载数据集
        """
        # 调用函数路径已切换至PaddleClas

        tar_name = value.split("/")[-1]
        if os.path.exists(tar_name.replace(".tar", "_" + priority)):
            print("#### already download {}".format(tar_name))
        else:
            print("#### value: {}".format(value.replace(" ", "")))
            # try:  #result不好用wget，总会断
            #     print("#### start download {}".format(tar_name))
            #     wget.download(value.replace(" ", ""))
            #     print("#### end download {}".format(tar_name))
            #     tf = tarfile.open(tar_name)
            #     tf.extractall(os.getcwd())
            # except:
            #     print("#### start download failed {} failed".format(value.replace(" ", "")))
            try:
                print("#### start download {}".format(tar_name))
                cmd = "wget {} --no-proxy && tar xf {}".format(value.replace(" ", ""), tar_name)
                os.system(cmd)
            except:
                print("#### start download failed {} failed".format(value.replace(" ", "")))
            os.rename(tar_name.replace(".tar", ""), tar_name.replace(".tar", "_" + priority))
            os.remove(tar_name)  # 删除下载的tar包防止重名
        return tar_name.replace(".tar", "_" + priority)

    def load_json(self):
        """
        解析report路径下allure json
        """
        files = os.listdir(self.report_path)
        for file in files:
            if file.endswith("result.json"):
                file_path = os.path.join(self.report_path, file)
                with open(file_path, encoding="UTF-8") as f:
                    for line in f.readlines():
                        data = json.loads(line)
                        yield data

    def clean_report(self):
        """
        清理中间保存的数据
        """
        for name in self.report_path_list:
            shutil.rmtree(name)
        os.remove(self.repo_name + ".tar.gz")
        shutil.rmtree(self.repo_name)

    def get_result_yaml(self):
        """
        获取yaml结果
        """
        self.report_path_list = list()
        self.case_info_list = list()
        for (key, value) in self.report_linux_cuda102_py37_develop.items():
            self.report_path = self.download_data(key, value)
            self.report_path_list.append(self.report_path)
            for case_detail in self.load_json():
                labels = case_detail.get("labels")
                for label in labels:
                    if label.get("name") == "case_info":
                        self.case_info_list.extend(json.loads(label.get("value")))

    def pop_yaml_useless(self, data_json):
        """
        去除原来yaml中的无效信息
        """
        if isinstance(data_json, dict):
            for key, val in data_json.items():
                if isinstance(data_json[key], dict):
                    self.pop_yaml_useless(data_json[key])
                elif isinstance(data_json[key], list):
                    # print('###data_json[key]',data_json[key])
                    for name in data_json[key]:
                        key_pop = []
                        for key1, val1 in name.items():
                            if key1 != "result" and key1 != "name":
                                key_pop.append(key1)
                                # print('###key1', key1)
                                # print('###val1', val1)
                        # print('###key11111', key1)
                        # print('###name11111', name)
                        for key1 in key_pop:
                            name.pop(key1)
                        # print('###name2222', name)
                        # input()
        return data_json

    def update_kpi(self):
        """
        根据之前的字典更新kpi监控指标, 原来的数据只起到确定格式, 没有实际用途
        其实可以在这一步把QA需要替换的全局变量给替换了,就不需要框架来做了,重组下qa的yaml
        kpi_name 与框架强相关, 要随框架更新, 目前是支持了单个value替换结果
        """
        cmd = "wget https://xly-devops.bj.bcebos.com/PaddleTest/{}.tar.gz --no-proxy \
            && tar xf {}.tar.gz".format(
            self.repo_name, self.repo_name
        )
        os.system(cmd)  # 下载并解压PaddleClas

        self.get_result_yaml()  # 更新yaml
        for (key, value) in self.base_yaml_dict.items():
            with open(os.path.join("../cases", value), "r") as f:
                content = yaml.load(f, Loader=yaml.FullLoader)
            globals()[key] = content
        content = {}
        for i, case_value in enumerate(self.case_info_list):
            # print('###case_value111', case_value)
            # print('###case_value111', content.keys())
            # print('###case_value111["model_name"]', case_value["model_name"])
            # print("    ")
            # input()
            if case_value["model_name"] not in content.keys():
                content[case_value["model_name"]] = eval(case_value["model_name"].split("-")[2])
                # print('###case_value222', content[case_value["model_name"]])
                # print("    ")
                content = self.pop_yaml_useless(content)
                # print('###content', content)
                # print('###content', type(content))
                # print("    ")
                with open(os.path.join("PaddleClas", case_value["model_name"].replace("-", "/") + ".yaml"), "r") as f:
                    content_rd_yaml = yaml.load(f, Loader=yaml.FullLoader)
                if "ATTRMetric" in str(content_rd_yaml):
                    self.kpi_value_eval = "label_f1"
                elif "Recallk" in str(content_rd_yaml):
                    self.kpi_value_eval = "recall1"
                elif "TopkAcc" in str(content_rd_yaml):
                    self.kpi_value_eval = "loss"
                else:
                    print("### use default kpi_value_eval {}".format(content_rd_yaml["Metric"]))
                    self.kpi_value_eval = "loss"
                content = json.dumps(content)
                content = content.replace("${{{0}}}".format("kpi_value_eval"), self.kpi_value_eval)
                content = json.loads(content)

            if case_value["kpi_name"] != "exit_code":
                for index, tag_value in enumerate(
                    content[case_value["model_name"]]["case"][case_value["system"]][case_value["tag"].split("_")[0]]
                ):
                    if tag_value["name"] == case_value["tag"].split("_")[1]:
                        if case_value["kpi_value"] != -1.0:  # 异常值不保存
                            print("####case_info_list   kpi_base: {}".format(case_value["kpi_base"]))
                            print("####case_info_list   kpi_value: {}".format(case_value["kpi_value"]))
                            print(
                                "####case_info_list   change: {}".format(
                                    content[case_value["model_name"]]["case"][case_value["system"]][
                                        case_value["tag"].split("_")[0]
                                    ][index]["result"][case_value["kpi_name"]]["base"]
                                )
                            )
                            content[case_value["model_name"]]["case"][case_value["system"]][
                                case_value["tag"].split("_")[0]
                            ][index]["result"][case_value["kpi_name"]]["base"] = case_value["kpi_value"]
                            if (
                                "-HRNet" in case_value["model_name"]
                                or "-LeViT" in case_value["model_name"]
                                or "-SwinTransformer" in case_value["model_name"]
                            ) and (
                                case_value["tag"].split("_")[0] == "train" or case_value["tag"].split("_")[0] == "eval"
                            ):
                                content[case_value["model_name"]]["case"][case_value["system"]][
                                    case_value["tag"].split("_")[0]
                                ][index]["result"][case_value["kpi_name"]]["threshold"] = 1
                                content[case_value["model_name"]]["case"][case_value["system"]][
                                    case_value["tag"].split("_")[0]
                                ][index]["result"][case_value["kpi_name"]]["evaluation"] = "-"
                            # 这里进行替换时要考虑到全局变量如何替换
            # print('###content333', content)
            # print('###content333', type(content))
            # print("    ")
            # input()
        with open(os.path.join("report_linux_cuda102_py37_develop.yaml"), "w") as f:  # 会删除之前的，重新生成一份
            yaml.dump(content, f, sort_keys=False)


def run():
    """
    执行入口
    """
    # kpi_value_eval="loss"
    # with open(os.path.join("../cases", "ppcls-configs-ImageNet-ResNet-ResNet50.yaml"), "r") as f:
    #     content = yaml.load(f, Loader=yaml.FullLoader)
    # print('###content',type(content))
    # print('###content',content["case"]["linux"]["eval"])
    # content = json.dumps(content)
    # content = content.replace("${{{0}}}".format("kpi_value_eval"), kpi_value_eval)
    # content = json.loads(content)
    # print('###content',content["case"]["linux"]["eval"])
    # input()

    model = PaddleClas_Collect()
    model.update_kpi()
    model.clean_report()
    return 0


if __name__ == "__main__":
    run()
