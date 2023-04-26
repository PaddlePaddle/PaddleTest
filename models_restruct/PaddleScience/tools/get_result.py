# encoding: utf-8
"""
根据之前执行的结果获取kpi值
"""
import os
import time
import sys
import copy
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

    def __init__(self, update_name, update_url):
        """
        初始化参数
        """
        self.repo_name = "PaddleClas"
        self.update_name = update_name
        print(" ")
        print(" ")
        print(" ")
        print("####update_name {}".format(self.update_name))
        self.update_url = update_url
        self.save_name = update_name + ".yaml"
        # print('###self.update_name', self.update_name)
        # print('###self.update_url', self.update_url)
        # print('###self.save_name', self.save_name)
        # input()
        self.base_yaml_dict = {
            "ImageNet": "ppcls^configs^ImageNet^ResNet^ResNet50.yaml",
            "slim": "ppcls^configs^slim^PPLCNet_x1_0_quantization.yaml",
            "DeepHash": "ppcls^configs^DeepHash^DCH.yaml",
            "GeneralRecognition": "ppcls^configs^GeneralRecognition^GeneralRecognition_PPLCNet_x2_5.yaml",
            "Cartoonface": "ppcls^configs^Cartoonface^ResNet50_icartoon.yaml",
            "GeneralRecognitionV2": "ppcls^configs^GeneralRecognitionV2^GeneralRecognitionV2_PPLCNetV2_base.yaml",
            "Logo": "ppcls^configs^Logo^ResNet50_ReID.yaml",
            "Products": "ppcls^configs^Products^ResNet50_vd_Inshop.yaml",
            "Vehicle": "ppcls^configs^Vehicle^ResNet50.yaml",
            "PULC": "ppcls^configs^PULC^car_exists^PPLCNet_x1_0.yaml",
            "reid": "ppcls^configs^reid^strong_baseline^baseline.yaml",
            "metric_learning": "ppcls^configs^metric_learning^adaface_ir18.yaml",
        }

    def download_data(self, value=None):
        """
        下载数据集
        """
        # 调用函数路径已切换至PaddleClas

        tar_name = value.split("/")[-1]
        # print('####tar_name', tar_name)
        if os.path.exists(tar_name.replace(".tar", "")):
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
                # cmd = "wget {} --no-proxy && unzip {} > /dev/null 2>&1".format(value.replace(" ", ""), tar_name)
                os.system(cmd)
            except:
                print("#### start download failed {} failed".format(value.replace(" ", "")))
            os.remove(tar_name)  # 删除下载的tar包防止重名
        return tar_name.replace(".tar", "")

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
        if self.update_url != "None" and os.path.exists(self.report_path):
            shutil.rmtree(self.report_path)
        if os.path.exists(self.repo_name + "-develop.tar.gz"):
            os.remove(self.repo_name + "-develop.tar.gz")
        if os.path.exists(self.repo_name):
            shutil.rmtree(self.repo_name)

    def get_result_yaml(self):
        """
        获取yaml结果
        """
        self.case_info_list = list()
        # 如果是字典也可以在这里重新支持
        self.report_path = self.download_data(self.update_url)
        for case_detail in self.load_json():
            labels = case_detail.get("labels")
            for label in labels:
                if label.get("name") == "case_info":
                    self.case_info_list.extend(json.loads(label.get("value")))

    def get_base_result(self, data_json):
        """
        获取原始base yaml中result信息
        """
        if isinstance(data_json, dict):
            for key, val in data_json.items():
                if key == "base":
                    with open(os.path.join("..", val), "r") as f:
                        content_base = yaml.load(f, Loader=yaml.FullLoader)
                if isinstance(data_json[key], dict):
                    self.get_base_result(data_json[key])
                elif isinstance(data_json[key], list):
                    # print('###data_json[key]', data_json[key])
                    data_json_copy = copy.deepcopy(data_json[key])
                    for i, name in enumerate(data_json_copy):
                        for key1, val1 in name.items():
                            if key1 == "name":
                                # print('###key1', key1)
                                # print('###key', key)
                                # print('###val1', val1)
                                for key_, val_ in content_base.items():
                                    if key == key_:
                                        for name_ in val_:
                                            if name_["name"] == val1:
                                                # print('###key_', key_)
                                                # print('###name_', name_)
                                                try:
                                                    # print('###name_', name_['result'])
                                                    data_json[key][i]["result"] = name_["result"]
                                                except:
                                                    pass
                                                #     print("###key {} val1 {} do not have result".format(key, val1))
                                                # print('###data_json[key]', data_json[key])
                                                # input()
                        # print('###name11111', name)
                        # input()
                        # for key1 in key_pop:
                        #     name.pop(key1)
                        # print('###name2222', name)
                        # input()
        return data_json

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
        if self.update_url == "None":  # 判空
            return 0

        if os.path.exists(self.repo_name):
            print("#### already download {}".format(self.repo_name))
        else:
            cmd = "wget https://xly-devops.bj.bcebos.com/PaddleTest/{}/{}-develop.tar.gz --no-proxy \
                && tar xf {}-develop.tar.gz && mv {}-develop {}".format(
                self.repo_name, self.repo_name, self.repo_name, self.repo_name, self.repo_name
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
            if case_value["model_name"] not in content.keys():
                content[case_value["model_name"]] = eval(case_value["model_name"].split("^")[2])
                # print('###case_value model_name', content[case_value["model_name"]])
                # print("    ")
                # 删除yaml中无效的params
                # print('###content000', content)
                content = self.pop_yaml_useless(content)
                # print('###content111', content)
                # 从base中获取result的格式填充到case中
                # print("    ")
                # print('###content model_name 111', content[case_value["model_name"]])
                content[case_value["model_name"]] = self.get_base_result(content[case_value["model_name"]])
                # print('###content model_name 222', content[case_value["model_name"]])
                # print("    ")
                # input()
                with open(os.path.join(self.repo_name, case_value["model_name"].replace("^", "/") + ".yaml"), "r") as f:
                    # 针对dy2st 进行.replace("_dy2st_convergence","")
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
                    # print('@@@@tag_value', tag_value)
                    # print('@@@@case_value tag', case_value["tag"].split("_", 1)[1])
                    # print('@@@@',\
                    #  content[case_value["model_name"]]["case"][case_value["system"]][case_value["tag"].split("_")[0]])
                    if tag_value["name"] == case_value["tag"].split("_", 1)[1]:
                        if case_value["kpi_value"] != -1.0:  # 异常值不保存
                            try:
                                # print("####case_info_list   model_name: {}".format(case_value["model_name"]))
                                # print("####case_info_list   case tag is : {}".format(case_value["tag"]))
                                # print("####case_info_list   kpi_name: {}".format(case_value["kpi_name"]))
                                # print("####case_info_list   kpi_base: {}".format(case_value["kpi_base"]))
                                # print("####case_info_list   kpi_value: {}".format(case_value["kpi_value"]))
                                content[case_value["model_name"]]["case"][case_value["system"]][
                                    case_value["tag"].split("_")[0]
                                ][index]["result"][case_value["kpi_name"]]["base"] = case_value["kpi_value"]
                                # print(
                                #     "####case_info_list   change: {}".format(
                                #         content[case_value["model_name"]]["case"][case_value["system"]][
                                #             case_value["tag"].split("_")[0]
                                #         ]
                                #     )
                                # )
                                # print(
                                #     "####case_info_list   change: {}".format(
                                #         content[case_value["model_name"]]["case"][case_value["system"]][
                                #             case_value["tag"].split("_")[0]
                                #         ][index]["result"][case_value["kpi_name"]]["base"]
                                #     )
                                # )
                                # input()
                            except:
                                print(
                                    "###maybe {}  already change class_ids to exit_code ".format(
                                        case_value["model_name"]
                                    )
                                )

                        # 单独处理固定不了随机量的 HRNet、 LeViT、 SwinTransformer、 VisionTransformer
                        if (
                            "^HRNet" in case_value["model_name"]
                            or "^PPLCNetV2_base" in case_value["model_name"]
                            or "^InceptionV3" in case_value["model_name"]
                            or "^LeViT" in case_value["model_name"]
                            or "^SwinTransformer" in case_value["model_name"]
                            or "^VisionTransformer" in case_value["model_name"]
                        ) and (case_value["tag"].split("_")[0] == "train" or case_value["tag"].split("_")[0] == "eval"):
                            print("### {} change threshold and evaluation ".format(case_value["model_name"]))
                            content[case_value["model_name"]]["case"][case_value["system"]][
                                case_value["tag"].split("_")[0]
                            ][index]["result"][case_value["kpi_name"]]["threshold"] = 99
                            content[case_value["model_name"]]["case"][case_value["system"]][
                                case_value["tag"].split("_")[0]
                            ][index]["result"][case_value["kpi_name"]]["evaluation"] = "-"
                            # 这里进行替换时要考虑到全局变量如何替换

            # 单独处理固定不了随机量的HRNet、LeViT、SwinTransformer  predict阶段防止不替换
            for index, tag_value in enumerate(
                content[case_value["model_name"]]["case"][case_value["system"]][case_value["tag"].split("_")[0]]
            ):
                if tag_value["name"] == case_value["tag"].split("_", 1)[1]:
                    if case_value["kpi_value"] != -1.0:  # 异常值不保存
                        # 处理存在随机量导致每次infer_trained结果不一致的情况
                        if (
                            (
                                "^HRNet" in case_value["model_name"]
                                or "^PPLCNetV2_base" in case_value["model_name"]
                                or "^InceptionV3" in case_value["model_name"]
                                or "^LeViT" in case_value["model_name"]
                                or "^SwinTransformer" in case_value["model_name"]
                                or "^VisionTransformer" in case_value["model_name"]
                            )
                            and (
                                case_value["tag"].split("_")[0] == "infer"
                                or case_value["tag"].split("_")[0] == "predict"
                            )
                            and (
                                tag_value["name"] == "trained"
                                or tag_value["name"] == "trained_mkldnn"
                                or tag_value["name"] == "trained_trt"
                            )
                        ):
                            try:  # 增加尝试方式报错，定死指标为class_ids 变成退出码 exit_code
                                print("### {} change class_ids to exit_code ".format(case_value["model_name"]))
                                dict_tmp = content[case_value["model_name"]]["case"][case_value["system"]][
                                    case_value["tag"].split("_")[0]
                                ][index]["result"]
                                dict_tmp.update({"exit_code": dict_tmp.pop("class_ids")})
                                content[case_value["model_name"]]["case"][case_value["system"]][
                                    case_value["tag"].split("_")[0]
                                ][index]["result"] = dict_tmp

                                content[case_value["model_name"]]["case"][case_value["system"]][
                                    case_value["tag"].split("_")[0]
                                ][index]["result"]["exit_code"]["base"] = 0
                                content[case_value["model_name"]]["case"][case_value["system"]][
                                    case_value["tag"].split("_")[0]
                                ][index]["result"]["exit_code"]["threshold"] = 0
                                content[case_value["model_name"]]["case"][case_value["system"]][
                                    case_value["tag"].split("_")[0]
                                ][index]["result"]["exit_code"]["evaluation"] = "="
                            except:
                                print("###can not change class_ids to exit_code")
            # print('###content333', content)
            # print('###content333', type(content))
            # print("    ")
            # input()
        with open(self.save_name, "w") as f:  # 会删除之前的，重新生成一份
            yaml.dump(content, f)  # 每次位置是一致的
            # yaml.dump(content, f, sort_keys=False)

        if os.path.exists(self.save_name):
            bce_whl_url = os.getenv("bce_whl_url")
            tar_name = "bos_new.tar.gz"
            python_name = "BosClient.py"
            if os.path.exists(tar_name) is False:
                wget.download(bce_whl_url)
            tf = tarfile.open(tar_name)
            tf.extractall(os.getcwd())
            os.system("python -m pip install bce-python-sdk")

            exit_code = os.system("python {} {} paddle-qa/PaddleMT/PaddleClas/".format(python_name, self.save_name))
            print("### upload {} and exit_code is  {}".format(self.save_name, exit_code))

            # 上传备份
            shutil.copyfile(
                self.save_name,
                self.save_name.replace(".yaml", time.strftime("_%Y_%m_%d", time.gmtime(time.time())) + ".yaml"),
            )
            exit_code = os.system(
                "python {} {} paddle-qa/PaddleMT/PaddleClas/date_kpi_value/".format(
                    python_name,
                    self.save_name.replace(".yaml", time.strftime("_%Y_%m_%d", time.gmtime(time.time())) + ".yaml"),
                )
            )
            print(
                "### upload {} and exit_code is  {}".format(
                    self.save_name.replace(".yaml", time.strftime("_%Y_%m_%d", time.gmtime(time.time())) + ".yaml"),
                    exit_code,
                )
            )

            #   暂不打开备份上传, check产出的结果
            if os.path.exists(self.save_name):
                os.remove(self.save_name)
            #     os.remove(self.save_name.replace(".yaml", \
            #         time.strftime("_%Y_%m_%d", time.gmtime(time.time())) + ".yaml"))
            if os.path.exists(python_name):
                os.remove(tar_name)
                os.remove(python_name)
                os.remove("bos_sample_conf.py")
                os.remove("sample.log")
                os.remove("sts_sample_conf.py")
                os.remove("StsClient.py")
                shutil.rmtree("__pycache__")
        else:
            print("### 未生成 {} ".format(self.save_name))


def run():
    """
    执行入口
    """
    # kpi_value_eval="loss"
    # with open(os.path.join("../cases", "ppcls^configs^ImageNet^ResNet^ResNet50.yaml"), "r") as f:
    #     content = yaml.load(f, Loader=yaml.FullLoader)
    # print('###content',type(content))
    # print('###content',content["case"]["linux"]["eval"])
    # content = json.dumps(content)
    # content = content.replace("${{{0}}}".format("kpi_value_eval"), kpi_value_eval)
    # content = json.loads(content)
    # print('###content',content["case"]["linux"]["eval"])
    # input()

    # update_name = {
    #     "PaddleClas-Linux-Cuda112-Python38-P0-Release": "21615184/result.tar",
    #     "PaddleClas-Linux-Cuda116-Python39-P0-Release": "21615143/result.tar",
    #     "PaddleClas-Linux-Cuda117-Python310-P0-Release": "21615164/result.tar",
    #     "PaddleClas-Linux-Cuda116-Python39-P0-Release-Centos": "21615167/result.tar",
    #     "PaddleClas-Linux-Cuda102-Python37-P1-Develop": "21615088/result.tar",
    # }

    # update_name = {
    #     "PaddleClas-Linux-Cuda102-Python37-P0-Develop": "22253813/result.tar",
    # }

    update_name = {
        "PaddleClas-Linux-Cuda102-Python37-P0-Develop": "22430823/result.tar",
        "PaddleClas-Linux-Cuda102-Python37-P1-Develop": "22430795/result.tar",
        "PaddleClas-Linux-Cuda102-Python37-P11-Develop": "22429738/result.tar",
        "PaddleClas-Linux-Cuda102-Python37-P12-Develop": "22429734/result.tar",
        "PaddleClas-Linux-Cuda112-Python38-P0-Develop": "22430901/result.tar",
        "PaddleClas-Linux-Cuda116-Python39-P0-Develop": "22430894/result.tar",
        "PaddleClas-Linux-Cuda117-Python310-P0-Develop": "22430885/result.tar",
        "PaddleClas-Linux-Cuda116-Python39-P0-Develop-Centos": "22415821/result.tar",
    }

    # update_name = {
    #     "PaddleClas-Linux-Cuda102-Python37-P0-Develop": "22311290/result.tar",
    #     "PaddleClas-Linux-Cuda102-Python37-P1-Develop": "22311301/result.tar",
    #     "PaddleClas-Linux-Cuda102-Python37-P11-Develop": "22310533/result.tar",
    #     "PaddleClas-Linux-Cuda102-Python37-P12-Develop": "22310540/result.tar",
    #     "PaddleClas-Linux-Cuda112-Python38-P0-Develop": "22305717/result.tar",
    #     "PaddleClas-Linux-Cuda116-Python39-P0-Develop": "22305700/result.tar",
    #     "PaddleClas-Linux-Cuda117-Python310-P0-Develop": "22311315/result.tar",
    #     "PaddleClas-Linux-Cuda116-Python39-P0-Develop-Centos": "22311386/result.tar",
    #     "PaddleClas-Linux-Cuda102-Python37-P0-Release": "22311319/result.tar",
    #     "PaddleClas-Linux-Cuda102-Python37-P1-Release": "22311317/result.tar",
    #     "PaddleClas-Linux-Cuda102-Python37-P11-Release": "22310525/result.tar",
    #     "PaddleClas-Linux-Cuda102-Python37-P12-Release": "22305047/result.tar",
    #     "PaddleClas-Linux-Cuda112-Python38-P0-Release": "22311282/result.tar",
    #     "PaddleClas-Linux-Cuda116-Python39-P0-Release": "22311353/result.tar",
    #     "PaddleClas-Linux-Cuda117-Python310-P0-Release": "22311377/result.tar",
    #     "PaddleClas-Linux-Cuda116-Python39-P0-Release-Centos": "22311372/result.tar",
    # }

    for (key, value) in update_name.items():
        if value != "None":
            value = "https://paddle-qa.bj.bcebos.com/PaddleMT/allure_result/" + value
        model = PaddleClas_Collect(key, value)
        model.update_kpi()
        # print("暂时关闭清理！！！！")
        model.clean_report()
    return 0


if __name__ == "__main__":
    try:
        os.environ["TZ"] = "Asia/Shanghai"
        time.tzset()
    except:
        print(" do not set time tz")
    run()
