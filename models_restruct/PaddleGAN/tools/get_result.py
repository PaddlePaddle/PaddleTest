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
class PaddleGAN_Collect(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化参数
        """
        self.repo_name = "PaddleGAN"
        self.whl_branch = "release"  # develop release
        # pytest结果下载地址
        self.report_linux_cuda102_py37_release = {
            "P0": "hhttps://paddle-qa.bj.bcebos.com/PaddleMT/allure_result/20342790/report/result.tar",
            "P1": "hhttps://paddle-qa.bj.bcebos.com/PaddleMT/allure_result/20426590/report/result.tar",
        }

        self.report_linux_cuda102_py37_develop = {
            "P0": "hhttps://paddle-qa.bj.bcebos.com/PaddleMT/allure_result/20342790/report/result.tar",
            "P1": "hhttps://paddle-qa.bj.bcebos.com/PaddleMT/allure_result/20426590/report/result.tar",
        }

        # self.report_linux_cuda102_py37_develop = {
        #     "P2_2": "hhttps://paddle-qa.bj.bcebos.com/PaddleMT/allure_result/19909321/report/result.tar"
        # }

        self.base_yaml_dict = {
            "base": "configs^edvr_m_wo_tsa.yaml",
            "single_only": "configs^lapstyle_draft.yaml",
        }

    def download_data(self, priority="P0", value=None):
        """
        下载数据集
        """
        # 调用函数路径已切换至PaddleGAN

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
        os.remove(self.repo_name + "-develop.tar.gz")
        shutil.rmtree(self.repo_name)

    def get_result_yaml(self):
        """
        获取yaml结果
        """
        self.report_path_list = list()
        self.case_info_list = list()
        if self.whl_branch == "develop":
            content = self.report_linux_cuda102_py37_develop
            self.content_name = "report_linux_cuda102_py37_develop.yaml"
        else:
            content = self.report_linux_cuda102_py37_release
            self.content_name = "report_linux_cuda102_py37_release.yaml"
        for (key, value) in content.items():
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
        if os.path.exists(self.repo_name):
            print("#### already download {}".format(self.repo_name))
        else:
            cmd = "wget https://xly-devops.bj.bcebos.com/PaddleTest/{}/{}-develop.tar.gz --no-proxy \
                && tar xf {}-develop.tar.gz && mv {}-develop {}".format(
                self.repo_name, self.repo_name, self.repo_name, self.repo_name, self.repo_name
            )
            os.system(cmd)  # 下载并解压PaddleGAN

        self.get_result_yaml()  # 更新yaml
        for (key, value) in self.base_yaml_dict.items():
            with open(os.path.join("../cases", value), "r") as f:
                content = yaml.load(f, Loader=yaml.FullLoader)
            globals()[key] = content  # 这里在干吗？
        content = {}
        for i, case_value in enumerate(self.case_info_list):
            # print('###case_value111', case_value)
            # print('###case_value111', content.keys())
            # print('###case_value111["model_name"]', case_value["model_name"])
            # print("    ")
            # input()
            if case_value["model_name"] not in content.keys():
                if (
                    "lapstyle_draft" in case_value["model_name"]
                    or "lapstyle_rev_first" in case_value["model_name"]
                    or "lapstyle_rev_second" in case_value["model_name"]
                    or "singan_finetune" in case_value["model_name"]
                    or "singan_animation" in case_value["model_name"]
                    or "singan_sr" in case_value["model_name"]
                    or "singan_universal" in case_value["model_name"]
                    or "prenet" in case_value["model_name"]
                    or "firstorder_vox_mobile_256" in case_value["model_name"]
                ):
                    source_yaml_name = "single_only"
                elif (
                    "cond_dcgan_mnist" in case_value["model_name"]
                    or "makeup" in case_value["model_name"]
                    or "wgan_mnist" in case_value["model_name"]
                    or "firstorder_vox_mobile_256" in case_value["model_name"]
                ):
                    continue
                else:
                    source_yaml_name = "base"
                content[case_value["model_name"]] = eval(source_yaml_name)
                # print('###case_value222', content[case_value["model_name"]])
                # print("    ")
                # input()
                content = self.pop_yaml_useless(content)
                # print('###content', content)
                # print('###content', type(content))
                # print("    ")

                if "singan_sr" in case_value["model_name"]:
                    self.kpi_value_eval = "D_gradient_penalty"
                elif "singan_universal" in case_value["model_name"] or "singan_animation" in case_value["model_name"]:
                    self.kpi_value_eval = "D_gradient_penalty"
                elif "singan_finetune" in case_value["model_name"]:
                    self.kpi_value_eval = "D_gradient_penalty"
                elif (
                    "firstorder_vox_mobile_256" in case_value["model_name"]
                    or "firstorder_vox_256" in case_value["model_name"]
                    or "firstorder_fashion" in case_value["model_name"]
                    or "aotgan" in case_value["model_name"]
                ):
                    self.kpi_value_eval = "perceptual"
                elif (
                    "basicvsr_reds" in case_value["model_name"]
                    or "basicvsr++_vimeo90k_BD" in case_value["model_name"]
                    or "lesrcnn_psnr_x4_div2k" in case_value["model_name"]
                    or "edvr_l_wo_tsa" in case_value["model_name"]
                    or "basicvsr++_reds" in case_value["model_name"]
                    or "pan_psnr_x4_div2k" in case_value["model_name"]
                    or "iconvsr_reds" in case_value["model_name"]
                    or "edvr_l_w_tsa" in case_value["model_name"]
                    or "edvr_m_w_tsa" in case_value["model_name"]
                    or "esrgan_psnr_x4_div2k" in case_value["model_name"]
                    or "edvr_m_wo_tsa" in case_value["model_name"]
                    or "esrgan_psnr_x2_div2k" in case_value["model_name"]
                    or "prenet" in case_value["model_name"]
                    or "rcan_rssr_x4" in case_value["model_name"]
                ):
                    self.kpi_value_eval = "loss_pixel"
                elif (
                    "msvsr_vimeo90k_BD" in case_value["model_name"]
                    or "realsr_bicubic_noise_x4_df2k" in case_value["model_name"]
                    or "realsr_kernel_noise_x4_dped" in case_value["model_name"]
                    or "msvsr_reds" in case_value["model_name"]
                    or "esrgan_x4_div2k" in case_value["model_name"]
                ):
                    self.kpi_value_eval = "loss_pix"
                elif "drn_psnr_x4_div2k" in case_value["model_name"]:
                    self.kpi_value_eval = "loss_dual"
                elif "lapstyle_draft" in case_value["model_name"]:
                    self.kpi_value_eval = "loss_c"
                elif "mprnet_denoising" in case_value["model_name"] or "mprnet_deraining" in case_value["model_name"]:
                    self.kpi_value_eval = "loss"
                elif "stylegan_v2_256_ffhq" in case_value["model_name"]:
                    self.kpi_value_eval = "l_d"
                elif "animeganv2_pretrain" in case_value["model_name"]:
                    self.kpi_value_eval = "init_c_loss"
                elif (
                    "cyclegan_cityscapes" in case_value["model_name"]
                    or "cyclegan_horse2zebra" in case_value["model_name"]
                ):
                    self.kpi_value_eval = "G_idt_A_loss"
                elif "photopen" in case_value["model_name"]:
                    self.kpi_value_eval = "g_featloss"
                elif "starganv2_celeba_hq" in case_value["model_name"] or "starganv2_afhq" in case_value["model_name"]:
                    self.kpi_value_eval = "G/latent_adv"
                elif (
                    "ugatit_selfie2anime_light" in case_value["model_name"]
                    or "ugatit_photo2cartoon" in case_value["model_name"]
                ):
                    self.kpi_value_eval = "discriminator_loss"
                elif "animeganv2" in case_value["model_name"]:
                    self.kpi_value_eval = "d_loss"
                elif (
                    "pix2pix_facades" in case_value["model_name"]
                    or "pix2pix_cityscapes_2gpus" in case_value["model_name"]
                    or "lapstyle_rev_first" in case_value["model_name"]
                    or "lapstyle_rev_second" in case_value["model_name"]
                    or "pix2pix_cityscapes" in case_value["model_name"]
                ):
                    self.kpi_value_eval = "D_fake_loss"
                else:
                    print("### use default kpi_value_eval loss")
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

                            # # 单独处理固定不了随机量的HRNet、LeViT、SwinTransformer
                            # if (
                            #     "^HRNet" in case_value["model_name"]
                            #     or "^LeViT" in case_value["model_name"]
                            #     or "^SwinTransformer" in case_value["model_name"]
                            # ) and (
                            #     case_value["tag"].split("_")[0] == \
                            #       "train" or case_value["tag"].split("_")[0] == "eval"
                            # ):
                            #     print("### {} change threshold and evaluation ".format(case_value["model_name"]))
                            #     content[case_value["model_name"]]["case"][case_value["system"]][
                            #         case_value["tag"].split("_")[0]
                            #     ][index]["result"][case_value["kpi_name"]]["threshold"] = 1
                            #     content[case_value["model_name"]]["case"][case_value["system"]][
                            #         case_value["tag"].split("_")[0]
                            #     ][index]["result"][case_value["kpi_name"]]["evaluation"] = "-"

            # print('###content333', content)
            # print('###content333', type(content))
            # print("    ")
            # input()
        with open(os.path.join(self.content_name), "w") as f:  # 会删除之前的，重新生成一份
            yaml.dump(content, f)  # 每次位置是一致的
            # yaml.dump(content, f, sort_keys=False)


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

    model = PaddleGAN_Collect()
    model.update_kpi()
    # print("暂时关闭清理！！！！")
    model.clean_report()
    return 0


if __name__ == "__main__":
    run()
