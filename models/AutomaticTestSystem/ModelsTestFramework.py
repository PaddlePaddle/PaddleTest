# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
  * @file
  * @author jiaxiao01
  * @date 2022/9/2 3:46 PM
  * @brief  model ats
  *
  **************************************************************************/
"""


import subprocess
import re
import ast
import logging
import os
import os.path
import platform
import filecmp
import allure
from plot_paddle_torch import *
import chardet
import paddle
import numpy as np
import yaml
import pytest
from pytest_assume.plugin import assume
from pytest import approx
from utility import *

rec_image_shape_dict = {"CRNN": "3,32,100", "ABINet": "3,32,128", "ViTSTR": "1,224,224", "VisionLAN": "3,64,256"}


def metricExtraction(keyword, output):
    """
    metricExtraction
    """
    for line in output.split("\n"):
        if (keyword + ":" in line) and ("best_accuracy" not in line):
            output_rec = line
            break
    print(output_rec)
    metric = output_rec.split(":")[-1]
    print(metric)
    return metric


def platformAdapter(cmd):
    """
    platformAdapter
    """
    if platform.system() == "Windows":
        cmd = cmd.replace(";", "&")
        cmd = cmd.replace("sed", "%sed%")
        cmd = cmd.replace("rm -rf", "rd /s /q")
        cmd = cmd.replace("export", "set")
    if platform.system() == "Darwin":
        cmd = cmd.replace("sed -i", 'sed -i ""')
    return cmd


class RepoInit:
    """
    RepoInit
    """

    def __init__(self, repo):
        """
        repo_init
        """
        self.repo = repo
        print("This is Repo Init!")
        cmd = """git clone -b dygraph https://github.com/paddlepaddle/%s.git --depth 1; \
cd %s; python -m pip install -r requirements.txt; python -m pip install -r ppstructure/kie/requirements.txt; cd ..; \
python -m pip install paddleocr; python -m pip install paddleclas; \
git clone -b develop https://github.com/paddlepaddle/PaddleDetection.git --depth 1; \
cd PaddleDetection; python -m pip install -r requirements.txt; cd ..;""" % (
            self.repo,
            self.repo,
        )
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        print(output)
        assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
        logging.info("git clone" + self.repo + "sucessfuly!")


class RepoInit3D:
    """
    RepoInit3D
    """

    def __init__(self, repo):
        """
        3d_init
        """
        self.repo = repo
        print("This is Repo Init!")
        cmd = """git clone -b develop https://github.com/paddlepaddle/%s.git --depth 1; cd %s; \
               python -m pip install -r requirements.txt; \
               python -m pip uninstall -y paddle3d; python -m pip install .""" % (
            self.repo,
            self.repo,
        )
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        print(output)
        assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
        logging.info("git clone" + self.repo + "sucessfuly!")


class RepoDataset3D:
    """
    RepoDataset3D
    """

    def __init__(self):
        """
        3d_init
        """
        sysstr = platform.system()
        if sysstr == "Linux":
            print("config Linux data_path")
            cmd = """cd Paddle3D; rm -rf datasets; ln -s /ssd2/ce_data/Paddle3D datasets;"""

        elif sysstr == "Windows":
            print("config windows data_path")
            cmd = """cd Paddle3D & rd /s /q datasets & mklink /j datasets E:\\ce_data\\Paddle3D"""
        elif sysstr == "Darwin":
            print("config mac data_path")
            cmd = """cd PaddleOCR; rm -rf datasets; ln -s /Users/paddle/PaddleTest/ce_data/Paddle3D datasets"""
        else:
            print("Other System tasks")
            exit(1)
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "configure failed!   log information:%s" % output


class RepoInitSpeech:
    """
    RepoInitSpeech
    """

    def __init__(self, repo):
        """
        speech_init
        """
        self.repo = repo
        print("This is Repo Init!")
        cmd = """git clone -b r1.1 https://github.com/paddlepaddle/%s.git --depth 1; cd %s; \
               yum update; yum install libsndfile -y; \
               python -m pip uninstall -y paddlespeech; python -m pip install .""" % (
            self.repo,
            self.repo,
        )
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
        logging.info("git clone" + self.repo + "sucessfuly!")


class RepoDatasetSpeech:
    """
    RepoDatasetSpeech
    """

    def __init__(self):
        """
        speech_init
        """
        sysstr = platform.system()
        if sysstr == "Linux":
            print("config Linux data_path")
            cmd = """cd PaddleSpeech/examples/zh_en_tts/tts3; rm -rf dump; \
                     ln -s /ssd2/ce_data/PaddleSpeech_t2s/preprocess_data/zh_en_tts3/dump dump; \
                     cd ../../csmsc/vits; rm -rf dump; \
                     ln -s /ssd2/ce_data/PaddleSpeech_t2s/preprocess_data/vits/dump dump; cd ../../../..; \
                     wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav \
                     https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav \
                     https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav \
                     https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav \
                     https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/test_long_audio_01.wav \
                     https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav; \
                     echo -e "demo1 85236145389.wav \n demo2 85236145389.wav" > vec.job"""
        else:
            cmd = 'wget -c https://paddlespeech.bj.bcebos.com/PaddleAudio/cat.wav \
                   https://paddlespeech.bj.bcebos.com/PaddleAudio/dog.wav \
                   https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav \
                   https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav \
                   https://paddlespeech.bj.bcebos.com/datasets/single_wav/zh/test_long_audio_01.wav \
                   https://paddlespeech.bj.bcebos.com/vector/audio/85236145389.wav; \
                   echo -e "demo1 85236145389.wav \n demo2 85236145389.wav" > vec.job'
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "configure failed!   log information:%s" % output


class RepoDataset:
    """
    RepoDataset
    """

    def __init__(self):
        """
        repo_init
        """
        self.config = yaml.load(open("TestCase.yaml", "rb"), Loader=yaml.Loader)
        sysstr = platform.system()
        if sysstr == "Linux":
            print("config Linux data_path")
            data_path = self.config["data_path"]["linux_data_path"]
            print(data_path)
            cmd = """cd PaddleOCR; rm -rf train_data; ln -s %s train_data;cd ..; cd PaddleDetection; \
                   rm -rf dataset; ln -s %s dataset""" % (
                data_path,
                data_path,
            )

        elif sysstr == "Windows":
            print("config windows data_path")
            data_path = self.config["data_path"]["windows_data_path"]
            print(data_path)
            cmd = """cd PaddleOCR & rd /s /q train_data & mklink /j train_data %s; cd PaddleDetection; \
                   rd /s /q dataset; mklink /j dataset %s""" % (
                data_path,
                data_path,
            )
        elif sysstr == "Darwin":
            print("config mac data_path")
            data_path = self.config["data_path"]["mac_data_path"]
            print(data_path)
            cmd = """cd PaddleOCR; rm -rf train_data; ln -s %s train_data; cd PaddleDetection; \
                   rm -rf dataset; ln -s %s dataset""" % (
                data_path,
                data_path,
            )
        else:
            print("Other System tasks")
            exit(1)
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "configure failed!   log information:%s" % output
        logging.info("configure dataset sucessfuly!")
        cmd = """cd PaddleOCR; \
wget -P pretrain_models https://paddle-qa.bj.bcebos.com/rocm/abinet_vl_pretrained.pdparams; \
cd pretrain_models \
wget https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams; cd .."""
        cmd = platformAdapter(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "pretrain_models configure failed!   log information:%s" % output
        if (platform.system() == "Windows") or (platform.system() == "Linux"):
            repo_result = subprocess.getstatusoutput(cmd)
            exit_code = repo_result[0]
            output = repo_result[1]
            assert exit_code == 0, "tensorRT dynamic shape configure  failed!   log information:%s" % output


class TestOcrModelFunction:
    """
    TestOcrModelFunction
    """

    def __init__(self, model="", yml="", category=""):
        """
        ocr+init
        """
        if model != "":
            self.model = model
            self.yaml = yml
            self.category = category
            self.testcase_yml = yaml.load(open("TestCase.yaml", "rb"), Loader=yaml.Loader)
            self.tar_name = os.path.splitext(os.path.basename(self.testcase_yml[self.model]["eval_pretrained_model"]))[
                0
            ]
            self.dataset = self.testcase_yml[self.model]["dataset"]

    def test_ocr_cli(self, cmd):
        """
        test_ocr_cli
        """
        cmd = platformAdapter(cmd)
        print(cmd)
        cmd_result = subprocess.getstatusoutput(cmd)
        exit_code = cmd_result[0]
        output = cmd_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "cli")

    def test_ocr_train(self, use_gpu):
        """
        test_ocr_train
        """
        if self.category == "rec":
            cmd = self.testcase_yml["cmd"][self.category]["train"] % (self.yaml, self.yaml, use_gpu, self.model)
        elif self.category == "picodet/legacy_model/application/layout_analysis":
            cmd = self.testcase_yml["cmd"][self.category]["train"] % (use_gpu)
            if self.model == "picodet_lcnet_x2_5_layout":
                cmd = (
                    cmd
                    + " --slim_config \
                    configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x2_5_layout.yml"
                )
        else:
            cmd = self.testcase_yml["cmd"][self.category]["train"] % (self.yaml, use_gpu, self.model)

        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
            cmd = cmd.replace("sed", "%sed%")
            cmd = cmd.replace("export", "set")
        if platform.system() == "Darwin":
            cmd = cmd.replace("sed -i", 'sed -i ""')
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        log_dir = "PaddleOCR/log_" + self.model
        exit_check_fucntion(exit_code, output, "train", log_dir)

    def test_ocr_train_acc(self):
        """
        test_ocr_train_acc
        """
        # if self.category=='rec':
        if self.model == "rec_vitstr_none_ce":
            data1 = getdata("log/rec/" + self.model + "_paddle.log", "loss:", ", avg_reader_cost")
            data2 = getdata("log/rec/" + self.model + "_torch.log", "tensor\\(", ", device=")
            allure.attach.file(
                "log/rec/" + self.model + "_paddle.log",
                name=self.model + "_paddle.log",
                attachment_type=allure.attachment_type.TEXT,
            )
            allure.attach.file(
                "log/rec/" + self.model + "_torch.log",
                name=self.model + "_torch.log",
                attachment_type=allure.attachment_type.TEXT,
            )
            plot_paddle_torch_loss(data1, data2, self.model)
            allure.attach.file(
                "paddle_torch_train_loss.png",
                name="paddle_torch_train_loss.png",
                attachment_type=allure.attachment_type.PNG,
            )
        elif self.model == "rec_r45_abinet":
            data1 = getdata_custom("log/rec/" + self.model + "_paddle.log", ", loss:", ", avg_reader_cost")
            data2 = getdata("log/rec/" + self.model + "_torch.log", "loss =", ",  smooth")

            allure.attach.file(
                "log/rec/" + self.model + "_paddle.log",
                name=self.model + "_paddle.log",
                attachment_type=allure.attachment_type.TEXT,
            )
            allure.attach.file(
                "log/rec/" + self.model + "_torch.log",
                name=self.model + "_torch.log",
                attachment_type=allure.attachment_type.TEXT,
            )
            plot_paddle_torch_loss(data1, data2, self.model)
            allure.attach.file(
                "paddle_torch_train_loss.png",
                name="paddle_torch_train_loss.png",
                attachment_type=allure.attachment_type.PNG,
            )
        elif self.model == "table_master":
            data1 = getdata("log/table/" + self.model + "_paddle.log", ", loss:", ", horizon_bbox_loss")
            data2 = getdata("log/table/" + self.model + "_torch.log", ", loss:", ", grad_norm")
            allure.attach.file(
                "log/table/" + self.model + "_paddle.log",
                name=self.model + "_paddle.log",
                attachment_type=allure.attachment_type.TEXT,
            )
            allure.attach.file(
                "log/table/" + self.model + "_torch.log",
                name=self.model + "_torch.log",
                attachment_type=allure.attachment_type.TEXT,
            )
            plot_paddle_torch_loss(data1, data2, self.model)
        else:
            pass

    def test_ocr_get_pretrained_model(self):
        """
        test_ocr_get_pretrained_model
        """
        if (self.category == "table") or (self.category == "kie/vi_layoutxlm") or (self.category == "e2e"):
            cmd = self.testcase_yml["cmd"][self.category]["get_pretrained_model"] % (
                self.testcase_yml[self.model]["eval_pretrained_model"],
                self.tar_name,
                self.tar_name,
                self.model,
            )
        elif self.category == "picodet/legacy_model/application/layout_analysis":
            cmd = self.testcase_yml["cmd"][self.category]["get_pretrained_model"]
        else:
            cmd = self.testcase_yml["cmd"][self.category]["get_pretrained_model"] % (
                self.testcase_yml[self.model]["eval_pretrained_model"],
                self.tar_name,
                self.model,
                self.model,
            )

        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
            cmd = cmd.replace("rm -rf", "del")
            cmd = cmd.replace("mv", "ren")
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "eval")

    def test_ocr_eval(self, use_gpu):
        """
        test_ocr_eval
        """
        if self.category == "picodet/legacy_model/application/layout_analysis":
            cmd = self.testcase_yml["cmd"][self.category]["eval"] % (use_gpu)
            if self.model == "picodet_lcnet_x2_5_layout":
                cmd = (
                    cmd
                    + " --slim_config \
                    configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x2_5_layout.yml"
                )
        else:
            cmd = self.testcase_yml["cmd"][self.category]["eval"] % (self.yaml, use_gpu, self.model)
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        cmd = cmd.replace("_udml.yml", ".yml")
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "eval")
        """
          if self.category=='rec' or self.category=='table':
             keyword='acc'
          elif (self.category=='det') or (self.category=='table') or (self.category=='kie/vi_layoutxlm'):
             keyword='hmean'
          elif self.category=='sr':
             keyword='psnr_avg'
          else:
             pass


          real_metric=metricExtraction(keyword, output)
          expect_metric=self.testcase_yml[self.model]['eval_'+keyword]

          # attach
          body="expect_"+keyword+": "+str(expect_metric)
          allure.attach(body, 'expect_metric', allure.attachment_type.TEXT)
          body="real_"+keyword+": "+real_metric
          allure.attach(body, 'real_metric', allure.attachment_type.TEXT)

          # assert
          real_metric=float(real_metric)
          expect_metric=float(expect_metric)
          with assume: assert real_metric == approx(expect_metric, abs=3e-2),\
                          "check eval_acc failed!   real eval_acc is: %s, \
                            expect eval_acc is: %s" % (real_metric, expect_metric)
          """

    def test_ocr_rec_infer(self, use_gpu):
        """
        test_ocr_rec_infer
        """
        if self.category == "picodet/legacy_model/application/layout_analysis":
            cmd = self.testcase_yml["cmd"][self.category]["infer"] % (use_gpu)
            if self.model == "picodet_lcnet_x2_5_layout":
                cmd = (
                    cmd
                    + " --slim_config  \
configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x2_5_layout.yml"
                )
        else:
            cmd = self.testcase_yml["cmd"][self.category]["infer"] % (self.yaml, use_gpu, self.model)
        if self.model == "re_vi_layoutxlm_xfund_zh":
            cmd = cmd.replace("infer_kie_token_ser", "infer_kie_token_ser_re")
            cmd = (
                cmd
                + " -c_ser configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml  \
-o_ser Architecture.Backbone.checkpoints=./ser_vi_layoutxlm_xfund_zh"
            )
        cmd = cmd.replace("_udml.yml", ".yml")
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")

        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "infer")
        check_infer_metric(self.category, output, self.dataset)

    def test_ocr_export_model(self, use_gpu):
        """
        test_ocr_export_model
        """
        if self.category == "picodet/legacy_model/application/layout_analysis":
            cmd = self.testcase_yml["cmd"][self.category]["export_model"] % (use_gpu)
            if self.model == "picodet_lcnet_x2_5_layout":
                cmd = (
                    cmd
                    + " --slim_config \
configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x2_5_layout.yml"
                )
        else:
            cmd = self.testcase_yml["cmd"][self.category]["export_model"] % (self.yaml, use_gpu, self.model, self.model)
        cmd = cmd.replace("_udml.yml", ".yml")
        print(cmd)
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "export_model")

    def test_ocr_rec_predict(self, use_gpu, use_tensorrt, enable_mkldnn):
        """
        test_ocr_rec_predict
        """
        if self.category == "rec":
            model_config = yaml.load(open(os.path.join("PaddleOCR", self.yaml), "rb"), Loader=yaml.Loader)
            algorithm = model_config["Architecture"]["algorithm"]
            rec_image_shape = rec_image_shape_dict[algorithm]
            rec_char_dict_path = self.testcase_yml[self.model]["rec_char_dict_path"]

            print(rec_image_shape)
            cmd = self.testcase_yml["cmd"][self.category]["predict"] % (
                self.model,
                rec_image_shape,
                algorithm,
                rec_char_dict_path,
                use_gpu,
                use_tensorrt,
                enable_mkldnn,
            )
        elif self.category == "det":
            model_config = yaml.load(open(os.path.join("PaddleOCR", self.yaml), "rb"), Loader=yaml.Loader)
            algorithm = model_config["Architecture"]["algorithm"]
            cmd = self.testcase_yml["cmd"][self.category]["predict"] % (
                self.model,
                algorithm,
                use_gpu,
                use_tensorrt,
                enable_mkldnn,
            )
        elif self.category == "table":
            cmd = self.testcase_yml["cmd"][self.category]["predict"] % (
                self.model,
                use_gpu,
                use_tensorrt,
                enable_mkldnn,
            )
        elif self.category == "sr":
            sr_image_shape = self.testcase_yml[self.model]["sr_image_shape"]
            cmd = self.testcase_yml["cmd"][self.category]["predict"] % (
                self.model,
                sr_image_shape,
                use_gpu,
                use_tensorrt,
                enable_mkldnn,
            )
        elif (self.category == "kie/vi_layoutxlm") or (self.category == "e2e"):
            cmd = self.testcase_yml["cmd"][self.category]["predict"] % (
                self.model,
                use_gpu,
                use_tensorrt,
                enable_mkldnn,
            )
        elif self.category == "picodet/legacy_model/application/layout_analysis":
            if use_gpu is True:
                use_gpu = "gpu"
            else:
                use_gpu = "cpu"
            cmd = self.testcase_yml["cmd"][self.category]["predict"] % (use_gpu)

        if self.model == "SLANet":
            cmd = self.testcase_yml["cmd"][self.category]["predict_SLANet"] % (
                self.model,
                use_gpu,
                use_tensorrt,
                enable_mkldnn,
            )

        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        detection_result = subprocess.getstatusoutput(cmd)
        print(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "predict")
        # acc
        # metricExtraction('Predicts', output)
        check_predict_metric(self.category, output, self.dataset)

    def test_ocr_predict_recovery(self, use_gpu):
        """
        test_ocr_predict_recovery
        """
        cmd = "cd PaddleOCR; python -m pip install -r ppstructure/recovery/requirements.txt; \
        mkdir inference && cd inference; \
        wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && tar xf en_PP-OCRv3_det_infer.tar; \
        wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar && tar xf en_PP-OCRv3_rec_infer.tar; \
        wget https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar \
        && tar xf en_ppstructure_mobile_v2.0_SLANet_infer.tar; \
        wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar \
        && tar xf picodet_lcnet_x1_0_fgd_layout_infer.tar; cd ..; \
        python ppstructure/predict_system.py --image_dir=./ppstructure/docs/table/1.png \
        --det_model_dir=inference/en_PP-OCRv3_det_infer \
        --rec_model_dir=inference/en_PP-OCRv3_rec_infer --rec_char_dict_path=./ppocr/utils/en_dict.txt \
        --table_model_dir=inference/en_ppstructure_mobile_v2.0_SLANet_infer \
        --table_char_dict_path=./ppocr/utils/dict/table_structure_dict.txt \
        --layout_model_dir=inference/picodet_lcnet_x1_0_fgd_layout_infer \
        --layout_dict_path=./ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt \
        --vis_font_path=./doc/fonts/simfang.ttf --recovery=True --save_pdf=False --output=./output/"
        print(cmd)
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "predict_recovery")


class Test3DModelFunction:
    """
    Test3DModelFunction
    """

    def __init__(self, model, yml, category):
        """
        3d_init
        """
        self.model = model
        self.yaml = yml
        self.category = category

    def test_3D_train(self, use_gpu):
        """
        test_3D_train
        """
        cmd = (
            'cd Paddle3D; rm -rf output; export CUDA_VISIBLE_DEVICES=0; \
             sed -i "/iters/d" %s; sed -i "1i\\iters: 200"  %s ; \
             python -m paddle.distributed.launch --log_dir=log_%s  tools/train.py --config %s \
             --num_workers 2 --log_interval 50 --save_interval 5000'
            % (self.yaml, self.yaml, self.model, self.yaml)
        )

        cmd = platformAdapter(cmd)
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        log_dir = "Paddle3D/log_" + self.model
        exit_check_fucntion(exit_code, output, "train", log_dir)

    def test_3D_get_pretrained_model(self):
        """
        3D_get_pretrained_model
        """
        if self.model == "smoke_dla34_no_dcn_kitti":
            cmd = "cd Paddle3D; mkdir smoke_dla34_no_dcn_kitti; cd smoke_dla34_no_dcn_kitti; \
                   wget https://paddle3d.bj.bcebos.com/models/smoke/smoke_dla34_no_dcn_kitti/model.pdparams;"
        elif self.model == "smoke_hrnet18_no_dcn_kitti":
            cmd = (
                "cd Paddle3D; mkdir %s; cd %s; \
                 wget https://paddle3d.bj.bcebos.com/models/smoke/smoke_hrnet18_no_dcn_kitti/model.pdparams"
                % (self.model, self.model)
            )

        elif (
            self.model == "pointpillars_xyres16_kitti_car"
            or self.model == "pointpillars_xyres16_kitti_cyclist_pedestrian"
        ):
            cmd = (
                "cd Paddle3D; mkdir %s; cd %s; \
            wget https://bj.bcebos.com/paddle3d/models/pointpillar/%s/model.pdparams"
                % (self.model, self.model, self.model)
            )
        elif self.model == "centerpoint_pillars_016voxel_kitti":
            cmd = (
                "cd Paddle3D; mkdir %s; cd %s; \
wget https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_016voxel_kitti/model.pdparams"
                % (self.model, self.model)
            )
        elif self.model == "centerpoint_pillars_02voxel_nuscenes_10sweep":
            cmd = (
                "cd Paddle3D; mkdir %s; cd %s; \
wget https://bj.bcebos.com/paddle3d/models/centerpoint//centerpoint_pillars_02voxel_nuscenes_10_sweep/model.pdparams"
                % (self.model, self.model)
            )
        elif (
            self.model == "squeezesegv3_rangenet21_semantickitti"
            or self.model == "squeezesegv3_rangenet53_semantickitti"
        ):
            cmd = (
                "cd Paddle3D; mkdir %s; cd %s; \
                 wget https://bj.bcebos.com/paddle3d/models/squeezesegv3/%s/model.pdparams"
                % (self.model, self.model, self.model)
            )
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
            cmd = cmd.replace("rm -rf", "del")
            cmd = cmd.replace("mv", "ren")
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "eval")

    def test_3D_eval(self, use_gpu):
        """
        test_3D_eval
        """
        cmd = "cd Paddle3D; python tools/evaluate.py --config %s --num_workers 2 --model %s/model.pdparams" % (
            self.yaml,
            self.model,
        )
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        print(cmd)
        if (
            (self.model == "smoke_dla34_no_dcn_kitti")
            or (self.model == "smoke_hrnet18_no_dcn_kitti")
            or (self.model == "centerpoint_pillars_016voxel_kitti")
            or (self.model == "centerpoint_pillars_02voxel_nuscenes_10sweep")
        ):
            cmd = 'echo "not supported for eval when bs >1"'
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "eval")

    def test_3D_eval_bs1(self, use_gpu):
        """
        test_3D_eval_bs1
        """
        cmd = (
            "cd Paddle3D; python tools/evaluate.py --config %s --num_workers 2 --model %s/model.pdparams --batch_size 1"
            % (self.yaml, self.model)
        )
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "eval")

    def test_3D_export_model(self, use_gpu):
        """
        test_3D_export_model
        """
        if (
            self.model == "squeezesegv3_rangenet21_semantickitti"
            or self.model == "squeezesegv3_rangenet53_semantickitti"
        ):
            cmd = (
                "cd Paddle3D; python tools/export.py --config %s --model %s/model.pdparams \
                 --input_shape 64 1024 --save_dir ./exported_model/%s"
                % (self.yaml, self.model, self.model)
            )
        else:
            cmd = (
                "cd Paddle3D; python tools/export.py --config %s \
                 --model %s/model.pdparams --save_dir ./exported_model/%s"
                % (self.yaml, self.model, self.model)
            )

        print(cmd)
        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "export_model")

    def test_3D_predict_python(self, use_gpu, use_trt):
        """
        test_3D_predict_python
        """
        infer_image = "datasets/KITTI/training/velodyne/000000.bin"
        if self.model == "smoke_dla34_no_dcn_kitti" or self.model == "smoke_hrnet18_no_dcn_kitti":
            infer_image = "datasets/KITTI/training/image_2/000000.png"
            cmd = (
                "cd Paddle3D; python deploy/smoke/python/infer.py \
                 --model_file exported_model/%s/inference.pdmodel \
                 --params_file exported_model/%s/inference.pdiparams --image %s --use_gpu"
                % (self.model, self.model, infer_image)
            )
            if use_trt is True:
                cmd = (
                    "cd Paddle3D; python deploy/smoke/python/infer.py \
                     --model_file exported_model/%s/inference.pdmodel\
                     --params_file exported_model/%s/inference.pdiparams --image %s --collect_dynamic_shape_info \
                     --dynamic_shape_file %s/shape_info.txt; \
                     python deploy/smoke/python/infer.py --model_file exported_model/%s/inference.pdmodel \
                     --params_file exported_model/%s/inference.pdiparams --image %s --use_gpu \
                     --use_trt --dynamic_shape_file %s/shape_info.txt;"
                    % (self.model, self.model, infer_image, self.model, self.model, self.model, infer_image, self.model)
                )
            if paddle.is_compiled_with_cuda() is False:
                cmd = (
                    "cd Paddle3D; python deploy/smoke/python/infer.py \
                     --model_file exported_model/%s/inference.pdmodel \
                     --params_file exported_model/%s/inference.pdiparams --image %s"
                    % (self.model, self.model, infer_image)
                )

        elif self.model == "pointpillars_xyres16_kitti_car":
            cmd = (
                "cd Paddle3D; python deploy/pointpillars/python/infer.py \
                 --model_file exported_model/%s/pointpillars.pdmodel \
                 --params_file exported_model/%s/pointpillars.pdiparams --lidar_file %s \
                 --point_cloud_range 0 -39.68 -3 69.12 39.68 1 --voxel_size .16 .16 4 \
                 --max_points_in_voxel 32  --max_voxel_num 40000"
                % (self.model, self.model, infer_image)
            )
            if paddle.is_compiled_with_cuda() is False:
                cmd = (
                    'cd Paddle3D; sed -i "/config.enable_use_gpu/d" deploy/%s/python/infer.py; \
                     python deploy/pointpillars/python/infer.py \
                     --model_file exported_model/%s/pointpillars.pdmodel \
                     --params_file exported_model/%s/pointpillars.pdiparams --lidar_file %s \
                     --point_cloud_range 0 -39.68 -3 69.12 39.68 1 \
                     --voxel_size .16 .16 4 --max_points_in_voxel 32  --max_voxel_num 40000'
                    % (self.category, self.model, self.model, infer_image)
                )

        elif self.model == "pointpillars_xyres16_kitti_cyclist_pedestrian":
            cmd = (
                "cd Paddle3D; python deploy/pointpillars/python/infer.py \
                 --model_file exported_model/%s/pointpillars.pdmodel \
                 --params_file exported_model/%s/pointpillars.pdiparams --lidar_file %s \
                 --point_cloud_range 0 -19.84 -2.5 47.36 19.84 0.5 --voxel_size .16 .16 3 \
                 --max_points_in_voxel 100 --max_voxel_num 12000"
                % (self.model, self.model, infer_image)
            )
            if paddle.is_compiled_with_cuda() is False:
                cmd = (
                    'cd Paddle3D; sed -i "/config.enable_use_gpu/d" deploy/%s/python/infer.py; \
                     python deploy/pointpillars/python/infer.py \
                     --model_file exported_model/%s/pointpillars.pdmodel \
                     --params_file exported_model/%s/pointpillars.pdiparams --lidar_file %s \
                     --point_cloud_range 0 -19.84 -2.5 47.36 19.84 0.5 --voxel_size .16 .16 3 \
                     --max_points_in_voxel 100 --max_voxel_num 12000'
                    % (self.category, self.model, self.model, infer_image)
                )

        elif (
            self.model == "centerpoint_pillars_016voxel_kitti"
            or self.model == "centerpoint_pillars_02voxel_nuscenes_10sweep"
        ):
            cmd = (
                "cd Paddle3D; python deploy/centerpoint/python/infer.py \
                 --model_file exported_model/%s/centerpoint.pdmodel \
                 --params_file exported_model/%s/centerpoint.pdiparams --lidar_file %s --num_point_dim 4"
                % (self.model, self.model, infer_image)
            )
            if paddle.is_compiled_with_cuda() is False:
                cmd = (
                    'cd Paddle3D; sed -i "/config.enable_use_gpu/d" deploy/%s/python/infer.py; \
                     python deploy/centerpoint/python/infer.py --model_file exported_model/%s/centerpoint.pdmodel \
                     --params_file exported_model/%s/centerpoint.pdiparams --lidar_file %s --num_point_dim 4'
                    % (self.category, self.model, self.model, infer_image)
                )
        elif (
            self.model == "squeezesegv3_rangenet21_semantickitti"
            or self.model == "squeezesegv3_rangenet53_semantickitti"
        ):
            cmd = (
                "cd Paddle3D; python deploy/%s/python/infer.py \
                 --model_file exported_model/%s/centerpoint.pdmodel \
                 --params_file exported_model/%s/squeezesegv3.pdiparams --lidar_file %s \
                 --img_mean 12.12,10.88,0.23,-1.04,0.21 --img_std 12.32,11.47,6.91,0.86,0.16"
                % (self.category, self.model, self.model, infer_image)
            )
            if paddle.is_compiled_with_cuda() is False:
                cmd = (
                    'cd Paddle3D; sed -i "/config.enable_use_gpu/d" deploy/%s/python/infer.py; \
                     python deploy/centerpoint/python/infer.py --model_file exported_model/%s/squeezesegv3.pdmodel \
                     --params_file exported_model/%s/squeezesegv3.pdiparams --lidar_file %s \
                     --img_mean 12.12,10.88,0.23,-1.04,0.21 \
                     --img_std 12.32,11.47,6.91,0.86,0.16'
                    % (self.category, self.model, self.model, infer_image)
                )

        else:
            cmd = 'echo "not supported"'

        if platform.system() == "Windows":
            cmd = cmd.replace(";", "&")
        detection_result = subprocess.getstatusoutput(cmd)
        print(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "predict")


class TestSpeechModelFunction:
    """
    TestSpeechModelFunction
    """

    def __init__(self, model=""):
        """
        speech_init
        """
        self.model = model
        self.testcase_yml = yaml.load(open("TestCaseSpeech.yaml", "rb"), Loader=yaml.Loader)

    def test_speech_cli(self, cmd):
        """
        test_speech_cli
        """
        cmd = platformAdapter(cmd)
        print(cmd)
        cmd_result = subprocess.getstatusoutput(cmd)
        exit_code = cmd_result[0]
        output = cmd_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "cli")

    def test_speech_get_pretrained_model(self):
        """
        test_speech_get_pretrained_model
        """
        cmd = self.testcase_yml[self.model]["get_pretrained_model"]
        cmd = platformAdapter(cmd)
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "get_pretrained_model")

    def test_speech_train(self):
        """
        test_speech_train
        """
        cmd = self.testcase_yml[self.model]["train"]
        cmd = platformAdapter(cmd)
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "train")

    def test_speech_synthesize_e2e(self):
        """
        test_speech_synthesize_e2e
        """
        cmd = self.testcase_yml[self.model]["synthesize_e2e"]
        cmd = platformAdapter(cmd)
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        allure_step(cmd, output)
        exit_check_fucntion(exit_code, output, "synthesize_e2e")
