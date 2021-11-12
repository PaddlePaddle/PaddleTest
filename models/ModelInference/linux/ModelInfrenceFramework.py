# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file
  * @author jiaxiao01
  * @date 2021/9/3 3:46 PM
  * @brief clas model inference test framework
  *
  **************************************************************************/
"""

import subprocess
import re
import os
import json
import ast
import pytest
from pytest_assume.plugin import assume
from pytest import approx
import numpy as np


def clean_process():
    """
    clean process
    """
    print("This is clean_process!")
    pid = os.getpid()
    cmd = """ps aux| grep python | grep -v %s | awk '{print $2}'| xargs kill -9;""" % pid
    subprocess.getstatusoutput(cmd)


class RepoInit:
    """
      git clone repo
      """

    def __init__(self, repo, branch):
        self.repo = repo
        self.branch = branch
        print("This is Repo Init!")
        pid = os.getpid()
        http_proxy=os.environ.get('http_proxy')
        cmd='''ps aux| grep python | grep -v %s | awk '{print $2}'| xargs kill -9;
                rm -rf %s;
                export http_proxy=%s;
                export https_proxy=%s;
                git clone https://github.com/paddlepaddle/%s.git;
                cd %s; git checkout %s; python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple;
                unset https_proxy;
                unset http_proxy;''' % (pid, self.repo, http_proxy, http_proxy, self.repo, self.repo, self.branch)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)


class RepoInstructions:
    """
      Repo Dataset
      """

    def __init__(self, cmd):
        self.cmd = cmd
        repo_result = subprocess.getstatusoutput(self.cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "configure failed!   log information:%s" % output


class TestClasInference:
    """
      Clas Inference
      """

    def __init__(self, model, yaml):
        self.model = model
        self.yaml = yaml

    def test_get_pretrained_model(self):
        """
          get pretrained model
          """
        legendary_models = ["ResNet50", "ResNet50_vd", "MobileNetV3_large_x1_0", "VGG16", "PPLCNet_x1_0"]
        if self.model in legendary_models:
            cmd = (
                "unset https_proxy; unset http_proxy;cd PaddleClas; \
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/%s_pretrained.pdparams"
                % self.model
            )
        else:
            cmd = (
                "unset https_proxy; unset http_proxy; cd PaddleClas; \
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/%s_pretrained.pdparams"
                % self.model
            )
        clas_result = subprocess.getstatusoutput(cmd)
        exit_code = clas_result[0]
        output = clas_result[1]
        assert exit_code == 0, "downlooad  model pretrained failed!   log information:%s" % output

    def test_clas_export_model(self):
        """
          clas export model
          """
        cmd = (
            "cd PaddleClas;python tools/export_model.py -c %s -o Global.pretrained_model=./%s_pretrained \
-o Global.save_inference_dir=./inference/%s"
            % (self.yaml, self.model, self.model)
        )
        clas_result = subprocess.getstatusoutput(cmd)
        exit_code = clas_result[0]
        output = clas_result[1]
        assert exit_code == 0, "export model failed!   log information:%s" % output

    def test_clas_predict(self, expect_id, expect_score):
        """
          clas predict
          """
        cmd_gpu = (
            """cd PaddleClas; cd deploy; python python/predict_cls.py -c configs/inference_cls.yaml \
-o Global.inference_model_dir=../inference/%s -o Global.batch_size=1 \
-o Global.use_gpu=True -o Global.use_tensorrt=False -o Global.enable_mkldnn=False"""
            % self.model
        )
        cmd_trt = (
            """cd PaddleClas; cd deploy; python python/predict_cls.py -c configs/inference_cls.yaml \
-o Global.inference_model_dir=../inference/%s -o Global.batch_size=1 \
-o Global.use_gpu=True -o Global.use_tensorrt=True -o Global.enable_mkldnn=False"""
            % self.model
        )
        cmd_fp16 = (
            """cd PaddleClas; cd deploy; python python/predict_cls.py -c configs/inference_cls.yaml \
-o Global.inference_model_dir=../inference/%s -o Global.batch_size=1 \
-o Global.use_gpu=True -o Global.use_tensorrt=True -o  Global.use_fp16=True -o Global.enable_mkldnn=False"""
            % self.model
        )
        cmd_cpu = (
            """cd PaddleClas; cd deploy; python python/predict_cls.py -c configs/inference_cls.yaml \
-o Global.inference_model_dir=../inference/%s -o Global.batch_size=1 \
-o Global.use_gpu=False -o Global.use_tensorrt=False -o Global.enable_mkldnn=False"""
            % self.model
        )
        cmd_mkldnn = (
            """cd PaddleClas; cd deploy; python python/predict_cls.py -c configs/inference_cls.yaml \
-o Global.inference_model_dir=../inference/%s -o Global.batch_size=1 \
-o Global.use_gpu=False -o Global.use_tensorrt=False -o Global.enable_mkldnn=True"""
            % self.model
        )

        for cmd in [cmd_gpu, cmd_trt, cmd_fp16, cmd_cpu, cmd_mkldnn]:
            clas_result = subprocess.getstatusoutput(cmd)
            exit_code = clas_result[0]
            output = clas_result[1]
            print(cmd)
            # check exit_code
            assert exit_code == 0, "predict model failed!   log information:%s" % output

            for line in output.split("\n"):
                if "class id(s)" in line:
                    # output_score_id=ast.iteral_eval(line)
                    output_list = re.findall(r"\[(.*?)\]", line)

            clas_id = output_list[0]
            print("clas_id:{}".format(clas_id))
            clas_score = output_list[1]
            print("clas_score:{}".format(clas_score))

            with assume:
                assert clas_id == expect_id, "check clas_id failed!   real clas_id is: %s, expect clas_id is: %s" % (
                    clas_id,
                    expect_id,
                )
            if cmd != cmd_fp16:
                with assume:
                    assert (
                        clas_score == expect_score
                    ), """check clas_score failed!
                        real clas_score is: %s, expect clas_score is: %s""" % (
                        clas_score,
                        expect_score,
                    )
            print("*************************************************************************")


class TestClasRecInference:
    """
      Clas Rec Inference
      """

    def __init__(self, yaml, infer_imgs):
        self.yaml = yaml
        self.infer_imgs = infer_imgs

    def test_get_clas_rec_inference_model(self, model):
        """
          get clas_rec inference mode
          """
        cmd = (
            "unset https_proxy; unset http_proxy; cd PaddleClas/deploy; cd models; \
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/%s.tar;\
 tar xf %s.tar; rm -rf %s.tar; cd .."
            % (model, model, model)
        )
        print(cmd)
        rec_result = subprocess.getstatusoutput(cmd)
        exit_code = rec_result[0]
        output = rec_result[1]
        assert exit_code == 0, "downlooad  model pretrained failed!   log information:%s" % output

    def test_clas_rec_predict(self, expect_bbox, expect_rec_docs, expect_rec_scores):
        """
          clas rec predict
          """
        cmd_gpu = """cd PaddleClas/deploy; python python/predict_system.py -c %s -o Global.infer_imgs=%s \
-o Global.use_gpu=True -o Global.use_tensorrt=False \
-o Global.use_fp16=False -o Global.enable_mkldnn=False""" % (
            self.yaml,
            self.infer_imgs,
        )
        cmd_trt = """cd PaddleClas/deploy; python python/predict_system.py -c %s -o Global.infer_imgs=%s \
-o Global.use_gpu=True -o Global.use_tensorrt=True \
-o Global.use_fp16=False -o Global.enable_mkldnn=False""" % (
            self.yaml,
            self.infer_imgs,
        )
        cmd_fp16 = """cd PaddleClas/deploy; python python/predict_system.py -c %s -o Global.infer_imgs=%s \
-o Global.use_gpu=True -o Global.use_tensorrt=True \
-o Global.use_fp16=True -o Global.enable_mkldnn=False""" % (
            self.yaml,
            self.infer_imgs,
        )
        cmd_cpu = """cd PaddleClas/deploy; python python/predict_system.py -c %s -o Global.infer_imgs=%s \
-o Global.use_gpu=False -o Global.use_tensorrt=False \
  -o Global.use_fp16=False -o Global.enable_mkldnn=False""" % (
            self.yaml,
            self.infer_imgs,
        )
        cmd_mkldnn = """cd PaddleClas/deploy; python python/predict_system.py -c %s -o Global.infer_imgs=%s \
-o Global.use_gpu=False -o Global.use_tensorrt=False \
  -o Global.use_fp16=False -o Global.enable_mkldnn=True""" % (
            self.yaml,
            self.infer_imgs,
        )

        for cmd in [cmd_gpu, cmd_trt, cmd_fp16, cmd_cpu, cmd_mkldnn]:
            clas_rec_result = subprocess.getstatusoutput(cmd)
            exit_code = clas_rec_result[0]
            output = clas_rec_result[1]
            # check exit_code
            assert exit_code == 0, "predict model failed!   log information:%s" % output
            # check bbox,rec_docs,rec_scores
            for line in output.split("\n"):
                if "bbox" in line:
                    output_bbox = ast.literal_eval(line)
            output_bbox = output_bbox[0]

            bbox = output_bbox["bbox"]
            rec_docs = output_bbox["rec_docs"]
            rec_scores = output_bbox["rec_scores"]

            print(cmd)
            print(output_bbox)
            print("bbox:{}".format(bbox))
            print("rec_docs:{}".format(rec_docs))
            print("rec_scores:{}".format(rec_scores))

            with assume:
                assert bbox == expect_bbox, (
                    "check bbox failed!   \
                   real bbox is: %s, expect bbox is: %s"
                    % (bbox, expect_bbox)
                )
            with assume:
                assert (
                    rec_docs == expect_rec_docs
                ), """check bbox failed!
                   real rec_docs is: %s, expect rec_docs is: %s """ % (
                    rec_docs,
                    expect_rec_docs,
                )
            with assume:
                assert rec_scores == approx(expect_rec_scores, abs=1e-2), (
                    "check rec_scores failed! \
                   real rec_scores is: %s, expect rec_scores is: %s"
                    % (rec_scores, expect_rec_scores)
                )
            print("*************************************************************************")


class TestOcrRecInference:
    """
      ocr rec model inference test framework
      """

    def __init__(
        self,
        model,
        infer_imgs,
        rec_char_dict,
        yaml="configs/rec/rec_mv3_none_bilstm_ctc.yml",
        algorithm="CRNN",
        rec_image_shape="3,32,100",
        use_space_char=False,
    ):
        self.model = model
        self.infer_imgs = infer_imgs
        self.rec_char_dict = rec_char_dict
        self.yaml = yaml
        self.algorithm = algorithm
        self.rec_image_shape = rec_image_shape
        self.use_space_char = use_space_char

    def test_get_ocr_rec_train_model(self, category="dygraph_v2.0/en"):
        """
          get ocr_rec train model
          """
        cmd = """cd PaddleOCR; wget https://paddleocr.bj.bcebos.com/%s/%s_train.tar; tar xf %s_train.tar;
                 rm -rf %s_train.tar; """ % (
            category,
            self.model,
            self.model,
            self.model,
        )
        print(cmd)
        rec_result = subprocess.getstatusoutput(cmd)
        exit_code = rec_result[0]
        output = rec_result[1]
        assert exit_code == 0, "downlooad  model pretrained failed!   log information:%s" % output

    def test_ocr_rec_export_model(self):
        """
           ocr_det_export_model(
           """
        cmd = (
            "cd PaddleOCR; python tools/export_model.py -c %s -o Global.pretrained_model=%s_train/best_accuracy \
                Global.save_inference_dir=%s_infer"
            % (self.yaml, self.model, self.model)
        )
        ocr_det_result = subprocess.getstatusoutput(cmd)
        exit_code = ocr_det_result[0]
        output = ocr_det_result[1]
        # check exit_code
        assert exit_code == 0, "export_model failed!   log information:%s" % output

    def test_get_ocr_rec_inference_model(self, category):
        """
          get ocr_rec inference mode
          """
        custom_models = [
            "chinese_cht_mobile_v2.0_rec",
            "ka_mobile_v2.0_rec",
            "ta_mobile_v2.0_rec",
            "te_mobile_v2.0_rec",
            "german_mobile_v2.0_rec",
        ]
        if self.model in custom_models:
            cmd = """cd PaddleOCR; wget -P %s_infer https://paddleocr.bj.bcebos.com/%s/%s_infer.tar; \
cd %s; tar xf %s_infer.tar; rm -rf %s_infer.tar; cd ..""" % (
                self.model,
                category,
                self.model,
                self.model,
                self.model,
                self.model,
            )
        else:
            cmd = """cd PaddleOCR; wget https://paddleocr.bj.bcebos.com/%s/%s_infer.tar; tar xf %s_infer.tar;
                    rm -rf %s_infer.tar""" % (
                category,
                self.model,
                self.model,
                self.model,
            )
        print(cmd)
        rec_result = subprocess.getstatusoutput(cmd)
        exit_code = rec_result[0]
        output = rec_result[1]
        assert exit_code == 0, "downlooad  model pretrained failed!   log information:%s" % output

    def test_ocr_rec_predict(self, expect_rec_docs, expect_rec_scores):
        """
          ocr rec predict
          """
        cmd_gpu = (
            "cd PaddleOCR; python tools/infer/predict_rec.py --image_dir=%s --rec_model_dir=%s_infer \
--rec_image_shape=%s --rec_char_dict_path=%s --rec_algorithm=%s --use_gpu=True --use_tensorrt=False \
--precision=fp32 --enable_mkldnn=False --rec_batch_num=1  --use_space_char=%s"
            % (
                self.infer_imgs,
                self.model,
                self.rec_image_shape,
                self.rec_char_dict,
                self.algorithm,
                self.use_space_char,
            )
        )
        cmd_trt = (
            "cd PaddleOCR; python tools/infer/predict_rec.py --image_dir=%s --rec_model_dir=%s_infer \
--rec_image_shape=%s --rec_char_dict_path=%s --rec_algorithm=%s --use_gpu=True --use_tensorrt=True \
 --precision=fp32 --enable_mkldnn=False --rec_batch_num=1  --use_space_char=%s"
            % (
                self.infer_imgs,
                self.model,
                self.rec_image_shape,
                self.rec_char_dict,
                self.algorithm,
                self.use_space_char,
            )
        )
        cmd_cpu = (
            "cd PaddleOCR; python tools/infer/predict_rec.py --image_dir=%s --rec_model_dir=%s_infer \
--rec_image_shape=%s--rec_char_dict_path=%s --rec_algorithm=%s --use_gpu=False --use_tensorrt=False \
--precision=fp32 --enable_mkldnn=False --rec_batch_num=1  --use_space_char=%s"
            % (
                self.infer_imgs,
                self.model,
                self.rec_image_shape,
                self.rec_char_dict,
                self.algorithm,
                self.use_space_char,
            )
        )
        cmd_mkldnn = (
            "cd PaddleOCR; python tools/infer/predict_rec.py --image_dir=%s --rec_model_dir=%s_infer \
--rec_image_shape=%s --rec_char_dict_path=%s --rec_algorithm=%s --use_gpu=False --use_tensorrt=False  \
 --precision=fp32 --enable_mkldnn=True --rec_batch_num=1  --use_space_char=%s"
            % (
                self.infer_imgs,
                self.model,
                self.rec_image_shape,
                self.rec_char_dict,
                self.algorithm,
                self.use_space_char,
            )
        )
        for cmd in [cmd_gpu, cmd_trt, cmd_cpu, cmd_mkldnn]:
            print(cmd)
            clas_rec_result = subprocess.getstatusoutput(cmd)
            exit_code = clas_rec_result[0]
            output = clas_rec_result[1]
            # check exit_code
            assert exit_code == 0, "predict model failed!   log information:%s" % output
            # check bbox,rec_docs,rec_scores
            for line in output.split("\n"):
                if "Predicts of" in line:
                    output_rec = line
            output_rec_list = re.findall(r"\((.*?)\)", output_rec)

            rec_docs = output_rec_list[0].split(",")[0].strip("'")
            rec_scores = output_rec_list[0].split(",")[1]
            rec_scores = float(rec_scores)

            print("rec_docs:{}".format(rec_docs))
            print("rec_scores:{}".format(rec_scores))

            with assume:
                assert rec_docs == expect_rec_docs, (
                    "check rec_docs failed! real rec_docs is: %s,\
                            expect rec_docs is: %s"
                    % (rec_docs, expect_rec_docs)
                )
            with assume:
                assert rec_scores == approx(expect_rec_scores, abs=1e-2), (
                    "check rec_scores failed!   real rec_scores is: %s, \
                            expect rec_scores is: %s"
                    % (rec_scores, expect_rec_scores)
                )
            print("*************************************************************************")


class TestOcrDetInference:
    """
      ocr model inference test framework
      """

    def __init__(self, model, infer_imgs, yaml="configs/det/det_r50_vd_east.yml", algorithm="DB"):
        self.model = model
        self.infer_imgs = infer_imgs
        self.yaml = yaml
        self.algorithm = algorithm

    def test_get_ocr_det_inference_model(self, category):
        """
          get ocr_det inference mode
          """
        cmd = """cd PaddleOCR; wget https://paddleocr.bj.bcebos.com/%s/%s_infer.tar; tar xf %s_infer.tar;
                 rm -rf %s_infer.tar""" % (
            category,
            self.model,
            self.model,
            self.model,
        )
        print(cmd)
        rec_result = subprocess.getstatusoutput(cmd)
        exit_code = rec_result[0]
        output = rec_result[1]
        assert exit_code == 0, "downlooad  model pretrained failed!   log information:%s" % output

    def test_get_ocr_det_train_model(self, category):
        """
          get ocr_det inference mode
          """
        cmd = """cd PaddleOCR; wget https://paddleocr.bj.bcebos.com/%s/%s_train.tar; tar xf %s_train.tar;
                 rm -rf %s_train.tar""" % (
            category,
            self.model,
            self.model,
            self.model,
        )
        rec_result = subprocess.getstatusoutput(cmd)
        exit_code = rec_result[0]
        output = rec_result[1]
        assert exit_code == 0, "downlooad  model pretrained failed!   log information:%s" % output

    def test_ocr_det_export_model(self):
        """
           ocr_det_export_model(
           """
        cmd = (
            "cd PaddleOCR; python tools/export_model.py -c %s -o Global.pretrained_model=%s_train/best_accuracy \
                Global.save_inference_dir=%s_infer"
            % (self.yaml, self.model, self.model)
        )
        ocr_det_result = subprocess.getstatusoutput(cmd)
        exit_code = ocr_det_result[0]
        output = ocr_det_result[1]
        # check exit_code
        assert exit_code == 0, "export_model failed!   log information:%s" % output

    def test_ocr_det_predict(self, expect_det_bbox):
        """
          ocr det predict
          """
        cmd_gpu = (
            "cd PaddleOCR; python tools/infer/predict_det.py --image_dir=%s --det_model_dir=%s_infer \
--det_algorithm=%s  --use_gpu=True --use_tensorrt=False --enable_mkldnn=False "
            % (self.infer_imgs, self.model, self.algorithm)
        )
        cmd_trt = (
            "cd PaddleOCR; python tools/infer/predict_det.py --image_dir=%s --det_model_dir=%s_infer \
--det_algorithm=%s --use_gpu=True --use_tensorrt=True --enable_mkldnn=False "
            % (self.infer_imgs, self.model, self.algorithm)
        )
        cmd_cpu = (
            "cd PaddleOCR; python tools/infer/predict_det.py --image_dir=%s --det_model_dir=%s_infer \
--det_algorithm=%s --use_gpu=False --use_tensorrt=False --enable_mkldnn=False "
            % (self.infer_imgs, self.model, self.algorithm)
        )
        cmd_mkldnn = (
            "cd PaddleOCR; python tools/infer/predict_det.py --image_dir=%s --det_model_dir=%s_infer \
--det_algorithm=%s --use_gpu=False --use_tensorrt=False --enable_mkldnn=True "
            % (self.infer_imgs, self.model, self.algorithm)
        )
        for cmd in [cmd_gpu, cmd_trt, cmd_cpu, cmd_mkldnn]:
            print(cmd)
            ocr_det_result = subprocess.getstatusoutput(cmd)
            exit_code = ocr_det_result[0]
            output = ocr_det_result[1]
            # check exit_code
            assert exit_code == 0, "predict model failed!   log information:%s" % output
            # check det_bbox
            for line in output.split("\n"):
                if "img_10.jpg" in line:
                    output_det = line
                    print(output_det)
                    break

            det_bbox = output_det.split("\t")[-1]
            det_bbox = ast.literal_eval(det_bbox)

            print("det_bbox:{}".format(det_bbox))

            with assume:
                assert np.array(det_bbox) == approx(np.array(expect_det_bbox), abs=2), (
                    "check det_bbox failed!  \
                           real det_bbox is: %s, expect det_bbox is: %s"
                    % (det_bbox, expect_det_bbox)
                )
            print("*************************************************************************")


class TestOcrClsInference:
    """
      ocr clas model inference test framework
      """

    def __init__(self, model, infer_imgs, cls_char_dict):
        self.model = model
        self.infer_imgs = infer_imgs
        self.cls_char_dict = cls_char_dict

    def test_get_ocr_cls_inference_model(self, category):
        """
          get ocr_cls inference mode
          """
        cmd = """cd PaddleOCR; wget https://paddleocr.bj.bcebos.com/%s/%s.tar; tar xf %s.tar;
                rm -rf %s.tar""" % (
            category,
            self.model,
            self.model,
            self.model,
        )
        rec_result = subprocess.getstatusoutput(cmd)
        exit_code = rec_result[0]
        output = rec_result[1]
        assert exit_code == 0, "downlooad  model pretrained failed!   log information:%s" % output

    def test_ocr_cls_predict(self, expect_cls_docs, expect_cls_scores):
        """
          ocr cls predict
          """
        cmd_gpu = (
            "cd PaddleOCR; python tools/infer/predict_cls.py --image_dir=%s --cls_model_dir=%s \
--use_gpu=True --use_tensorrt=False --precision=fp32 --enable_mkldnn=False --cls_batch_num=1"
            % (self.infer_imgs, self.model)
        )
        cmd_trt = (
            "cd PaddleOCR; python tools/infer/predict_cls.py --image_dir=%s --cls_model_dir=%s \
--use_gpu=True --use_tensorrt=True --precision=fp32 --enable_mkldnn=False --cls_batch_num=1"
            % (self.infer_imgs, self.model)
        )
        cmd_cpu = (
            "cd PaddleOCR; python tools/infer/predict_cls.py --image_dir=%s --cls_model_dir=%s \
--use_gpu=False --use_tensorrt=False --precision=fp32 --enable_mkldnn=False --cls_batch_num=1"
            % (self.infer_imgs, self.model)
        )
        cmd_mkldnn = (
            "cd PaddleOCR; python tools/infer/predict_cls.py --image_dir=%s --cls_model_dir=%s \
--use_gpu=False --use_tensorrt=False  --precision=fp32 --enable_mkldnn=True --cls_batch_num=1"
            % (self.infer_imgs, self.model)
        )
        for cmd in [cmd_gpu, cmd_trt, cmd_cpu, cmd_mkldnn]:
            clas_cls_result = subprocess.getstatusoutput(cmd)
            exit_code = clas_cls_result[0]
            output = clas_cls_result[1]
            # check exit_code
            assert exit_code == 0, "predict model failed!   log information:%s" % output
            # check bbox,cls_docs,cls_scores
            for line in output.split("\n"):
                if "Predicts of" in line:
                    output_cls = line
            output_cls_list = re.findall(r"\[(.*?)\]", output_cls)

            cls_docs = output_cls_list[1].split(",")[0].strip("'")
            cls_scores = output_cls_list[1].split(",")[1]
            cls_scores = float(cls_scores)

            print(cmd)
            print("cls_docs:{}".format(cls_docs))
            print("cls_scores:{}".format(cls_scores))

            with assume:
                assert cls_docs == expect_cls_docs, (
                    "check cls_docs failed! \
                           real cls_docs is: %s, expect cls_docs is: %s"
                    % (cls_docs, expect_cls_docs)
                )
            with assume:
                assert cls_scores == approx(expect_cls_scores, abs=1e-5), (
                    "check cls_scores failed! \
                           real cls_scores is: %s, expect cls_scores is: %s"
                    % (cls_scores, expect_cls_scores)
                )
            print("**************************************************************************")
