# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file
  * @author jiaxiao01
  * @date 2021/9/23 3:46 PM
  * @brief clas model inference test framework
  *
  **************************************************************************/
"""

import re
import subprocess
import ast
import os
import logging
import numpy as np
import pytest
from pytest_assume.plugin import assume
from pytest import approx

# 删除文件的方式有变，需要增加 rsync --delete-before -d 220701

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


def dependency_install(package):
    """
    function
    """
    exit_code = 1
    while exit_code:
        cmd = "python -m pip install %s -i https://pypi.tuna.tsinghua.edu.cn/simple;" % package
        os.system(cmd)
        cmd = """python -c 'import %s'""" % package
        exit_code = os.system(cmd)
        print("###exit_code", exit_code)


def exit_check_fucntion(exit_code, output, mode, log_dir=""):
    """
    function
    """
    assert exit_code == 0, " %s  model pretrained failed!   log information:%s" % (mode, output)
    # assert "Error" not in output, "%s  model failed!   log information:%s" % (mode, output)
    # 220729 框架打印无效log导致出现error，暂时规避
    if "ABORT!!!" in output:
        log_dir = os.path.abspath(log_dir)
        all_files = os.listdir(log_dir)
        for file in all_files:
            print(file)
            filename = os.path.join(log_dir, file)
            if os.path.isdir(filename) is False:  # 判断是否是文件
                with open(filename) as file_obj:  # 这里容易出现utf-8字符问题，注意
                    # content = file_obj.read()
                    content = file_obj.read().decode("utf-8", "ignore")
                    print(content)
    assert "ABORT!!!" not in output, "%s  model failed!   log information:%s" % (mode, output)
    logging.info("train model sucessfuly!")
    print(output)


def clean_process():
    """
    function
    """
    print("This is clean_process!")
    pid = os.getpid()
    cmd = """ps aux| grep python | grep -v main.py |grep -v %s | awk '{print $2}'| xargs kill -9;""" % pid
    repo_result = subprocess.getstatusoutput(cmd)
    exit_code = repo_result[0]
    print("###exit_code", exit_code)


class RepoInit:
    """
    class
    """

    def __init__(self, repo):
        self.repo = repo
        print("This is Repo Init!")
        pid = os.getpid()
        cmd = """ps aux| grep python | grep -v main.py |grep -v %s | awk '{print $2}'| xargs kill -9; \
                rm -rf %s;\
                wget -q https://xly-devops.bj.bcebos.com/PaddleTest/%s.tar.gz --no-proxy  >/dev/null ; \
                tar xf %s.tar.gz  >/dev/null 2>&1 ;cd %s""" % (
            pid,
            self.repo,
            self.repo,
            self.repo,
            self.repo,
        )
        #  cmd='''ps aux| grep python | grep -v main.py |grep -v %s | awk '{print $2}'| xargs kill -9; \
        #       rsync --delete-before -d /root/blank/ %s; rm -rf %s; \
        #       wget -q https://xly-devops.bj.bcebos.com/PaddleTest/%s.tar.gz --no-proxy  >/dev/null ; \
        #       tar xf %s.tar.gz  >/dev/null 2>&1 ;\
        #       cd %s''' % (pid, self.repo, self.repo, self.repo, self.repo, self.repo)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
        logging.info("git clone" + self.repo + "sucessfuly!")


# class RepoInit():
#       def __init__(self, repo):
#          self.repo=repo
#          print("This is Repo Init!")
#          pid = os.getpid()
#          cmd='''ps aux| grep python | grep -v main.py |grep -v %s | awk '{print $2}'| xargs kill -9; \
#           rm -rf %s; \
#           git clone https://gitee.com/paddlepaddle/%s.git; cd %s; \
#           python -m pip install -r requirements.txt -i \
#           https://pypi.tuna.tsinghua.edu.cn/simple''' % (pid, self.repo, self.repo, self.repo)
#          repo_result=subprocess.getstatusoutput(cmd)
#          exit_code=repo_result[0]
#          output=repo_result[1]
#          assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
#          logging.info("git clone"+self.repo+"sucessfuly!" )


class RepoInitSummer:
    """
    class
    """

    def __init__(self, repo):
        self.repo = repo
        print("This is Repo Init!")
        pid = os.getpid()
        cmd = """ps aux | grep python |grep -v main.py |grep -v %s | awk '{print $2}'| xargs kill -9; \
                rm -rf %s; \
                git clone https://gitee.com/summer243/%s.git; cd %s;\
                python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple""" % (
            pid,
            self.repo,
            self.repo,
            self.repo,
        )
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
        logging.info("git clone" + self.repo + "sucessfuly!")


class RepoInitCustom:
    """
    class
    """

    def __init__(self, repo):
        self.repo = repo
        print("This is Repo Init!")
        pid = os.getpid()
        cmd = """ps aux | grep python | grep -v main.py  |grep -v %s | awk '{print $2}'| xargs kill -9;\
                rm -rf %s; git clone https://gitee.com/paddlepaddle/%s.git;""" % (
            pid,
            self.repo,
            self.repo,
        )
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
        logging.info("git clone" + self.repo + "sucessfuly!")


class RepoInitCustom_video:
    """
    class
    """

    def __init__(self, repo):
        self.repo = repo
        print("This is Repo Init!")
        pid = os.getpid()
        cmd = """ps aux | grep python | grep -v main.py |grep -v %s | awk '{print $2}'| xargs kill -9; \
                rm -rf %s; git clone https://gitee.com/paddlepaddle/%s.git -b develop;""" % (
            pid,
            self.repo,
            self.repo,
        )
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "git clone %s failed!   log information:%s" % (self.repo, output)
        logging.info("git clone" + self.repo + "sucessfuly!")


class RepoRemove:
    """
    class
    """

    def __init__(self, repo):
        self.repo = repo
        print("This is Repo remove!")
        pid = os.getpid()
        cmd = """ps aux | grep python | grep -v main.py |grep -v %s | awk '{print $2}'| xargs kill -9; \
                rm -rf %s; rm -rf %s.tar.gz;""" % (
            pid,
            self.repo,
            self.repo,
        )
        #  cmd='''ps aux | grep python | grep -v main.py |grep -v %s | awk '{print $2}'/| xargs kill -9; \
        #       rsync --delete-before -d /root/blank/ %s; \
        #           rm -rf %s; rm -rf %s.tar.gz;'''% (pid, self.repo, self.repo, self.repo)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "remove %s failed!   log information:%s" % (self.repo, output)


class RepoDataset:
    """
    class
    """

    def __init__(self, cmd):
        self.cmd = cmd
        repo_result = subprocess.getstatusoutput(self.cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "configure failed!   log information:%s" % output
        logging.info("configure dataset sucessfuly!")


class CustomInstruction:
    """
    class
    """

    def __init__(self, cmd, model, mode):
        self.cmd = cmd
        self.model = model
        self.mode = mode
        repo_result = subprocess.getstatusoutput(self.cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        assert exit_code == 0, "%s of %s failed!   log information:%s" % (self.mode, self.model, output)


class TestClassModel:
    """
    class
    """

    def __init__(self, model, yaml):
        self.model = model
        self.yaml = yaml

    def test_class_train(self):
        """
        function
        """
        cmd = (
            'cd PaddleClas; export HIP_VISIBLE_DEVICES=0,1,2,3; \
                python -m paddle.distributed.launch --gpus=0,1,2,3 \
                tools/train.py  -c %s \
                -o DataLoader.Train.dataset.cls_label_path="./dataset/flowers102/train_list.txt" \
                -o DataLoader.Train.dataset.image_root="./dataset/flowers102/" \
                -o DataLoader.Eval.dataset.cls_label_path="./dataset/flowers102/val_list.txt" \
                -o DataLoader.Eval.dataset.image_root="./dataset/flowers102/" -o Global.epochs=2 \
                -o DataLoader.Train.sampler.batch_size=32 -o DataLoader.Eval.sampler.batch_size=32'
            % self.yaml
        )
        clas_result = subprocess.getstatusoutput(cmd)
        exit_code = clas_result[0]
        output = clas_result[1]
        exit_check_fucntion(exit_code, output, "train")

    def test_get_pretrained_model(self):
        """
        get pretrained model
        """
        legendary_models = ["ResNet50", "ResNet50_vd", "ResNet101", "MobileNetV3_large_x1_0", "VGG16"]
        if self.model in legendary_models:
            cmd = (
                "cd PaddleClas; \
        wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/%s_pretrained.pdparams"
                % self.model
            )
        else:
            cmd = (
                "cd PaddleClas; \
                    wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/%s_pretrained.pdparams"
                % self.model
            )
        clas_result = subprocess.getstatusoutput(cmd)
        exit_code = clas_result[0]
        output = clas_result[1]
        exit_check_fucntion(exit_code, output, "downlooad")

    def test_class_export_model(self):
        """
          class export model
          """
        cmd = (
            "cd PaddleClas;python tools/export_model.py -c %s -o Global.pretrained_model=./%s_pretrained \
                -o Global.save_inference_dir=./inference/%s"
            % (self.yaml, self.model, self.model)
        )
        clas_result = subprocess.getstatusoutput(cmd)
        exit_code = clas_result[0]
        output = clas_result[1]
        exit_check_fucntion(exit_code, output, "export_model")

    def test_class_predict(self, expect_id, expect_score):
        """
          class predict
          """
        cmd_gpu = (
            "cd PaddleClas; cd deploy; python python/predict_cls.py -c configs/inference_cls.yaml \
                -o Global.inference_model_dir=../inference/%s -o Global.batch_size=1 -o Global.use_gpu=True \
                    -o Global.use_tensorrt=False -o Global.enable_mkldnn=False"
            % self.model
        )
        cmd_cpu = (
            "cd PaddleClas; cd deploy; python python/predict_cls.py -c configs/inference_cls.yaml \
                -o Global.inference_model_dir=../inference/%s -o Global.batch_size=1 -o Global.use_gpu=False \
                -o Global.use_tensorrt=False -o Global.enable_mkldnn=False"
            % self.model
        )
        for cmd in [cmd_gpu, cmd_cpu]:
            clas_result = subprocess.getstatusoutput(cmd)
            exit_code = clas_result[0]
            output = clas_result[1]
            # check exit_code
            exit_check_fucntion(exit_code, output, "predict")

            # check class_id and class_score
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
            with assume:
                assert clas_score == approx(expect_score, abs=2e-2), (
                    "check class_score failed!   real class_score is: %s, expect class_score is: %s"
                    % (clas_score, expect_score)
                )
            print("*************************************************************************")


class TestOcrModel:
    """
    class
    """

    def __init__(self, model, yaml):
        self.model = model
        self.yaml = yaml

    def test_ocr_train(self):
        """
        function
        """
        cmd = (
            "cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; \
                sed -i s!data_lmdb_release/training!data_lmdb_release/validation!g %s; \
                python -m paddle.distributed.launch --gpus=0,1,2,3 --log_dir=log_%s  \
                tools/train.py -c %s \
                -o Global.use_gpu=True Global.epoch_num=1 Global.save_epoch_step=1 \
                Global.eval_batch_step=200 Global.print_batch_step=10 \
                Global.save_model_dir=output/%s Train.loader.batch_size_per_card=10 \
                Global.print_batch_step=1"
            % (self.yaml, self.model, self.yaml, self.model)
        )
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        log_dir = "PaddleOCR/log_" + self.model
        exit_check_fucntion(exit_code, output, "train", log_dir)

    def test_ocr_eval(self):
        """
        function
        """
        cmd = (
            "cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3;  python tools/eval.py -c %s  \
                -o Global.use_gpu=True Global.checkpoints=output/%s/latest"
            % (self.yaml, self.model)
        )
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "eval")

    def test_ocr_rec_infer(self):
        """
        function
        """
        cmd = (
            "cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer_rec.py -c %s  \
                -o Global.use_gpu=True Global.checkpoints=output/%s/latest \
                Global.infer_img=doc/imgs_words/en/word_1.png"
            % (self.yaml, self.model)
        )
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "infer")

    def test_ocr_export_model(self):
        """
        function
        """
        cmd = (
            "cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/export_model.py -c %s \
                -o Global.use_gpu=True Global.checkpoints=output/%s/latest  \
                Global.save_inference_dir=./models_inference/%s"
            % (self.yaml, self.model, self.model)
        )
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "export_model")

    def test_ocr_rec_predict(self):
        """
        function
        """
        #   cmd='cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer/predict_rec.py \
        #       --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"%s \
        #       --rec_image_shape="3, 32, 100" --rec_char_type="en" --rec_algorithm=CRNN' % (self.model)
        cmd = (
            'cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer/predict_rec.py \
                --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./models_inference/"%s \
                --rec_image_shape="3, 32, 100" --rec_algorithm=CRNN'
            % (self.model)
        )
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "predict")

    def test_ocr_det_infer(self):
        """
        function
        """
        cmd = (
            'cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer_det.py -c %s \
                -o Global.use_gpu=True Global.checkpoints=output/%s/latest Global.infer_img="./doc/imgs_en/" \
                Global.test_batch_size_per_card=1'
            % (self.yaml, self.model)
        )
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "infer")

    def test_ocr_det_predict(self):
        """
        function
        """
        cmd = (
            'cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer/predict_det.py \
                --image_dir="./doc/imgs_en/img_10.jpg" \
                --det_model_dir="./models_inference/"%s --det_algorithm=DB '
            % (self.model)
        )
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "predict")

    def test_ocr_e2e_infer(self):
        """
        function
        """
        cmd = (
            'cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer_e2e.py -c %s  \
            -o Global.use_gpu=True Global.checkpoints=output/%s/latest Global.infer_img="./doc/imgs_en/img_10.jpg"'
            % (self.yaml, self.model)
        )
        e2eection_result = subprocess.getstatusoutput(cmd)
        exit_code = e2eection_result[0]
        output = e2eection_result[1]
        exit_check_fucntion(exit_code, output, "infer")

    def test_ocr_e2e_predict(self):
        """
        function
        """
        #   cmd='cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer/predict_e2e.py \
        #       --image_dir="./doc/imgs_en/img623.jpg" --e2e_model_dir=./models_inference/%s --e2e_algorithm=PGNet \
        #       --e2e_pgnet_polygon=True --use_gpu=True' % (self.model)
        cmd = (
            'cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer/predict_e2e.py \
                --image_dir="./doc/imgs_en/img623.jpg" --e2e_model_dir=./models_inference/%s \
                --e2e_algorithm=PGNet --use_gpu=True'
            % (self.model)
        )
        e2eection_result = subprocess.getstatusoutput(cmd)
        exit_code = e2eection_result[0]
        output = e2eection_result[1]
        exit_check_fucntion(exit_code, output, "predict")

    def test_ocr_cls_infer(self):
        """
        function
        """
        cmd = (
            'cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer_cls.py -c %s  \
                -o Global.use_gpu=True \
                Global.checkpoints=output/%s/latest Global.infer_img="./doc/imgs_en/img_10.jpg"'
            % (self.yaml, self.model)
        )
        clsection_result = subprocess.getstatusoutput(cmd)
        exit_code = clsection_result[0]
        output = clsection_result[1]
        exit_check_fucntion(exit_code, output, "infer")

    def test_ocr_cls_predict(self):
        """
        function
        """
        cmd = (
            'cd PaddleOCR; export HIP_VISIBLE_DEVICES=0,1,2,3; python tools/infer/predict_cls.py \
                --image_dir="./doc/imgs_en/img623.jpg" --cls_model_dir=./models_inference/%s \
                --use_gpu=True'
            % (self.model)
        )
        clsection_result = subprocess.getstatusoutput(cmd)
        exit_code = clsection_result[0]
        output = clsection_result[1]
        exit_check_fucntion(exit_code, output, "predict")


class TestDetectionDygraphModel:
    """
    class
    """

    def __init__(self, model, yaml):
        self.model = model
        self.yaml = yaml

    def test_detection_train(self):
        """
        function
        """
        cmd = (
            "cd PaddleDetection; export HIP_VISIBLE_DEVICES=0,1,2,3; python -m paddle.distributed.launch \
                --gpus=0,1,2,3 --log_dir=log_%s tools/train.py -c %s -o TrainReader.batch_size=1 epoch=3"
            % (self.model, self.yaml)
        )
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        log_dir = "PaddleDetection/log_" + self.model
        exit_check_fucntion(exit_code, output, "train", log_dir)


class TestDetectionStaticModel:
    """
    class
    """

    def __init__(self, model, yaml):
        self.model = model
        self.yaml = yaml

    def test_detection_train(self):
        """
        function
        """
        cmd = (
            "cd PaddleDetection/static; export HIP_VISIBLE_DEVICES=0,1,2,3; \
                python -m paddle.distributed.launch \
                --gpus=0,1,2,3 --log_dir=log_%s tools/train.py -c %s \
                -o TrainReader.batch_size=1  -o max_iters=10"
            % (self.model, self.yaml)
        )
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        log_dir = "PaddleDetection/static/log_" + self.model
        exit_check_fucntion(exit_code, output, "train", log_dir)


class TestSegModel:
    """
    class
    """

    def __init__(self, model, yaml):
        self.model = model
        self.yaml = yaml

    def test_seg_train(self):
        """
        function
        """
        #   cmd='cd PaddleSeg; sed -i s/"iters: 80000"/"iters: 50"/g %s; \
        #           rsync --delete-before -d /root/blank/ log; \
        #       rm -rf log; export HIP_VISIBLE_DEVICES=0,1,2,3; \
        #           python -u -m paddle.distributed.launch --gpus="0,1,2,3" \
        #       --log_dir=log_%s train.py --config %s --do_eval --use_vdl \
        #           --num_workers 6 --save_dir log/%s \
        #       --save_interval 50 --iters 50' % (self.yaml, self.model, self.yaml, self.model)
        cmd = (
            'cd PaddleSeg; sed -i s/"iters: 80000"/"iters: 50"/g %s; \
                rm -rf log; export HIP_VISIBLE_DEVICES=0,1,2,3; \
                python -u -m paddle.distributed.launch --gpus="0,1,2,3" \
                --log_dir=log_%s train.py --config %s \
                --do_eval --use_vdl --num_workers 6 --save_dir log/%s \
                --save_interval 50 --iters 50'
            % (self.yaml, self.model, self.yaml, self.model)
        )
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        log_dir = "PaddleSeg/log_" + self.model
        exit_check_fucntion(exit_code, output, "train", log_dir)

    def test_seg_eval(self):
        """
        function
        """
        cmd = (
            "cd PaddleSeg; \
                wget -q https://bj.bcebos.com/paddleseg/dygraph/cityscapes/%s/model.pdparams; \
                mv model.pdparams %s.pdparams; export HIP_VISIBLE_DEVICES=0,1,2,3; \
                python -m paddle.distributed.launch val.py --config %s --model_path=%s.pdparams"
            % (self.model, self.model, self.yaml, self.model)
        )
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "eval")


class TestGanModel:
    """
    class
    """

    def __init__(self, model, yaml):
        self.model = model
        self.yaml = yaml

    def test_gan_train(self):
        """
        function
        """
        cmd = (
            'cd PaddleGAN; sed -i 1s/epochs/total_iters/ %s; export HIP_VISIBLE_DEVICES=0,1,2,3; \
                python -u -m paddle.distributed.launch --gpus="0,1,2,3" tools/main.py --config-file %s \
                -o total_iters=20 snapshot_config.interval=10 log_config.interval=1 output_dir=output'
            % (self.yaml, self.yaml)
        )
        print(cmd)
        gan_result = subprocess.getstatusoutput(cmd)
        exit_code = gan_result[0]
        output = gan_result[1]
        exit_check_fucntion(exit_code, output, "train")

    def test_gan_eval(self, cmd):
        """
        function
        """
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        exit_check_fucntion(exit_code, output, "eval")


class TestNlpModel:
    """
    class
    """

    def __init__(self, directory):
        self.directory = directory

    def test_nlp_train(self, cmd):
        """
        function
        """
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        exit_check_fucntion(exit_code, output, "train")


class TestParakeetModel:
    """
    class
    """

    def __init__(self, model):
        self.model = model

    def test_parakeet_train(self):
        """
        function
        """
        cmd = (
            'cd Parakeet/examples/%s; ln -s /data/ljspeech_%s ljspeech_%s; \
                export HIP_VISIBLE_DEVICES=0,1,2,3; \
                python train.py --data=ljspeech_%s --output=output --device="gpu" --nprocs=4 \
                --opts data.batch_size 2 training.max_iteration 10 \
                training.valid_interval 10 training.save_interval 10'
            % (self.model, self.model, self.model, self.model)
        )
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "train")


class TestRecModel:
    """
    class
    """

    def __init__(self, model, directory):
        self.model = model
        self.directory = directory

    def test_rec_train(self):
        """
        function
        """
        cmd = (
            'cd PaddleRec/%s; sed -i s/"use_gpu: False"/"use_gpu: True"/g config.yaml; \
                python -u ../../../tools/static_trainer.py -m config.yaml'
            % (self.directory)
        )
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "train")


class TestVideoModel:
    """
    class
    """

    def __init__(self, model, yaml):
        self.model = model
        self.yaml = yaml

    def test_video_train(self):
        """
        function
        """
        cmd = (
            'cd PaddleVideo; python -B -m paddle.distributed.launch --gpus="0,1,2,3" main.py  \
                --validate -c %s -o epochs=1'
            % (self.yaml)
        )
        print(cmd)
        detection_result = subprocess.getstatusoutput(cmd)
        exit_code = detection_result[0]
        output = detection_result[1]
        exit_check_fucntion(exit_code, output, "train")

    def test_video_eval(self, cmd):
        """
        function
        """
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        exit_code = repo_result[0]
        output = repo_result[1]
        exit_check_fucntion(exit_code, output, "train")
