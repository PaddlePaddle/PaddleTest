# encoding: utf-8
"""
自定义环境准备
"""
import os
import sys
import shutil
import time
import logging
import tarfile
import argparse
import subprocess
import platform
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleOCR_Build(Model_Build):
    """
    自定义环境准备
    """

    def __init__(self, args):
        """
        初始化变量
        """
        self.paddle_whl = args.paddle_whl
        self.get_repo = args.get_repo
        self.branch = args.branch
        self.system = args.system
        self.set_cuda = args.set_cuda
        self.dataset_org = args.dataset_org
        self.dataset_target = args.dataset_target

        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame)  # 所有和yaml相关的变量与此拼接
        self.test_root_path = os.getcwd()

        self.reponame = args.reponame
        self.models_list = args.models_list
        self.models_file = args.models_file
        self.test_model_list = []
        self.mount_path = str(os.getenv("mount_path"))
        self.use_data_cfs = str(args.use_data_cfs)

        if str(self.models_list) != "None":
            for line in self.models_list.split(","):
                if ".yaml" or ".yml" in line:
                    self.test_model_list.append(line.strip().replace("^", "/"))
                    print("self.test_model_list:{}".format(self.test_model_list))
        elif str(self.models_file) != "None":  # 获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                for line in open(file_name):
                    if ".yaml" or ".yml" in line:
                        self.test_model_list.append(line.strip().replace("^", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" or ".yml" in file_name:
                    self.test_model_list.append(file_name.strip().replace("^", "/"))

    def build_dataset(self):
        """
        make datalink
        """
        if os.path.exists(self.reponame):
            os.chdir(self.reponame)

            sysstr = platform.system()
            if sysstr == "Linux":
                if os.path.exists(self.mount_path) and self.use_data_cfs == "True":
                    src_path = self.mount_path
                else:
                    if os.path.exists("/ssd2/ce_data/PaddleOCR"):
                        src_path = "/ssd2/ce_data/PaddleOCR"
                    else:
                        src_path = "/home/data/cfs/models_ce/PaddleOCR"
            elif sysstr == "Windows":
                # src_path = "F:\\PaddleOCR"
                # os.system("mklink /d train_data F:\\PaddleOCR\\train_data")
                # os.system("mklink /d pretrain_models F:\\PaddleOCR\\pretrain_models")
                src_path = "H:\\MT_data\\PaddleOCR"
            elif sysstr == "Darwin":
                # src_path = "/Users/paddle/PaddleTest/ce_data/PaddleOCR"
                src_path = "/Volumes/210-share-data/MT_data/PaddleOCR"
            print("PaddleOCR dataset path:{}".format(src_path))
            # dataset link
            # train_data_path = os.path.join(src_path, "train_data")
            # pretrain_models_path = os.path.join(src_path, "pretrain_models")

            # if not os.path.exists(train_data_path):
            #    os.makedirs(train_data_path)
            # if not os.path.exists(pretrain_models_path):
            #    os.makedirs(pretrain_models_path)
            # if sysstr != "Windows":
            if not os.path.exists("train_data"):
                os.symlink(os.path.join(src_path, "train_data"), "train_data")
            if not os.path.exists("pretrain_models"):
                os.symlink(os.path.join(src_path, "pretrain_models"), "pretrain_models")
            if not os.path.exists("train_data"):
                print("train_data not exists!")
                #                sys.exit(1)

                # dataset
                """
                if not os.path.exists("train_data/ctw1500"):
                    self.download_data("https://paddle-qa.bj.bcebos.com/PaddleOCR/train_data/ctw1500.tar", "train_data")

                if not os.path.exists("train_data/icdar2015"):
                    self.download_data(
                        "https://paddle-qa.bj.bcebos.com/PaddleOCR/train_data/icdar2015.tar", "train_data"
                    )
                if not os.path.exists("train_data/data_lmdb_release"):
                    self.download_data(
                        "https://paddle-qa.bj.bcebos.com/PaddleOCR/train_data/data_lmdb_release.tar", "train_data"
                    )
                """
            # configs/rec/rec_resnet_stn_bilstm_att.yml
            # os.system("python -m pip install fasttext")
            # if not os.path.exists("cc.en.300.bin"):
            #    self.download_data(
            #        "https://paddle-qa.bj.bcebos.com/PaddleOCR/pretrain_models/cc.en.300.bin.tar", os.getcwd()
            #    )

            # kie requirements
            os.system("python -m pip install -U  paddlenlp")
            os.system("python -m pip install -r ppstructure/kie/requirements.txt")
            # mac: Downgrade the protobuf package to 3.20.x or lower.
            os.system("python -m pip install -U protobuf==3.20.0")

            if sysstr == "Windows":
                os.environ["PATH"] = "F:\\install\\GnuWin32\\bin;" + os.environ.get("PATH")

            for filename in self.test_model_list:
                print("filename:{}".format(filename))
                if "rec" in filename:
                    if sysstr == "Darwin":
                        cmd = "sed -i '' 's!data_lmdb_release/training!data_lmdb_release/validation!g' %s" % filename
                    else:
                        cmd = "sed -i s!data_lmdb_release/training!data_lmdb_release/validation!g %s" % filename

                    os.system(cmd)

                if "e2e_r50_vd_pg" in filename:
                    if sysstr == "Darwin":
                        cmd = "sed -i '' 's/batch_size: 14/batch_size: 1/g' %s" % filename
                    else:
                        cmd = """sed -i "s/batch_size: 14/batch_size: 1/g" %s""" % filename
                    os.system(cmd)
            if sysstr == "Linux":
                # dygraph2static_dataset
                os.chdir("benchmark/PaddleOCR_DBNet")
                self.download_data("https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/benchmark_train/datasets.tar")
                os.system("python -m pip install -r requirement.txt")
                for filename in self.test_model_list:
                    print("filename:{}".format(filename))
                    if "benchmark" in filename:
                        os.system("python -m pip install -U numpy==1.23.5")
                        os.system("python -m pip install Polygon3")

            os.chdir(self.test_root_path)
            print("build dataset!")

            if platform.machine() == "arm64":
                print("mac M1")
                os.system("conda install -y scikit-image")
                os.system("conda install -y imgaug")

    def download_data(self, data_link, destination="."):
        """
        下载数据集
        """
        tar_name = data_link.split("/")[-1]
        logger.info("start download {}".format(tar_name))
        wget.download(data_link, destination)
        logger.info("start tar extract {}".format(tar_name))
        tf = tarfile.open(os.path.join(destination, tar_name))
        tf.extractall(destination)
        time.sleep(10)
        os.remove(os.path.join(destination, tar_name))

    def compile_c_predict_demo(self):
        """
        compile_c_predict_demo
        """
        print(os.getcwd())
        os.chdir("PaddleOCR/deploy/cpp_infer")

        OPENCV_DIR = os.environ.get("OPENCV_DIR")
        LIB_DIR = os.environ.get("paddle_inference_LIB_DIR")
        CUDA_LIB_DIR = os.environ.get("CUDA_LIB_DI")
        CUDNN_LIB_DIR = os.environ.get("CUDNN_LIB_DIR")
        TENSORRT_DIR = os.environ.get("TENSORRT_DIR")

        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build")
        os.chdir("build")
        print(os.getcwd())
        cmd = (
            "cmake .. -DPADDLE_LIB=%s -DWITH_MKL=ON -DWITH_GPU=OFF -DWITH_STATIC_LIB=OFF -DWITH_TENSORRT=OFF \
    -DOPENCV_DIR=%s -DCUDNN_LIB=%s -DCUDA_LIB=%s -DTENSORRT_DIR=%s"
            % (LIB_DIR, OPENCV_DIR, CUDNN_LIB_DIR, CUDA_LIB_DIR, TENSORRT_DIR)
        )
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        # exit_code = repo_result[0]
        output = repo_result[1]
        print(output)
        os.system("make -j")
        os.chdir(self.test_root_path)

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleOCR_Build, self).build_env()
        ret = 0
        ret = self.build_dataset()
        if ret:
            logger.info("build env dataset failed")
            return ret

        sysstr = platform.system()
        if sysstr == "Linux" and os.environ.get("c_plus_plus_predict") == "True":
            self.compile_c_predict_demo()

        return ret
