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
                if os.path.exists("/ssd2/ce_data/PaddleOCR"):
                    src_path = "/ssd2/ce_data/PaddleOCR"
                else:
                    src_path = "/home/data/cfs/models_ce/PaddleOCR"
            elif sysstr == "Windows":
                # src_path = "F:\\PaddleOCR"
                # os.system("mklink /d train_data F:\\PaddleOCR\\train_data")
                # os.system("mklink /d pretrain_models F:\\PaddleOCR\\pretrain_models")
                src_path="H:\MT_data\PaddleOCR"
            elif sysstr == "Darwin":
                # src_path = "/Users/paddle/PaddleTest/ce_data/PaddleOCR"
                src_path ="/Volumes/210-share-data/MT_data/PaddleOCR"
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
            os.chdir(self.test_root_path)
            print("build dataset!")

            if platform.machine() == "arm64":
                print("mac M1")
                os.system("conda install -y scikit-image")
                os.system("conda install -y imgaug")

    def download_data(self, data_link, destination):
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

    def prepare_opencv(self):
        """
        prepare_opencv
        """

        print(os.getcwd())
        os.chdir("PaddleOCR/deploy/cpp_infer")
        # os.chdir('deploy/cpp_infer')

        # download opencv source code
        wget.download("https://paddleocr.bj.bcebos.com/libs/opencv/opencv-3.4.7.tar.gz")
        tf = tarfile.open("opencv-3.4.7.tar.gz")
        tf.extractall(os.getcwd())

        os.chdir("opencv-3.4.7")
        root_path = os.getcwd()
        install_path = os.path.join(root_path, "opencv3")

        # build
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build")
        os.chdir("build")

        # cmake
        print(os.getcwd())
        cmd = (
            "cmake .. -DCMAKE_INSTALL_PREFIX=%s \
    -DCMAKE_BUILD_TYPE=Release -DWITH_IPP=OFF -DBUILD_IPP_IW=OFF-DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF -DCMAKE_INSTALL_LIBDIR=lib64 -DWITH_ZLIB=ON -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON -DWITH_TIFF=ON -DBUILD_TIFF=ON"
            % (install_path)
        )
        repo_result = subprocess.getstatusoutput(cmd)
        # exit_code = repo_result[0]
        output = repo_result[1]
        print(output)
        # make
        os.system("make -j")
        # make install
        os.system("make install")
        os.chdir(self.test_root_path)

    def prepare_c_predict_library(self):
        """
        prepare_c_predict_library
        """
        print(os.getcwd())
        os.chdir("PaddleOCR/deploy/cpp_infer")
        wget.download(
            "https://paddle-inference-lib.bj.bcebos.com/2.3.2/cxx_c/Linux/GPU/\
x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddle_inference.tgz"
        )
        tf = tarfile.open("paddle_inference.tgz")
        tf.extractall(os.getcwd())
        os.chdir(self.test_root_path)

    def compile_c_predict_demo(self):
        """
        compile_c_predict_demo
        """
        print(os.getcwd())
        os.chdir("PaddleOCR/deploy/cpp_infer")
        # os.chdir('deploy/cpp_infer')
        root_path = os.getcwd()
        OPENCV_DIR = os.path.join(root_path, "opencv-3.4.7/opencv3")
        LIB_DIR = os.path.join(root_path, "paddle_inference")
        CUDA_LIB_DIR = "/usr/local/cuda/lib64"
        CUDNN_LIB_DIR = "/usr/lib/x86_64-linux-gnu/"
        TENSORRT_DIR = "/usr/local/TensorRT-6.0.1.8/"
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

        """
        sysstr = platform.system()
        if sysstr == "Linux":
            self.prepare_opencv()
            self.prepare_c_predict_library()
            self.compile_c_predict_demo()
        """

        return ret
