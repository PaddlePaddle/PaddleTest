# encoding: utf-8
"""
自定义环境准备
"""
import os
import sys
import logging
import tarfile
import argparse
import shutil
import subprocess
import platform
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class Paddle3D_Build(Model_Build):
    """
    自定义环境准备
    """

    def __init__(self, args):
        """
        初始化变量
        """
        super(Paddle3D_Build, self).__init__(args)
        # self.paddle_whl = args.paddle_whl
        # self.branch = args.branch
        # self.system = args.system
        # self.set_cuda = args.set_cuda
        # self.dataset_org = args.dataset_org
        # self.dataset_target = args.dataset_target
        # self.reponame = args.reponame
        # self.get_repo = args.get_repo

        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame)  # 所有和yaml相关的变量与此拼接

        print("self.reponame:{}".format(self.reponame))
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
            path_now = os.getcwd()
            os.chdir(self.reponame)

            sysstr = platform.system()
            if sysstr == "Linux":
                if os.path.exists(self.mount_path) and self.use_data_cfs == "True":
                    src_path = self.mount_path
                else:
                    if os.path.exists("/ssd2/ce_data/Paddle3D"):
                        src_path = "/ssd2/ce_data/Paddle3D"
                    else:
                        src_path = "/home/data/cfs/models_ce/Paddle3D"
            elif sysstr == "Windows":
                src_path = "F:\\ce_data\\Paddle3D"
            elif sysstr == "Darwin":
                src_path = "/Users/paddle/PaddleTest/ce_data/Paddle3D"

            if not os.path.exists("datasets"):
                os.symlink(src_path, "datasets")
            print("build dataset!")
            # os.system("apt-get update")
            # os.system("apt-get install -y python3-setuptools")
            os.system("python -m pip install -U scikit-learn")
            # os.system("python -m pip install -U nuscenes-devkit")
            # linux-python3.10
            # os.system("python -m pip install -U pip setuptools")
            # os.system("python -m pip install numba")
            # os.system("cat requirements.txt | xargs -n 1 pip install -i https://mirror.baidu.com/pypi/simple")
            # paddle3d
            sysstr = platform.system()
            if sysstr == "Linux":
                # linux：xx are installed in '/root/.local/bin' which is not on PATH
                os.environ["PATH"] += os.pathsep + "/root/.local/bin"
                if not os.path.exists("/root/.paddle3d/pretrained/dla34/"):
                    os.makedirs("/root/.paddle3d/pretrained/dla34/")
                if os.path.exists("/root/.paddle3d/pretrained/dla34/dla34.pdparams"):
                    os.remove("/root/.paddle3d/pretrained/dla34/dla34.pdparams")
                wget.download(
                    "https://bj.bcebos.com/paddle3d/pretrained/dla34.pdparams", out="/root/.paddle3d/pretrained/dla34/"
                )

            os.system("python -m pip install . ")

            print("build wheel!")

            # petr
            if not os.path.exists("data"):
                os.makedirs("data")
                os.symlink(os.path.join(src_path, "nuscenes_petr"), "data/nuscenes")
                os.makedirs("/workspace/datset/nuScenes/", exist_ok=True)
                os.symlink(os.path.join(src_path, "nuscenes_petr"), "/workspace/datset/nuScenes/nuscenes")

                os.symlink(os.path.join(src_path, "kitti"), "data/kitti")

            for filename in self.test_model_list:
                print("filename:{}".format(filename))
                cmd = 'sed -i "/iters/d;1i\\iters: 200" %s' % (filename)
                subprocess.getstatusoutput(cmd)
                # cmd = "cat %s" % (filename)
                # subprocess.getstatusoutput(cmd)
                # cmd = 'sed -i "s!data/kitti!datasets/kitti!g" %s' % (filename)
                # subprocess.getstatusoutput(cmd)
            print("change iters number!")
            os.chdir(path_now)

    def compile_c_predict_demo(self):
        """
        compile_c_predict_demo
        """
        print(os.getcwd())

        OPENCV_DIR = os.environ.get("OPENCV_DIR")
        LIB_DIR = os.environ.get("paddle_inference_LIB_DIR")
        CUDA_LIB_DIR = os.environ.get("CUDA_LIB_DI")
        CUDNN_LIB_DIR = os.environ.get("CUDNN_LIB_DIR")
        TENSORRT_DIR = os.environ.get("TENSORRT_DIR")

        os.chdir("Paddle3D/deploy/smoke/cpp")
        # paddle_inference
        os.chdir("lib")
        if os.path.exists("paddle_inference"):
            os.unlink("paddle_inference")
        os.symlink(LIB_DIR, "paddle_inference")
        os.chdir("..")
        # smoke compile
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build")
        os.chdir("build")
        print(os.getcwd())

        cmd = (
            "export OpenCV_DIR=%s; cmake .. -DPADDLE_LIB=%s -DWITH_MKL=ON -DDEMO_NAME=infer -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF -DUSE_TENSORRT=OFF  -DWITH_ROCM=OFF -DROCM_LIB=/opt/rocm/lib \
    -DCUDNN_LIB=%s -DCUDA_LIB=%s -DTENSORRT_ROOT=%s"
            % (OPENCV_DIR, LIB_DIR, CUDNN_LIB_DIR, CUDA_LIB_DIR, TENSORRT_DIR)
        )
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        # exit_code = repo_result[0]
        output = repo_result[1]
        print(output)
        os.system("make -j")

        # pointpillars compile
        os.chdir(self.test_root_path)
        os.chdir("Paddle3D/deploy/pointpillars/cpp")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build")
        os.chdir("build")
        print(os.getcwd())

        cmd = (
            "cmake .. -DPADDLE_LIB=%s -DWITH_MKL=ON -DDEMO_NAME=main -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF -DUSE_TENSORRT=OFF  -DWITH_ROCM=OFF \
    -DROCM_LIB=/opt/rocm/lib -DCUDNN_LIB=%s -DCUDA_LIB=%s -DTENSORRT_ROOT=%s \
    -DCUSTOM_OPERATOR_FILES='custom_ops/iou3d_cpu.cpp;custom_ops/\
    iou3d_nms_api.cpp;custom_ops/iou3d_nms.cpp;custom_ops/iou3d_nms_kernel.cu'"
            % (LIB_DIR, CUDNN_LIB_DIR, CUDA_LIB_DIR, TENSORRT_DIR)
        )
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        # exit_code = repo_result[0]
        output = repo_result[1]
        print(output)
        os.system("make -j")
        os.chdir(self.test_root_path)

        # centerpoint compile
        os.chdir(self.test_root_path)
        os.chdir("Paddle3D/deploy/centerpoint/cpp")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build")
        os.chdir("build")
        print(os.getcwd())

        cmd = (
            "cmake .. -DPADDLE_LIB=%s -DWITH_MKL=ON -DDEMO_NAME=main -DWITH_GPU=OFF  -DWITH_STATIC_LIB=OFF \
    -DUSE_TENSORRT=OFF -DWITH_ROCM=OFF -DROCM_LIB=/opt/rocm/lib -DCUDNN_LIB=%s -DCUDA_LIB=%s -DTENSORRT_ROOT=%s \
    -DCUSTOM_OPERATOR_FILES='custom_ops/voxelize_op.cu;custom_ops/voxelize_op.cc;\
    custom_ops/iou3d_nms_kernel.cu;custom_ops/postprocess.cc;custom_ops/postprocess.cu'"
            % (LIB_DIR, CUDNN_LIB_DIR, CUDA_LIB_DIR, TENSORRT_DIR)
        )
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        # exit_code = repo_result[0]
        output = repo_result[1]
        print(output)
        os.system("make -j")
        os.chdir(self.test_root_path)

        # petr compile
        os.chdir("Paddle3D/deploy/petr/cpp")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build")
        os.chdir("build")
        print(os.getcwd())

        cmd = (
            "cmake .. -DOPENCV_DIR=%s -DPADDLE_LIB=%s -DWITH_MKL=ON -DDEMO_NAME=main \
    -DWITH_GPU=OFF -DWITH_STATIC_LIB=OFF -DUSE_TENSORRT=OFF \
    -DWITH_ROCM=OFF -DROCM_LIB=/opt/rocm/lib -DCUDNN_LIB=%s \
    -DCUDA_LIB=%s -DTENSORRT_ROOT=%s -DCUSTOM_OPERATOR_FILES=''"
            % (OPENCV_DIR, LIB_DIR, CUDNN_LIB_DIR, CUDA_LIB_DIR, TENSORRT_DIR)
        )
        print(cmd)
        repo_result = subprocess.getstatusoutput(cmd)
        # exit_code = repo_result[0]
        output = repo_result[1]
        print(output)
        os.system("make -j")
        os.chdir(self.test_root_path)

        # caddn compile
        os.chdir("Paddle3D/deploy/caddn/cpp")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build")
        os.chdir("build")
        print(os.getcwd())

        cmd = (
            "cmake .. -DOPENCV_DIR=%s -DPADDLE_LIB=%s -DWITH_MKL=ON \
    -DDEMO_NAME=main -DWITH_GPU=OFF -DWITH_STATIC_LIB=OFF -DUSE_TENSORRT=OFF \
    -DWITH_ROCM=OFF -DROCM_LIB=/opt/rocm/lib -DCUDNN_LIB=%s -DCUDA_LIB=%s -DTENSORRT_ROOT=%s \
    -DCUSTOM_OPERATOR_FILES='custom_ops/iou3d_nms.cpp;custom_ops/iou3d_nms_api.cpp;custom_ops/iou3d_nms_kernel.cu'"
            % (OPENCV_DIR, LIB_DIR, CUDNN_LIB_DIR, CUDA_LIB_DIR, TENSORRT_DIR)
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
        super(Paddle3D_Build, self).build_env()
        ret = 0
        ret = self.build_dataset()
        if ret:
            logger.info("build env dataset failed")
            return ret
        sysstr = platform.system()
        if sysstr == "Linux" and os.environ.get("c_plus_plus_predict") == "True":
            self.compile_c_predict_demo()
        return ret
