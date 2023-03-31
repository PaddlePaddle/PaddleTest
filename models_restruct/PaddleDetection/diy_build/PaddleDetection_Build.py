# encoding: utf-8
"""
自定义环境准备
"""

import os
import sys
import logging
import tarfile
import argparse
import subprocess
import platform
import shutil
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleDetection_Build(Model_Build):
    """
    自定义环境准备
    """

    def __init__(self, args):
        """
        初始化变量
        """
        self.data_path_endswith = "dataset"
        self.paddle_whl = args.paddle_whl
        self.get_repo = args.get_repo
        self.branch = args.branch
        self.system = args.system
        self.set_cuda = args.set_cuda
        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame)  # 所有和yaml相关的变量与此拼接
        self.reponame = args.reponame
        self.dataset_org = args.dataset_org
        self.dataset_target = args.dataset_target
        self.mount_path = str(os.getenv("mount_path"))
        if ("Windows" in platform.system() or "Darwin" in platform.system()) and os.path.exists(
            self.mount_path
        ):  # linux 性能损失使用自动下载的数据,不使用mount数据
            logger.info("#### mount_path diy_build is {}".format(self.mount_path))
            if os.listdir(self.mount_path) != []:
                self.dataset_org = self.mount_path
                os.environ["dataset_org"] = self.mount_path
                self.dataset_target = os.path.join(os.getcwd(), self.reponame, self.data_path_endswith)
                os.environ["dataset_target"] = os.path.join(os.getcwd(), self.reponame, self.data_path_endswith)
        logger.info("#### dataset_org in diy_build is  {}".format(self.dataset_org))
        logger.info("#### dataset_target in diy_build is  {}".format(self.dataset_target))
        self.models_list = args.models_list
        self.models_file = args.models_file
        self.detection_model_list = []
        if str(self.models_list) != "None":
            for line in self.models_list.split(","):
                if ".yaml" in line:
                    self.detection_model_list.append(line.strip().replace("-", "/"))
        elif str(self.models_file) != "None":  # 获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                for line in open(file_name):
                    if ".yaml" in line:
                        self.detection_model_list.append(line.strip().replace("-", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" in file_name:
                    self.detection_model_list.append(file_name.strip().replace("-", "/"))

    def build_paddledetection(self):
        """
        安装依赖包
        """
        path_now = os.getcwd()
        os.chdir(self.reponame)
        path_repo = os.getcwd()
        os.system("python -m pip install --upgrade pip")
        os.system("python -m pip install Cython")
        logger.info("***start setuptools update")
        os.system("python -m pip uninstall setuptools -y")
        os.system("python -m pip install setuptools")
        os.system("python -m pip install -r requirements.txt")
        os.system("python -m pip install zip --ignore-installed")
        os.system("python -m pip uninstall paddleslim -y")
        os.system(
            "python -m pip install https://paddle-qa.bj.bcebos.com/PaddleSlim/paddleslim-0.0.0.dev0-py3-none-any.whl"
        )
        os.system("python -m pip install paddle2onnx")
        os.system("python -m pip install onnxruntime")
        logger.info("***start ffmpeg install***")
        os.system("rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro")
        os.system("rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm")
        os.system("yum install ffmpeg ffmpeg-devel -y")
        os.system("apt-get update")
        os.system("apt-get install ffmpeg -y")
        os.system("python -m pip uninstall bce-python-sdk -y")
        os.system("python -m pip install bce-python-sdk==0.8.74 --ignore-installed")
        # set sed
        if os.path.exists("C:/Program Files/Git/usr/bin/sed.exe"):
            os.environ["sed"] = "C:/Program Files/Git/usr/bin/sed.exe"
            cmd_weight = '{} -i "s#~/.cache/paddle/weights#dataset/det_pretrained#g" ppdet/utils/download.py'.format(
                os.getenv("sed")
            )
            subprocess.run(cmd_weight)
        else:
            os.environ["sed"] = "sed"
        # get video
        wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/test_demo.mp4")
        # avoid hang in yolox
        cmd = '{} -i "s|norm_type: sync_bn|norm_type: bn|g" configs/yolox/_base_/yolox_cspdarknet.yml'.format(
            os.getenv("sed")
        )
        if platform.system() == "Windows":
            subprocess.run(cmd)
        else:
            subprocess.run(cmd, shell=True)
        # use small data
        cmd_voc = '{} -i "s/trainval.txt/test.txt/g" configs/datasets/voc.yml'.format(os.getenv("sed"))
        if platform.system() == "Windows":
            subprocess.run(cmd_voc)
        else:
            subprocess.run(cmd_voc, shell=True)
        cmd_iter1 = (
            '{} -i "/for step_id, data in enumerate(self.loader):/i\\            max_step_id'
            '=1" ppdet/engine/trainer.py'.format(os.getenv("sed"))
        )
        cmd_iter2 = (
            '{} -i "/for step_id, data in enumerate(self.loader):/a\\                if step_id == '
            'max_step_id: break" ppdet/engine/trainer.py'.format(os.getenv("sed"))
        )
        if platform.system() == "Windows":
            subprocess.run(cmd_iter1)
            subprocess.run(cmd_iter2)
        else:
            subprocess.run(cmd_iter1, shell=True)
            subprocess.run(cmd_iter2, shell=True)
        # mot use small data
        cmd_mot1 = '{} -i "/for seq in seqs/for seq in [seqs[0]]/g" ppdet/engine/tracker.py'.format(os.getenv("sed"))
        cmd_mot2 = (
            '{} -i "/for step_id, data in enumerate(dataloader):/i\\        '
            'max_step_id=1" ppdet/engine/tracker.py'.format(os.getenv("sed"))
        )
        cmd_mot3 = (
            '{} -i "/for step_id, data in enumerate(dataloader):/a\\            if step_id == '
            'max_step_id: break" ppdet/engine/tracker.py'.format(os.getenv("sed"))
        )
        if platform.system() == "Windows":
            subprocess.run(cmd_mot1)
            subprocess.run(cmd_mot2)
            subprocess.run(cmd_mot3)
        else:
            subprocess.run(cmd_mot1, shell=True)
            subprocess.run(cmd_mot2, shell=True)
            subprocess.run(cmd_mot3, shell=True)
        # tiny_pose use coco data
        os.chdir(path_repo + "/configs/keypoint")
        if os.path.exists("tiny_pose"):
            shutil.rmtree("tiny_pose")
        wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/tiny_pose.zip")
        os.system("unzip -q tiny_pose.zip")
        os.chdir(path_repo)
        # compile op
        os.system("python ppdet/ext_op/setup.py install")
        if os.path.exists("/root/.cache/paddle/weights"):
            os.system("rm -rf /root/.cache/paddle/weights")
        os.system("ln -s {}/data/ppdet_pretrained /root/.cache/paddle/weights".format("/ssd2/ce_data/PaddleDetection"))
        # dataset download
        if platform.system() == "Linux":
            os.chdir("dataset")
            if os.path.exists("coco"):
                shutil.rmtree("coco")
            if os.path.exists("voc"):
                shutil.rmtree("voc")
            logger.info("***start download data")
            wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip")
            os.system("unzip -q coco.zip")
            wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/dota.zip")
            os.system("unzip -q dota.zip")
            wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/dota_ms.zip")
            os.system("unzip -q dota_ms.zip")
            wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/mot.zip")
            os.system("unzip -q mot.zip")
            wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/visdrone.zip")
            os.system("unzip -q visdrone.zip")
            wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/VisDrone2019_coco.zip")
            os.system("unzip -q VisDrone2019_coco.zip")
            # wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/mainbody.zip")
            # os.system("unzip mainbody.zip")
            wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/voc.zip")
            os.system("unzip -q voc.zip")
            # wget.download("https://paddle-qa.bj.bcebos.com/PaddleDetection/aic_coco_train_cocoformat.json")
            logger.info("***download data ended")
        else:
            if os.path.exists(self.dataset_target):
                shutil.rmtree(self.dataset_target)
            exit_code = os.symlink(self.dataset_org, self.dataset_target)
            if exit_code:
                logger.info("#### link_dataset failed")
        # compile cpp
        os.chdir(path_repo + "/deploy/cpp")
        wget.download(
            "https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-Centos"
            "-Gcc82-Cuda102-Cudnn76-Trt6018-Py38-Compile/latest/paddle_inference.tgz"
        )
        os.system("tar -xf paddle_inference.tgz")
        os.system('sed -i "s|WITH_GPU=OFF|WITH_GPU=ON|g" scripts/build.sh')
        os.system('sed -i "s|CUDA_LIB=/path/to/cuda/lib|CUDA_LIB=/usr/local/cuda/lib64|g" scripts/build.sh')
        os.system('sed -i "s|/path/to/paddle_inference|../paddle_inference|g" scripts/build.sh')
        os.system('sed -i "s|CUDNN_LIB=/path/to/cudnn/lib|CUDNN_LIB=/usr/lib/x86_64-linux-gnu|g" scripts/build.sh')
        os.system("sh scripts/build.sh")
        os.chdir(path_now)
        return 0

    def link_dataset(self):
        """
        软链数据
        """
        if os.path.exists(self.reponame):
            exit_code = os.symlink(self.dataset_org, self.dataset_target)
            if exit_code:
                logger.info("#### link_dataset failed")
        else:
            logger.info("check you {} path".format(self.reponame))
        return 0

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleDetection_Build, self).build_env()
        ret = 0
        ret = self.build_paddledetection()
        if ret:
            logger.info("build env failed")
            return ret
        return ret


if __name__ == "__main__":

    def parse_args():
        """
        接收和解析命令传入的参数
            最好尽可能减少输入给一些默认参数就能跑的示例!
        """
        parser = argparse.ArgumentParser("Tool for running CE task")
        parser.add_argument("--models_file", help="模型列表文件", type=str, default=None)
        parser.add_argument("--reponame", help="输入repo", type=str, default=None)
        args = parser.parse_args()
        return args

    args = parse_args()
    print("args:{}".format(args))
    # logger.info('###args {}'.format(args.models_file))
    model = PaddleDetection_Build(args)
    model.build_paddledetection()
