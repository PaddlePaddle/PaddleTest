# encoding: utf-8
"""
自定义环境准备
"""
import os
import sys
import time
import logging
import platform
import tarfile
import argparse
import numpy as np
import yaml
import wget
import shutil
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleMIX_Build(Model_Build):
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

        self.REPO_PATH = os.path.join(
            os.getcwd(), args.reponame)  # 所有和yaml相关的变量与此拼接
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
                        self.test_model_list.append(
                            line.strip().replace("^", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" or ".yml" in file_name:
                    self.test_model_list.append(
                        file_name.strip().replace("^", "/"))

    def build_paddlemix(self):
        """
        安装依赖包
        """
        path_now = os.getcwd()
        os.chdir("PaddleMIX")
        print("install paddlemix")
        os.system("python -m pip install --upgrade pip")
        os.system("pip install -r requirements.txt")
        os.system("pip install -e .")
        os.system("pip install -r paddlemix/appflow/requirements.txt")
        os.system(
            "pip install git+https://github.com/PaddlePaddle/PaddleSpeech.git")
        self.download_data(
            "https://paddle-qa.bj.bcebos.com/PaddleMIX/application.tar.gz")
        self.download_data("https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/ILSVRC2012/imagenet-val.tar",
                           destination_folder="/home", dir_name="data")
        self.download_data("https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/ILSVRC2012/ILSVRC2012_tiny.tar",
                           destination_folder="/home", dir_name="dataset")
        print("install ppdiffusers")
        import nltk
        nltk.download(
            ["punkt", "averaged_perceptron_tagger", "wordnet", "cmudict"])
        os.chdir("ppdiffusers")
        os.system("pip install -r requirements.txt")
        os.system("pip install -e .")
        self.download_data("https://paddle-qa.bj.bcebos.com/PaddleMIX/ppdiffusers_infer.tar.gz",
                           destination_folder="/home", dir_name="ppdiffusers_infer_file")
        for test_model_path in self.test_model_list:
            test_model_name = test_model_path.split("/")[-1]
            root_path = "/home/ppdiffusers_infer_file"
            target_path = os.path.join(path_now, "PaddleMIX", test_model_name)
            self.copy_files_with_prefix(
                root_path, target_path, test_model_name)
        os.chdir(path_now)
        return 0

    def download_data(self, data_link, destination_folder=".", dir_name=""):
        """
        下载数据集和使用的文件
        """
        tar_name = data_link.split("/")[-1]
        logger.info("start download {}".format(tar_name))
        # 下载文件
        file_name = wget.download(data_link, destination_folder)
        logger.info("start tar extract {}".format(tar_name))
        # 获取文件名和扩展名
        base_name, file_extension = os.path.splitext(file_name)
        if '.' in base_name:
            base_name, additional_extension = os.path.splitext(base_name)
            file_extension = additional_extension + file_extension
        if dir_name:
            base_name = dir_name
            destination_path = os.path.join(destination_folder, base_name)
            os.makedirs(destination_path, exist_ok=True)
        else:
            destination_path = destination_folder
        try:
            # 判断文件扩展名，选择相应的解压方式
            if file_extension == '.tar':
                with tarfile.open(file_name, 'r:') as tar:
                    tar.extractall(destination_path)
            elif file_extension == '.tar.gz':
                with tarfile.open(file_name, 'r:gz') as tar:
                    tar.extractall(destination_path)
            else:
                logger.info(f"不支持的文件格式: {file_name}")
                return

            logger.info(f"文件解压成功: {file_name}")
        except Exception as e:
            logger.info(f"文件解压失败: {file_name}, 错误信息: {str(e)}")
        finally:
            # 删除下载的文件
            time.sleep(5)
            os.remove(file_name)

    def copy_files_with_prefix(self, root_dir, target_dir, model_name, prefix="infer"):
        """
        复制指定目录下子目录内以指定前缀开头的的文件到指定目录
        """
        # 遍历根目录下的所有子文件夹
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                # 遍历子文件夹下的所有文件
                for filename in os.listdir(subdir_path):
                    if model_name == subdir:
                        # 检查文件是否以指定前缀开头
                        if filename.startswith(prefix):
                            source_path = os.path.join(subdir_path, filename)
                            target_path = os.path.join(target_dir, filename)
                            shutil.copy(source_path, target_path)
                            print(f"File {filename} copied to {target_dir}")

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleMIX_Build, self).build_env()
        ret = 0
        ret = self.build_paddlemix()
        if ret:
            logger.info("build env whl failed")
            return ret
        return ret


if __name__ == "__main__":

    def parse_args():
        """
        接收和解析命令传入的参数
            最好尽可能减少输入给一些默认参数就能跑的示例!
        """
        parser = argparse.ArgumentParser("Tool for running CE task")
        parser.add_argument("--models_file", help="模型列表文件",
                            type=str, default=None)
        parser.add_argument("--reponame", help="输入repo",
                            type=str, default=None)
        args = parser.parse_args()
        return args

    args = parse_args()
    model = PaddleMIX_Build(args)
    model.build_paddlemix()
