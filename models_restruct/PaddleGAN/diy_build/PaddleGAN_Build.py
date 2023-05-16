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
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleGAN_Build(Model_Build):
    """
    自定义环境准备
    """

    def __init__(self, args):
        """
        初始化变量
        """
        self.data_path_endswith = "data"
        self.reponame = args.reponame
        self.paddle_whl = args.paddle_whl
        self.get_repo = args.get_repo
        self.branch = args.branch
        self.system = args.system
        self.set_cuda = args.set_cuda

        self.dataset_org = str(args.dataset_org)
        self.dataset_target = str(args.dataset_target)
        self.mount_path = str(os.getenv("mount_path"))
        if ("Windows" in platform.system() or "Darwin" in platform.system()) and os.path.exists(
            self.mount_path
        ):  # linux 性能损失使用自动下载的数据,不使用mount数据
            if os.listdir(self.mount_path) != []:
                self.dataset_org = self.mount_path
                os.environ["dataset_org"] = self.mount_path
                self.dataset_target = os.path.join(os.getcwd(), self.reponame, self.data_path_endswith)
                os.environ["dataset_target"] = os.path.join(os.getcwd(), self.reponame, self.data_path_endswith)
        logger.info("#### dataset_org in diy_build is  {}".format(self.dataset_org))
        logger.info("#### dataset_target in diy_build is  {}".format(self.dataset_target))

        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame)  # 所有和yaml相关的变量与此拼接
        self.models_list = args.models_list
        self.models_file = args.models_file
        self.gan_model_list = []
        if str(self.models_list) != "None":
            for line in self.models_list.split(","):
                if ".yaml" in line:
                    self.gan_model_list.append(line.strip().replace("^", "/"))
        elif str(self.models_file) != "None":  # 获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                for line in open(file_name):
                    if ".yaml" in line:
                        self.gan_model_list.append(line.strip().replace("^", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" in file_name:
                    self.gan_model_list.append(file_name.strip().replace("^", "/"))

    def value_in_modellist(self, value=None):
        """
        判断字段是否存在model_list
        """
        for line in self.gan_model_list:
            if value in line:
                return 1
        return 0

    def download_data(self, value=None):
        """
        下载数据集
        """
        # 调用函数路径已切换至PaddleGAN
        tar_name = value.split("/")[-1]
        # if os.path.exists(tar_name) and os.path.exists(tar_name.replace(".tar", "")):
        # 有end回收数据，只判断文件夹
        if os.path.exists(tar_name.replace(".tar", "")):
            logger.info("#### already download {}".format(tar_name))
        else:
            try:
                logger.info("#### start download {}".format(tar_name))
                wget.download(value.replace(" ", ""))
                logger.info("#### end download {}".format(tar_name))
                tf = tarfile.open(tar_name)
                tf.extractall(os.getcwd())
            except:
                logger.info("#### prepare download failed {} failed".format(tar_name))
        return 0

    def get_image_name(self, value=None, label=None):
        """
        获取数据集名称
        """
        with open(os.path.join(self.REPO_PATH, value), "r", encoding="utf-8") as y:
            cfg = yaml.full_load(y)
            # print('#### cfg', cfg.keys())
            if label in cfg["dataset"].keys():
                # print('#### dataset', cfg["dataset"][label].keys())
                if "dataroot_a" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["dataroot_a"]
                elif "dataroot_b" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["dataroot_b"]
                elif "gt_folder" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["gt_folder"]
                elif "lq_folder" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["lq_folder"]
                elif "dataroot" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["dataroot"]
                elif "dataset" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["dataset"]
                elif "rgb_dir" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["rgb_dir"]
                elif "dataset_name" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["dataset_name"]
                elif "content_root" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["content_root"]
                elif "opt" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["opt"]
                elif "dataroot_gt" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["dataroot_gt"]
                elif "dataset_path" in cfg["dataset"][label].keys():
                    image_name = cfg["dataset"][label]["dataset_path"]
                else:
                    return None

                if isinstance(image_name, dict):
                    if "gt_folder" in image_name.keys():
                        image_name = image_name["gt_folder"]
                    elif "lq_folder" in image_name.keys():
                        image_name = image_name["lq_folder"]
                    elif "dataroot_H" in image_name.keys():
                        image_name = image_name["dataroot_H"]
                    elif "train_dir" in image_name.keys():
                        image_name = image_name["train_dir"]
                    elif "val_dir" in image_name.keys():
                        image_name = image_name["val_dir"]
                    # print('@@@@image_name', image_name)

                # print('####image_name: {}'.format(image_name))
                if "data/" in image_name and "_data" not in image_name:
                    image_name = image_name.split("data/")[1]
                elif "data/" in image_name and "_data" in image_name:
                    image_name = image_name.split("data/")[1] + "data"
                elif "test/" in image_name:
                    image_name = image_name.split("test/")[1]
                else:
                    return None
                # print('####image_name: {}'.format(image_name))
                image_name = image_name.split("/")[0]
                # print('####image_name: {}'.format(image_name))
                image_name = image_name.replace('"', "")
                # print('####image_name: {}'.format(image_name))
                return image_name
            else:
                # print('#### dataset', cfg["dataset"][label].keys())
                return None

    def change_yaml_batch_size(self, data_json):
        """
        递归使所有batch_size,存在即为1
        """
        if isinstance(data_json, dict):
            for key, val in data_json.items():
                if key == "batch_size" and "@" not in str(val):
                    data_json[key] = "1" + "  #@"
                if isinstance(data_json[key], dict):
                    self.change_yaml_batch_size(data_json[key])
        return data_json

    def build_yaml(self):
        """
        更改RD原始的yaml文件
        """
        if os.path.exists(self.reponame):
            for line in self.gan_model_list:
                with open(os.path.join(self.REPO_PATH, line), "r") as f:
                    content = yaml.load(f, Loader=yaml.FullLoader)

                # 改变特殊字符
                if "pretrain_ckpt" in str(content):
                    content["model"]["pretrain_ckpt"] = None
                    logger.info("#### change animeganv2 pretrain_ckpt")
                if "max_eval_steps" in str(content):
                    content["model"]["max_eval_steps"] = "100 #@"
                    logger.info("#### change stylegan_v2_256_ffhq wav2lip max_eval_steps")
                if "epochs" in content.keys():
                    content["total_iters"] = content.pop("epochs")
                    logger.info("#### change epochs to total_iters")
                # 改变 batch_size
                content_new = self.change_yaml_batch_size(content)  # 写在with里面不能够全部生效
                with open(os.path.join(self.REPO_PATH, line), "w") as f:
                    yaml.dump(content_new, f, sort_keys=False)
        else:
            logger.info("check you {} path".format(self.reponame))
        return 0

    def build_infer_dataset(self):
        """
        下载预测相关数据集
        """
        if os.path.exists(self.reponame):
            # 下载dataset数据集 解析yaml下载
            path_now = os.getcwd()
            os.chdir(self.reponame)
            if os.path.exists(self.data_path_endswith) is False:
                os.mkdir(self.data_path_endswith)
            os.chdir(self.data_path_endswith)

            # 下载特殊数据
            logger.info("#### start download other data")
            if os.path.exists("Peking_input360p_clip_10_11.mp4") is False:
                wget.download("https://paddle-qa.bj.bcebos.com/PaddleGAN/ce_data/Peking_input360p_clip_10_11.mp4")
            if os.path.exists("starrynew.png") is False:
                wget.download("https://paddle-qa.bj.bcebos.com/PaddleGAN/ce_data/starrynew.png")
            logger.info("#### end download other data")
            os.chdir(path_now)
        else:
            logger.info("check you {} path".format(self.reponame))
        return 0

    def link_dataset(self):
        """
        软链数据
        """
        if os.path.exists(self.reponame):
            if os.path.exists(os.path.join(self.reponame, self.dataset_target)):
                logger.info(
                    "already have {} will mv {} to {}".format(
                        self.dataset_target, self.dataset_target, self.dataset_target + "_" + str(int(time.time()))
                    )
                )
                os.rename(
                    os.path.join(self.reponame, self.dataset_target),
                    os.path.join(self.reponame, self.dataset_target + "_" + str(int(time.time()))),
                )
            exit_code = os.symlink(self.dataset_org, os.path.join(self.reponame, self.dataset_target))
            if exit_code:
                logger.info("#### link_dataset failed")
        else:
            logger.info("check you {} path".format(self.reponame))
        return 0

    def build_dataset(self):
        """
        自定义下载数据集
        """
        if os.path.exists(self.reponame):
            # 下载dataset数据集 解析yaml下载
            path_now = os.getcwd()
            os.chdir(self.reponame)
            if os.path.exists(self.data_path_endswith) is False:
                os.mkdir(self.data_path_endswith)
            os.chdir(self.data_path_endswith)

            # 下载特殊数据
            logger.info("#### start download other data")
            if os.path.exists("Peking_input360p_clip_10_11.mp4") is False:
                wget.download("https://paddle-qa.bj.bcebos.com/PaddleGAN/ce_data/Peking_input360p_clip_10_11.mp4")
            if os.path.exists("starrynew.png") is False:
                wget.download("https://paddle-qa.bj.bcebos.com/PaddleGAN/ce_data/starrynew.png")
            logger.info("#### end download other data")

            # 根据不同的模型下载数据
            for line in self.gan_model_list:
                image_name = None  # 初始化
                for train_test in ["train", "test"]:
                    image_name = self.get_image_name(value=line, label=train_test)
                    if image_name is None:
                        logger.info("do not need download")
                    else:
                        self.download_data(
                            value="https://paddle-qa.bj.bcebos.com\
                            /{}/ce_data/{}.tar".format(
                                self.reponame, image_name
                            )
                        )
            os.chdir(path_now)
        else:
            logger.info("check you {} path".format(self.reponame))
        return 0

    def build_paddlegan(self):
        """
        安装依赖包
        """
        # 固定随机量需要，默认打开
        logger.info("#### or AGILE_PIPELINE_NAME is {}".format(os.getenv("AGILE_PIPELINE_NAME")))
        # 230505 for wanghuan
        if str(os.getenv("AGILE_PIPELINE_NAME")) == "PaddleGAN-Linux-Cuda102-Python37-ALL-Release-test3":
            os.environ["FLAGS_use_stride_kernel"] = "1"
        logger.info("#### set FLAGS_use_stride_kernel as {}".format(os.getenv("FLAGS_use_stride_kernel")))

        os.environ["FLAGS_cudnn_deterministic"] = "True"
        logger.info("#### set FLAGS_cudnn_deterministic as {}".format(os.getenv("FLAGS_cudnn_deterministic")))

        path_now = os.getcwd()
        os.chdir(self.reponame)  # 执行setup要先切到路径下面
        # cmd_return = os.system("python -m pip install paddlegan")
        cmd_return = os.system("python setup.py install > paddlegan_install.log 2>&1 ")
        cmd_return1 = os.system("python -m pip install dlib >> paddlegan_install.log 2>&1 ")
        os.chdir(path_now)

        if cmd_return and cmd_return1:
            logger.info("repo {} python -m pip install paddlegan or dlib failed".format(self.reponame))
            # return 1

        return 0

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleGAN_Build, self).build_env()
        ret = 0
        ret = self.build_paddlegan()
        if ret:
            logger.info("build env whl failed")
            return ret

        logger.info("self.system  is {}".format(self.system))
        if "convergence" not in self.system:
            ret = self.build_yaml()
            if ret:
                logger.info("build env yaml failed")
                return ret

        ret = self.build_infer_dataset()
        if ret:
            logger.info("build env infer dataset failed")
            return ret

        logger.info("self.dataset_target is {}".format(self.dataset_target))
        if str(self.dataset_target) == "None":
            ret = self.build_dataset()
        else:
            ret = self.link_dataset()
        if ret:
            logger.info("build env dataset failed")
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

    # logger.info('###args {}'.format(args.models_file))
    model = PaddleGAN_Build(args)
    model.build_paddlegan()
    # model.build_yaml()
    # model.build_dataset()
