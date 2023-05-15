# encoding: utf-8
"""
自定义环境准备
"""
import os
import sys
import time
import platform
import logging
import tarfile
import argparse
import numpy as np
import yaml
import wget
from Model_Build import Model_Build

logger = logging.getLogger("ce")


class PaddleClas_Build(Model_Build):
    """
    自定义环境准备
    """

    def __init__(self, args):
        """
        初始化变量
        """
        self.data_path_endswith = "dataset"
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
            logger.info("#### mount_path diy_build is {}".format(self.mount_path))
            if os.listdir(self.mount_path) != []:
                self.dataset_org = self.mount_path
                os.environ["dataset_org"] = self.mount_path
                self.dataset_target = os.path.join(os.getcwd(), self.reponame, self.data_path_endswith)
                os.environ["dataset_target"] = os.path.join(os.getcwd(), self.reponame, self.data_path_endswith)
        logger.info("#### dataset_org in diy_build is  {}".format(self.dataset_org))
        logger.info("#### dataset_target in diy_build is  {}".format(self.dataset_target))

        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame)  # 所有和yaml相关的变量与此拼接
        self.reponame = args.reponame
        self.models_list = args.models_list
        self.models_file = args.models_file
        self.clas_model_list = []
        if str(self.models_list) != "None":
            for line in self.models_list.split(","):
                if ".yaml" in line:
                    self.clas_model_list.append(line.strip().replace("^", "/"))
        elif str(self.models_file) != "None":  # 获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                for line in open(file_name):
                    if ".yaml" in line:
                        self.clas_model_list.append(line.strip().replace("^", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" in file_name:
                    self.clas_model_list.append(file_name.strip().replace("^", "/"))

    def value_in_modellist(self, value=None):
        """
        判断字段是否存在model_list
        """
        for line in self.clas_model_list:
            if value in line:
                return 1
        return 0

    def download_data(self, value=None):
        """
        下载数据集
        """
        # 调用函数路径已切换至PaddleClas

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
        image_name = None
        try:
            with open(os.path.join(self.REPO_PATH, value), "r", encoding="utf-8") as y:
                cfg = yaml.full_load(y)
                image_name = cfg["DataLoader"]["Train"]["dataset"][label]
                # logger.info('####image_name: {}'.format(image_name))
                image_name = image_name.split("dataset")[1]
                # logger.info('####image_name: {}'.format(image_name))
                image_name = image_name.split("/")[1]
                # logger.info('####image_name: {}'.format(image_name))
                image_name = image_name.replace('"', "")
                # logger.info('####image_name: {}'.format(image_name))
        except:
            logger.info("#### {} can not open yaml".format(value))
        return image_name

    def change_yaml_batch_size(self, data_json, scale):
        """
        递归使所有batch_size,默认除以3
        """
        if isinstance(data_json, dict):
            for key, val in data_json.items():
                if (key == "batch_size" and "@" not in str(val)) or (key == "first_bs" and "@" not in str(val)):
                    data_json[key] = str(int(np.ceil(float(val) / scale))) + "  #@"
                if isinstance(data_json[key], dict):
                    self.change_yaml_batch_size(data_json[key], scale)
        return data_json

    def build_yaml(self):
        """
        更改RD原始的yaml文件
        demo: PaddleClas/ppcls/configs/ImageNet/ResNet/ResNet50.yaml
        """
        if os.path.exists(self.reponame):
            for line in self.clas_model_list:
                with open(os.path.join(self.REPO_PATH, line), "r", encoding="utf-8") as f:
                    content = yaml.load(f, Loader=yaml.FullLoader)

                # 改变 batch_size
                if "PKSampler" in str(content) or "DistributedRandomIdentitySampler" in str(content):
                    # logger.info("#### do not change batch_size in {}".format(line))
                    content_new = self.change_yaml_batch_size(content, 4)  # 写在with里面不能够全部生效
                    with open(os.path.join(self.REPO_PATH, line), "w") as f:
                        yaml.dump(content_new, f, sort_keys=False)
                else:
                    content_new = self.change_yaml_batch_size(content, 3)  # 写在with里面不能够全部生效
                    with open(os.path.join(self.REPO_PATH, line), "w") as f:
                        yaml.dump(content_new, f, sort_keys=False)

                # 改变 GeneralRecognition 依赖的数据集
                if "GeneralRecognition" in line:
                    content["DataLoader"]["Train"]["dataset"]["image_root"] = "./dataset/Inshop/"
                    content["DataLoader"]["Train"]["dataset"]["cls_label_path"] = "./dataset/Inshop/train_list.txt"
                    content["DataLoader"]["Eval"]["Query"]["dataset"]["image_root"] = "./dataset/iCartoonFace/"
                    content["DataLoader"]["Eval"]["Gallery"]["dataset"]["image_root"] = "./dataset/iCartoonFace/"
                    content["DataLoader"]["Eval"]["Query"]["dataset"][
                        "cls_label_path"
                    ] = "./dataset/iCartoonFace/gallery.txt"
                    content["DataLoader"]["Eval"]["Gallery"]["dataset"][
                        "cls_label_path"
                    ] = "./dataset/iCartoonFace/gallery.txt"
                    with open(os.path.join(self.REPO_PATH, line), "w") as f:
                        yaml.dump(content, f)
        else:
            logger.info("check you {} path".format(self.reponame))
        return 0

    def build_infer_dataset(self):
        """
        下载预测相关数据集
        """
        if os.path.exists(self.reponame):
            # 下载rec数据集 速度很快就默认下载了
            path_now = os.getcwd()
            os.chdir(self.reponame)
            os.chdir("deploy")
            if (
                os.path.exists("recognition_demo_data_v1.1")
                and os.path.exists("drink_dataset_v1.0")
                and os.path.exists("drink_dataset_v2.0")
            ):
                logger.info("#### already download rec_demo")
            else:
                logger.info("#### start download rec_demo")
                self.download_data(
                    value="https://paddle-imagenet-models-name.bj.bcebos.com\
                    /dygraph/rec/data/drink_dataset_v1.0.tar"
                )
                self.download_data(
                    value="https://paddle-imagenet-models-name.bj.bcebos.com\
                    /dygraph/rec/data/drink_dataset_v2.0.tar"
                )
                self.download_data(
                    value="https://paddle-imagenet-models-name.bj.bcebos.com\
                    /dygraph/rec/data/recognition_demo_data_en_v1.1.tar"
                )
                logger.info("#### end download rec_demo")
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
            # 根据不同的模型下载数据
            for line in self.clas_model_list:
                image_name = None  # 初始化
                if "face" in line and "metric_learning" in line:
                    image_name = self.get_image_name(value=line, label="root_dir")
                elif "traffic_sign" in line and "PULC" in line:
                    image_name = self.get_image_name(value=line, label="cls_label_path")
                elif "GeneralRecognition" in line:
                    self.download_data(
                        value="https://paddle-qa.bj.bcebos.com\
                        /{}/ce_data/iCartoonFace.tar".format(
                            self.reponame
                        )
                    )
                    self.download_data(
                        value="https://paddle-qa.bj.bcebos.com\
                        /{}/ce_data/Inshop.tar".format(
                            self.reponame
                        )
                    )
                elif "strong_baseline" in line and "reid" in line:
                    self.download_data(
                        value="https://paddle-qa.bj.bcebos.com\
                        /{}/ce_data/market1501.tar".format(
                            self.reponame
                        )
                    )
                elif (
                    "MV3_Large_1x_Aliproduct_DLBHC" in line or "ResNet50_vd_Aliproduct" in line
                ) and "Products" in line:
                    image_name = self.get_image_name(value=line, label="image_root")
                    self.download_data(
                        value="https://paddle-qa.bj.bcebos.com\
                        /{}/ce_data/Inshop.tar".format(
                            self.reponame
                        )
                    )
                else:
                    image_name = self.get_image_name(value=line, label="image_root")

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

    def build_paddleclas(self):
        """
        安装依赖包
        """
        # 固定随机量需要，默认打开
        logger.info("#### or AGILE_PIPELINE_NAME is {}".format(os.getenv("AGILE_PIPELINE_NAME")))
        # 230505 for wanghuan
        if str(os.getenv("AGILE_PIPELINE_NAME")) == "PaddleClas-Linux-Cuda102-Python37-ALL-Release-test3":
            os.environ["FLAGS_use_stride_kernel"] = "1"
        logger.info("#### set FLAGS_use_stride_kernel as {}".format(os.getenv("FLAGS_use_stride_kernel")))

        os.environ["FLAGS_cudnn_deterministic"] = "True"
        logger.info("#### set FLAGS_cudnn_deterministic as {}".format(os.getenv("FLAGS_cudnn_deterministic")))

        path_now = os.getcwd()
        os.chdir(self.reponame)  # 执行setup要先切到路径下面
        # cmd_return = os.system("python -m pip install paddleclas")
        cmd_return = os.system("python setup.py install > paddleclas_install.log 2>&1 ")
        logger.info("repo {} python -m pip install paddleclas done".format(self.reponame))
        if cmd_return:
            logger.info("repo {} python -m pip install paddleclas failed".format(self.reponame))
            # return 1
        os.chdir(path_now)

        if self.value_in_modellist(value="slim"):
            logger.info("#### slim install")
            # 安装paddleslim
            exit_code_slim = os.system(
                "python -m  pip install \
            https://paddle-qa.bj.bcebos.com/PaddleSlim/paddleslim-0.0.0.dev0-py3-none-any.whl \
                -i https://mirror.baidu.com/pypi/simple"
            )
            if exit_code_slim and ("Windows" not in platform.system() and "Darwin" not in platform.system()):
                exit_code_slim = os.system(
                    "python -m  pip install --user\
            https://paddle-qa.bj.bcebos.com/PaddleSlim/paddleslim-0.0.0.dev0-py3-none-any.whl \
                -i https://mirror.baidu.com/pypi/simple"
                )
            if exit_code_slim:
                logger.info("repo {} python -m pip install nvidia_dali_cuda110 failed".format(self.reponame))

        # if os.path.exists("PaddleSlim") is False:
        #     try:
        #         wget.download("https://xly-devops.bj.bcebos.com/PaddleTest/PaddleSlim/PaddleSlim-develop.tar.gz")
        #         tf = tarfile.open("PaddleSlim-develop.tar.gz")
        #         tf.extractall(os.getcwd())
        #         if os.path.exists("PaddleSlim-develop"):
        #             os.rename("PaddleSlim-develop", "PaddleSlim")
        #     except:
        #         logger.info("#### prepare download failed {} failed".format("PaddleSlim.tar.gz"))
        # if os.path.exists("PaddleSlim") and (
        #     "develop" in str(self.paddle_whl) or "Develop" in str(self.paddle_whl) or "None" in str(self.paddle_whl)
        # ):
        #     logger.info("#### install devlop paddleslim")
        #     path_now = os.getcwd()
        #     os.chdir("PaddleSlim")
        #     os.system("git checkout develop")
        #     os.system("git pull")
        #     exit_code_paddleslim = os.system(
        #         "python -m pip install -r requirements.txt \
        #         -i https://mirror.baidu.com/pypi/simple"
        #     )
        #     if exit_code_paddleslim and ("Windows" not in platform.system() and "Darwin" not in platform.system()):
        #         exit_code_paddleslim = os.system(
        #             "python -m pip install --user -r requirements.txt \
        #         -i https://mirror.baidu.com/pypi/simple"
        #         )
        #     os.system("python -m pip uninstall paddleslim -y")
        #     cmd_return = os.system("python setup.py install > paddleslim_install.log 2>&1 ")
        #     os.chdir(path_now)
        # else:
        #     logger.info("#### install release paddleslim")
        #     exit_code_paddleslim = os.system(
        #         "python -m pip install -U paddleslim \
        #         -i https://mirror.baidu.com/pypi/simple"
        #     )
        # if exit_code_paddleslim and ("Windows" not in platform.system() and "Darwin" not in platform.system()):
        #     exit_code_paddleslim = os.system(
        #         "python -m pip install --user -U paddleslim \
        #         -i https://mirror.baidu.com/pypi/simple"
        #     )
        #     logger.info("repo {} python -m pip install paddleslim failed".format(self.reponame))
        #     # return 1

        if self.value_in_modellist(value="face") and self.value_in_modellist(value="metric_learning"):
            logger.info("#### face and metric_learning install")
            exit_code_setuptools = os.system(
                "python -m  pip install -U pip setuptools cython \
                -i https://mirror.baidu.com/pypi/simple"
            )
            if exit_code_setuptools:
                exit_code_setuptools = os.system(
                    "python -m  pip install --user -U pip setuptools cython \
                    -i https://mirror.baidu.com/pypi/simple"
                )
            if exit_code_setuptools:
                logger.info("repo {} python -m pip install setuptools failed".format(self.reponame))
                # return 1
            exit_code_bcolz = os.system(
                "python -m  pip install bcolz==1.2.0 \
                -i https://mirror.baidu.com/pypi/simple"
            )
            if exit_code_bcolz and ("Windows" not in platform.system() and "Darwin" not in platform.system()):
                exit_code_bcolz = os.system(
                    "python -m  pip install bcolz==1.2.0 --user \
                -i https://mirror.baidu.com/pypi/simple"
                )
            if exit_code_bcolz:
                logger.info("repo {} python -m pip install bcolz failed".format(self.reponame))
                # return 1

        if (self.value_in_modellist(value="amp") or self.value_in_modellist(value="dy2st_convergence")) and (
            "Windows" not in platform.system() and "Darwin" not in platform.system()
        ):
            logger.info("#### fp16 or amp install")
            if os.path.exists("nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl") and os.path.exists(
                "nvidia_dali_cuda110-1.23.0-7355173-py3-none-manylinux2014_x86_64.whl"
            ):
                logger.info("#### already download nvidia_dali_cuda102 nvidia_dali_cuda110")
            else:
                try:
                    wget.download(
                        "https://paddle-qa.bj.bcebos.com/PaddleClas/{}".format(
                            "nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl"
                        )
                    )
                    wget.download(
                        "https://paddle-qa.bj.bcebos.com/PaddleClas/{}".format(
                            "nvidia_dali_cuda110-1.23.0-7355173-py3-none-manylinux2014_x86_64.whl"
                        )
                    )
                except:
                    logger.info("#### prepare download failed {} failed".format("nvidia_dali"))
            # # 改变numpy版本
            # cuda102 只支持py310以下
            # logger.info("because of dali have np.int, so change numpy version")
            # exit_code_numpy = os.system(
            #     "python -m  pip install numpy==1.20.2 \
            #     -i https://mirror.baidu.com/pypi/simple"
            # )
            # if exit_code_numpy and ("Windows" not in platform.system() and "Darwin" not in platform.system()):
            #     exit_code_numpy = os.system(
            #         "python -m  pip install --user numpy==1.20.2 \
            #         -i https://mirror.baidu.com/pypi/simple"
            #     )
            # # 安装nvidia
            # exit_code_nvidia = os.system(
            #     "python -m  pip install \
            # nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl \
            #     -i https://mirror.baidu.com/pypi/simple"
            # )
            # if exit_code_nvidia and ("Windows" not in platform.system() and "Darwin" not in platform.system()):
            #     exit_code_nvidia = os.system(
            #         "python -m  pip install --user\
            # nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl \
            #     -i https://mirror.baidu.com/pypi/simple"
            #     )
            # if exit_code_nvidia:
            #     logger.info("repo {} python -m pip install nvidia_dali_cuda102 failed".format(self.reponame))
            #     # return 1

            # cuda11  最新numpy有BUG
            logger.info("because of dali have np.int, so change numpy version")
            exit_code_numpy = os.system(
                "python -m  pip install numpy==1.21.2 \
                -i https://mirror.baidu.com/pypi/simple"
            )
            if exit_code_numpy and ("Windows" not in platform.system() and "Darwin" not in platform.system()):
                exit_code_numpy = os.system(
                    "python -m  pip install --user numpy==1.21.2 \
                    -i https://mirror.baidu.com/pypi/simple"
                )
            # 安装nvidia
            exit_code_nvidia = os.system(
                "python -m  pip install \
            nvidia_dali_cuda110-1.23.0-7355173-py3-none-manylinux2014_x86_64.whl \
                -i https://mirror.baidu.com/pypi/simple"
            )
            if exit_code_nvidia and ("Windows" not in platform.system() and "Darwin" not in platform.system()):
                exit_code_nvidia = os.system(
                    "python -m  pip install --user\
            nvidia_dali_cuda110-1.23.0-7355173-py3-none-manylinux2014_x86_64.whl \
                -i https://mirror.baidu.com/pypi/simple"
                )
            if exit_code_nvidia:
                logger.info("repo {} python -m pip install nvidia_dali_cuda110 failed".format(self.reponame))
            #     # return 1

        # amp 静态图 单独考虑，不能固定随机量，否则报错如下, 或者 set FLAGS_cudnn_exhaustive_search=False
        # if self.value_in_modellist(value="amp"):
        #     os.environ["FLAGS_cudnn_deterministic"] = "False"
        #     logger.info("set FLAGS_cudnn_deterministic as {}".format("False"))
        return 0

    def build_env(self):
        """
        使用父类实现好的能力
        """
        super(PaddleClas_Build, self).build_env()
        ret = 0
        ret = self.build_paddleclas()
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
    model = PaddleClas_Build(args)
    model.build_paddleclas()
    # model.build_yaml()
    # model.build_dataset()
