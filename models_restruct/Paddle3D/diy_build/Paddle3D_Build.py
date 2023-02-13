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
        self.paddle_whl = args.paddle_whl

        self.branch = args.branch
        self.system = args.system
        self.set_cuda = args.set_cuda
        self.dataset_org = args.dataset_org
        self.dataset_target = args.dataset_target

        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame)  # 所有和yaml相关的变量与此拼接
        self.reponame = args.reponame
        self.get_repo = args.get_repo
        print("self.reponame:{}".format(self.reponame))
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
            path_now = os.getcwd()
            os.chdir(self.reponame)

            sysstr = platform.system()
            if sysstr == "Linux":
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
                # setuptools not found on python3.10
                os.system('/usr/bin/python3.10  -m pip uninstall -y setuptools')
                os.system('/usr/bin/python3.10  -m pip install --upgrade setuptools')
                os.system('python -m pip install --upgrade pip')
                os.system('apt-get install -y python3-setuptools')
                os.system('apt-get install -y python-setuptools')
                os.system('python -m pip install ez_setup')
                os.system('curl -O http://python-distribute.org/distribute_setup.py')
                os.system('python distribute_setup.py')
                os.system('easy_install pip')
                os.system("python -m pip install https://files.pythonhosted.org/packages/34/6a/d89fe6b83a7fa4c39686d8a5cda83b48da0b117a13484f0e294c2cc2c337/setuptools-67.2.0-py3-none-any.whl")
               
                os.system('wget https://files.pythonhosted.org/packages/5f/36/7374297692bb9dbd7569a0f84887c7e5e314c41d5d9518cb76fbb130620d/setuptools-67.2.0.tar.gz')
                os.system('tar xf setuptools-67.2.0.tar.gz')
                os.system('cd setuptools-67.2.0')
               
                os.system("python setup.py build")
                os.system("python setup.py install”")
                os.system("cd ..")
                os.system("python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple")

            sys.path
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
        return ret
