"""
start before model running
"""
import os
import sys
import json
import shutil
import urllib
import logging
import wget

logger = logging.getLogger("ce")


class DeepXDE_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        init
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("###self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        logger.info("###self.step: {}".format(self.step))
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)
        self.env_dict = {}
        self.model = self.qa_yaml_name.split("^")[-1]
        logger.info("###self.model_name: {}".format(self.model))
        self.env_dict["model"] = self.model
        os.environ["model"] = self.model

    def prepare_gpu_env(self):
        """
        根据操作系统获取用gpu还是cpu
        """
        if "cpu" in self.system or "mac" in self.system:
            self.env_dict["set_cuda_flag"] = "cpu"  # 根据操作系统判断
        else:
            self.env_dict["set_cuda_flag"] = "gpu"  # 根据操作系统判断
        return 0

    def add_paddle_to_pythonpath(self):
        """
        paddlescience 打包路径添加到python的路径中
        """
        cwd = os.getcwd()
        paddle_path = os.path.join(cwd, "deepxde")
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        new_pythonpath = f"{paddle_path}:{old_pythonpath}"
        os.environ["PYTHONPATH"] = new_pythonpath
        os.environ["DDE_BACKEND"] = "paddle"
        return 0

    def alter(self, file, old_str, new_str, flag=True, except_str="model.train(0"):
        """
        replaced the backend
        """
        file_data = ""
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if flag:
                    if old_str in line and new_str not in line and except_str not in line:
                        line = line.replace(old_str, new_str)
                else:
                    if old_str in line:
                        line = line.replace(old_str, new_str)
                file_data += line
        with open(file, "w", encoding="utf-8") as f:
            f.write(file_data)
        return 0

    def add_seed(self, file, old_str, new_str):
        """
        add the seed
        """
        file_data = ""
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if old_str in line:
                    if old_str == "L-BFGS":
                        if "    " not in line:
                            global flag_LBFGS
                            flag_LBFGS = True
                            line += new_str
                    else:
                        line += new_str
                    # line += "paddle.seed(1)\n"
                    # line += "np.random.seed(1)\n"
                file_data += line
        with open(file, "w", encoding="utf-8") as f:
            f.write(file_data)
        return 0

    def change_backend(self, file, backend, flag):
        """
        change models.py backend
        """
        file_data = ""
        if flag is True:
            index = False
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    if index is True:
                        if "# " in line and "Backend jax" not in line:
                            line = line.replace("# ", "")
                        else:
                            index = False
                    if backend in line:
                        index = True
                    file_data += line
            with open(file, "w", encoding="utf-8") as f:
                f.write(file_data)
        else:
            index = False
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    if index is True:
                        if "Backend paddle" not in line:
                            line = "# " + line
                        else:
                            index = False
                    if backend in line:
                        index = True
                    file_data += line
            with open(file, "w", encoding="utf-8") as f:
                f.write(file_data)
        return 0

    def get_example_dir(self):
        """
        get_example_dir
        """
        example_dir = self.qa_yaml_name.replace("^", "/")
        if "lulu" in example_dir:
            example_dir = "deepxde" + example_dir[4:] + ".py"
        elif "rd" in example_dir:
            example_dir = "deepxde" + example_dir[2:] + ".py"
        return example_dir

    def get_deepxde_data(self):
        """
        get_deepxde_data
        """
        os.system("cp -r deepxde/examples/dataset/ ./")
        return 0

    def build_prepare(self):
        """
        build prepare
        """
        ret = 0
        ret = self.prepare_gpu_env()
        if ret:
            logger.info("build prepare_gpu_env failed")
            return ret
        os.environ[self.reponame] = json.dumps(self.env_dict)
        return ret

    def download_datasets(self):
        """
        download dataset
        """
        url = "https://paddle-qa.bj.bcebos.com/deepxde/datasets.tar.gz"
        file_name = "datasets.tar.gz"
        urllib.request.urlretrieve(url, file_name)
        os.system("tar -zxvf " + file_name + " -C deepxde/")
        return 0

def run():
    """
    执行入口
    """
    model = DeepXDE_Start()
    model.build_prepare()
    model.add_paddle_to_pythonpath()
    model.get_deepxde_data()
    filedir = model.get_example_dir()
    model.alter(filedir, "tf", "paddle")
    model.change_backend(filedir, "Backend paddle", True)
    model.change_backend(filedir, "Backend tensorflow.compat.v1", False)
    model.alter(filedir, "model.train(", "model.train(display_every=1,", True, "model.train(0")
    model.alter(filedir, "model.train(", "losshistory, train_state = model.train(")
    model.alter(filedir, "display_every=1000,", " ", False)
    model.alter(filedir, "display_every=1000", " ", False)
    model.alter(filedir, "display_every=500", " ", False)
    model.add_seed(filedir, "import deepxde", "import paddle\n")
    # add_seed(filedir, "import paddle", "paddle.seed(1)\n")
    model.add_seed(filedir, "import deepxde", "import numpy as np\n")
    model.add_seed(filedir, "import deepxde", "dde.config.set_random_seed(1)\n")
    if "antiderivative" in model.qa_yaml_name:
        model.download_datasets()
    return 0


if __name__ == "__main__":
    run()
