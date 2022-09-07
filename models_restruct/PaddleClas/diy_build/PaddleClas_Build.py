# encoding: utf-8
"""
自定义环境准备
"""
import os
import sys
import wget
import yaml
import tarfile
import argparse
import numpy as np
from Model_Build import Model_Build

class PaddleClas_Build(Model_Build):
    """
    自定义环境准备
    """
    def __init__(self, args):
        """
        初始化变量
        """
        # print('###args', args.models_file)
        self.REPO_PATH = os.path.join(os.getcwd(), args.reponame) #所有和yaml相关的变量与此拼接
        self.reponame = args.reponame
        self.dataset_org = args.dataset_org
        self.models_list = args.models_list
        # print('####models_list', self.models_list)
        self.models_file = args.models_file
        # print('####models_file', self.models_file)
        self.clas_model_list=[]
        if self.models_file: #获取要执行的yaml文件列表
            for file_name in self.models_file.split(","):
                with open(file_name, encoding='utf-8') as readfile:
                    line = readfile.readline()
                    while line:
                        if ".yaml" in line:
                            self.clas_model_list.append(line.strip().replace("-", "/"))
        elif self.models_list:
            for line in self.models_list.split(","):
                if ".yaml" in line:
                    self.clas_model_list.append(line.strip().replace("-", "/"))
        else:
            for file_name in os.listdir("cases"):
                if ".yaml" in file_name:
                    self.clas_model_list.append(file_name.strip().replace("-", "/"))

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
        #调用函数路径已切换至PaddleClas
        
        tar_name = value.split("/")[-1]
        if os.path.exists(tar_name) and os.path.exists(tar_name.replace(".tar", "")):
            print("#### already download {}".format(tar_name))
        else:
            print("#### start download {}".format(tar_name))
            wget.download(value.replace(' ', ''))
            print("#### end download {}".format(tar_name))
            if os.path.exists(tar_name):
                tf = tarfile.open(tar_name)
                tf.extractall(os.getcwd())
            else:
                return 1
        return 0
    
    def get_image_name(self, value=None, label=None):
        """
        获取数据集名称
        """
        with open(os.path.join(self.REPO_PATH, value), 'r', encoding='utf-8') as y:
            cfg = yaml.full_load(y)
            image_name = cfg['DataLoader']['Train']['dataset'][label]
            # print('####image_name', image_name)
            image_name = image_name.split("dataset")[1]
            # print('####image_name', image_name)
            image_name = image_name.split("/")[1]
            # print('####image_name', image_name)
            image_name = image_name.replace('\"', '')
            # print('####image_name', image_name)
        return image_name

    def change_yaml_batch_size(self, data_json):
        """
        递归使所有batch_size,默认除以3
        """
        if isinstance(data_json, dict):
            for key, val in data_json.items():
                if key == "batch_size":
                    # print("###data_json[key]",data_json[key])
                    # print("###val",val)
                    # input()
                    data_json[key] = int(np.ceil(float(val)/3)) #batch_size除以3向上取整，TODO 添加特殊符号标识，防止多次除以3
                if isinstance(data_json[key], dict):
                    self.change_yaml_batch_size(data_json[key])
        return data_json

    def build_yaml(self):
        """
        更改RD原始的yaml文件
        demo: PaddleClas/ppcls/configs/ImageNet/ResNet/ResNet50.yaml
        """
        if os.path.exists(self.reponame):
            for line in self.clas_model_list:
                if "strong_baseline" in line or "Logo" in line:
                    print("{} do no need change bs".format(line)) #针对特殊的sampler 需要满足整除某数的bs
                else:
                    with open(os.path.join(self.REPO_PATH, line), 'r') as f:
                        content = yaml.load(f, Loader = yaml.FullLoader)
                    content_new = self.change_yaml_batch_size(content) #卸载with里面不能够全部生效
                    with open(os.path.join(self.REPO_PATH, line), 'w') as f:
                        yaml.dump(content_new, f)
        else:
            print("check you {} path".format(self.reponame))
        return 0

    # def build_pretrain(self):
    #     """
    #     自定义下载预训练模型
    #     """
    #     print("待支持下载预训练模型")
    #     retrun 0

    def build_dataset(self):
        """
        自定义下载数据集
        """
        if os.path.exists(self.reponame):
            #下载rec数据集 速度很快就默认下载了
            path_now = os.getcwd()
            os.chdir(self.reponame)
            os.chdir("deploy")
            if os.path.exists("recognition_demo_data_en_v1.1.tar") and \
                os.path.exists("drink_dataset_v1.0.tar"):
                print("#### already download rec_demo")
            else:
                print("#### start download rec_demo")
                self.download_data(value="https://paddle-imagenet-models-name.bj.bcebos.com\
                    /dygraph/rec/data/drink_dataset_v1.0.tar")
                self.download_data(value="https://paddle-imagenet-models-name.bj.bcebos.com\
                    /dygraph/rec/data/recognition_demo_data_en_v1.1.tar")
                print("#### end download rec_demo")
            os.chdir(path_now)

            #下载dataset数据集 解析yaml下载
            path_now = os.getcwd()
            os.chdir(self.reponame)
            if os.path.exists("dataset") == False:
                os.mkdir("dataset")
            os.chdir("dataset")
            # 根据不同的模型下载数据
            for line in self.clas_model_list:
                if "face" in line and "metric_learning" in line:
                    image_name = self.get_image_name(value=line, label="root_dir")
                elif "traffic_sign" in line and "PULC" in line:
                    image_name = self.get_image_name(value=line, label="cls_label_path")
                elif "GeneralRecognition" in line:
                    self.download_data(value="https://paddle-qa.bj.bcebos.com\
                        /{}/ce_data/iCartoonFace.tar".format(self.reponame))
                    self.download_data(value="https://paddle-qa.bj.bcebos.com\
                        /{}/ce_data/Inshop.tar".format(self.reponame))
                    del image_name
                elif "strong_baseline" in line and "reid" in line:
                    self.download_data(value="https://paddle-qa.bj.bcebos.com\
                        /{}/ce_data/market1501.tar".format(self.reponame))
                    del image_name
                elif "MV3_Large_1x_Aliproduct_DLBHC" in line and "Products" in line:
                    image_name = self.get_image_name(value=line, label="image_root")
                    self.download_data(value="https://paddle-qa.bj.bcebos.com\
                        /{}/ce_data/Inshop.tar".format(self.reponame))
                else:
                    image_name = self.get_image_name(value=line, label="image_root")

                if 'image_name' in dir() == False:
                    print("do not need download")
                else:
                    self.download_data(value="https://paddle-qa.bj.bcebos.com\
                        /{}/ce_data/{}.tar".format(self.reponame, image_name))
            os.chdir(path_now)
        else:
            print("check you {} path".format(self.reponame))
        return 0

    def build_paddleclas(self):
        """
        安装依赖包
        """
        #固定随机量需要，默认打开
        os.environ['FLAGS_cudnn_deterministic'] = "True" 
        print("#### set FLAGS_cudnn_deterministic as {}".format(os.environ['FLAGS_cudnn_deterministic']))

        cmd_return = os.system("python -m pip install paddleclas==2.4.3")
        if cmd_return:
            print("repo {} python -m pip install paddleclas failed".format(self.reponame))
            return 1

        if self.value_in_modellist(value="slim"):
            print("#### slim install")
            cmd_return = os.system("python -m pip install paddleslim")
            if cmd_return:
                print("repo {} python -m pip install paddleslim failed".format(self.reponame))
                return 1
            
        if self.value_in_modellist(value="face") and self.value_in_modellist(value="metric_learning"):
            print("#### face and metric_learning install")
            cmd_return = os.system(" python -m  pip install -U pip setuptools cython")
            if cmd_return:
                print("repo {} python -m pip install setuptools failed".format(self.reponame))
                return 1
            cmd_return = os.system("python -m  pip install bcolz==1.2.0")
            if cmd_return:
                print("repo {} python -m pip install bcolz failed".format(self.reponame))
                return 1

        if self.value_in_modellist(value="amp"):
            print("#### fp16 or amp install")
            if os.path.exists("nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl") and \
                os.path.exists("nvidia_dali_cuda110-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl"):
                print("#### already download nvidia_dali_cuda102 nvidia_dali_cuda110")
            else:
                wget.download("https://paddle-qa.bj.bcebos.com/PaddleClas/\
                    nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl")
                wget.download("https://paddle-qa.bj.bcebos.com/PaddleClas/\
                    nvidia_dali_cuda110-1.8.0-3362434-py3-none-manylinux2014_x86_64.whl")

            cmd_return = os.system("python -m  pip install \
               nvidia_dali_cuda102-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl")
            if cmd_return:
                print("repo {} python -m pip install nvidia_dali_cuda102 failed".format(self.reponame))
                return 1
            cmd_return = os.system("python -m  pip install \
               nvidia_dali_cuda110-1.8.0-3362432-py3-none-manylinux2014_x86_64.whl")
            if cmd_return:
                print("repo {} python -m pip install nvidia_dali_cuda110 failed".format(self.reponame))
                return 1

            os.environ['FLAGS_cudnn_deterministic'] = False
            print("set FLAGS_cudnn_deterministic as {}".format("False"))
            #amp单独考虑，不能固定随机量，否则报错如下
        return 0 

    def build_env(self, args):
        """
        使用父类实现好的能力
        """
        super(PaddleClas_Build, self).build_env(args)
        ret = 0
        ret = self.build_paddleclas()
        if ret:
            print("build env whl failed")
            return ret
        ret = self.build_yaml()
        if ret:
            print("build env yaml failed")
            return ret
        ret = self.build_dataset()
        if ret:
            print("build env dataset failed")
            return ret
        return ret

if __name__ == '__main__':
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

    # print('###args', args.models_file)
    model = PaddleClas_Build(args)
    model.build_paddleclas()
    # model.build_yaml()
    # model.build_dataset()