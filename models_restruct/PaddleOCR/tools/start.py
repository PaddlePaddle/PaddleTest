# encoding: utf-8
"""
执行case前：生成不同模型的配置参数，例如算法、字典等
"""

import os
import sys
import re
import json
import shutil
import logging
import tarfile
import argparse
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")


class PaddleOCR_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("###self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接
        self.env_dict = {}
        self.model=os.path.splitext(os.path.basename(self.rd_yaml_path))[0]
        self.category=re.search('/(.*?)/',self.rd_yaml_path).group(1)


 

    def prepare_config_params(self):
        """
        准备配置参数
        """
        print('start prepare_config_params!')
        yaml_absolute_path=os.path.join(self.REPO_PATH, self.rd_yaml_path)
        self.rd_config=yaml.load(open(yaml_absolute_path,'rb'), Loader=yaml.Loader)
        algorithm=self.rd_config['Architecture']['algorithm']
        self.env_dict["algorithm"] = algorithm
        # os.environ['algorithm'] = algorithm
        if 'character_dict_path' in self.rd_config['Global'].keys():
            rec_dict=self.rd_config['Global']['character_dict_path']
            if not rec_dict:
                rec_dict='ppocr/utils/ic15_dict.txt'
            self.env_dict["rec_dict"] = rec_dict
            with open(yaml_absolute_path) as f:
                 lines = f.readlines()
                 for line in lines:
                     if 'image_shape' in line:
                          image_shape_list=line.strip('\n').split(':')[-1]
                          print(image_shape_list)
                          image_shape_list=(image_shape_list.replace(' ',''))
                          image_shape = re.findall(r'\[(.*?)\]',image_shape_list)
                          if not image_shape:
                              image_shape='2,32,320'
                          else:
                              image_shape=image_shape[0]
                          print(image_shape)
                          break
                     else:
                         image_shape='3,32,128'
            self.env_dict["image_shape"] = image_shape

    def prepare_pretrained_model(self):
        path_now=os.getcwd()
        pretrained_yaml_path=os.path.join(os.getcwd(),'tools/ocr_pretrained.yaml')
        pretrained_yaml=yaml.load(open(pretrained_yaml_path,'rb'), Loader=yaml.Loader)
        if self.model in pretrained_yaml[self.category].keys():
            print('{} exist in pretrained_yaml!'.format(self.model))
            print(pretrained_yaml[self.category][self.model])
            pretrained_model_link=pretrained_yaml[self.category][self.model]
            os.chdir('PaddleOCR')
            tar_name = pretrained_model_link.split("/")[-1]
            if not os.path.exists(tar_name):
               wget.download(pretrained_model_link)
               tf = tarfile.open(tar_name)
               tf.extractall(os.getcwd())
               os.rename(os.path.splitext(tar_name)[0],self.model)
            os.chdir(path_now)
            self.env_dict["model"] = self.model
        else: 
            print('{} not exist in pretrained_yaml!'.format(self.model))


    def gengrate_test_case(self):
        print(os.getcwd())
        print(os.path.join('cases',self.qa_yaml_name))
        path_now=os.getcwd()
        pretrained_yaml_path=os.path.join(os.getcwd(),'tools/ocr_pretrained.yaml')
        pretrained_yaml=yaml.load(open(pretrained_yaml_path,'rb'), Loader=yaml.Loader)
        if not os.path.exists('cases'):
           os.makedirs('cases')
        with open((os.path.join('cases',self.qa_yaml_name)+'.yml'), 'w') as f:
             if self.model in pretrained_yaml[self.category].keys():
                  f.writelines(('case:'+ os.linesep,
                           '    linux:'+ os.linesep,
                           '        base: ./base/ocr_'+self.category+'_base_pretrained.yaml' + os.linesep))
             else:
                  f.writelines(('case:'+ os.linesep,
                           '    linux:'+ os.linesep,
                           '        base: ./base/ocr_'+self.category+'_base.yaml' + os.linesep))
     

    def build_prepare(self):
        """
        执行准备过程
        """
        # 进入repo中
        ret = 0
        ret = self.prepare_config_params()
        if ret:
            logger.info("build prepare_config_params failed")
        self.prepare_pretrained_model()
        self.gengrate_test_case()
        os.environ[self.reponame] = json.dumps(self.env_dict)
        for k,v in self.env_dict.items():
            os.environ[k] = v
        return ret


def run():
    """
    执行入口
    """
    model = PaddleOCR_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    run()
