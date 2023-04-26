# encoding: utf-8
"""
执行case后：回收数据避免占用太多存储空间
"""
import os
import sys
import json
import glob
import shutil
import argparse
import logging
import yaml
import wget
import numpy as np

# from picture.analysis import analysis, draw
from picture.analysis import plt_dy2st

logger = logging.getLogger("ce")


class PaddleClas_End(object):
    """
    回收类
    """

    def __init__(self):
        """
        初试化
        """
        self.reponame = os.environ["reponame"]
        self.qa_yaml_name = os.environ["qa_yaml_name"]

    def dy2st_plt(self):
        """
        绘制动转静图片
        """
        path_now = os.getcwd()
        os.chdir("picture")
        try:
            path_list = []
            for name in os.listdir(os.path.join("../logs", self.reponame, self.qa_yaml_name)):
                path_list.append(os.path.join("../logs", self.reponame, self.qa_yaml_name, name))
            plt_dy2st(path_list, self.qa_yaml_name)
            # with open("dy2st.yaml", "r", encoding="utf-8") as f:
            #     content = yaml.load(f, Loader=yaml.FullLoader)
            # self.dy2st_yaml = content[self.qa_yaml_name]
            # logger.info("self.dy2st_yaml is {}".format(self.dy2st_yaml))
            # # train
            # train_log_info_map = analysis(
            #     os.path.join("../logs", self.reponame, self.qa_yaml_name), self.dy2st_yaml, "train"
            # )
            # draw(train_log_info_map, self.dy2st_yaml, "train")
            # # eval
            # if "eval" in self.dy2st_yaml.keys():
            #     train_log_info_map = analysis(
            #         os.path.join("../logs", self.reponame, self.qa_yaml_name), self.dy2st_yaml, "eval"
            #     )
            #     draw(train_log_info_map, self.dy2st_yaml, "eval")
        except Exception as e:
            logger.info("draw picture failed")
            logger.info("error info : {}".format(e))
        os.chdir(path_now)
        return 0

    def remove_data(self):
        """
        回收之前下载的数据
        """
        list_dir = os.listdir(self.reponame)
        path_now = os.getcwd()
        os.chdir(self.reponame)

        for file_name in list_dir:
            if ".tar" in file_name:
                os.remove(file_name)
                if os.path.exists(file_name.replace(".tar", "")):
                    shutil.rmtree(file_name.replace(".tar", ""))
                logger.info("#### clean data _infer: {}".format(file_name))

            if "_pretrained.pdparams" in file_name:
                os.remove(file_name)
                logger.info("#### clean data: {}".format(file_name))

            # if file_name == "inference":
            #     shutil.rmtree("inference")
            #     logger.info("#### clean data inference: {}".format("inference"))

            if file_name == "output":
                del_pdparams = glob.glob(r"output/*/*/*.pdparams")
                del_pdopt = glob.glob(r"output/*/*/*.pdopt")
                if del_pdparams != [] and del_pdopt != []:
                    logger.info("#### clean data pdparams: {}".format("pdparams"))
                    for del_name in del_pdparams:
                        if "latest" not in del_name:  # 保留最后一个epoch是完整的，其它只保留结构参数
                            os.remove(del_name)
                    for del_name in del_pdopt:
                        if "latest" not in del_name:
                            os.remove(del_name)

            if file_name == "dataset":
                del_dataset = glob.glob(r"dataset/*.tar")
                if del_dataset != []:
                    logger.info("#### clean data dataset: {}".format("dataset"))
                    for del_name in del_dataset:
                        os.remove(del_name)
                if os.path.exists(os.path.join("dataset", "face")):
                    shutil.rmtree(os.path.join("dataset", "face"))
                    logger.info("#### clean data face: {}".format("dataset"))

        os.chdir(path_now)
        return 0

    def build_end(self):
        """
        执行准备过程
        """
        # 进入repo中
        logger.info("build remove_data start")
        ret = 0
        ret = self.remove_data()
        if ret:
            logger.info("build remove_data failed")
            return ret
        if "dy2st_convergence" in self.qa_yaml_name:
            logger.info("dy2st_convergence star draw picture")
            self.dy2st_plt()
        logger.info("build remove_data end")
        return ret


def run():
    """
    执行入口
    """
    model = PaddleClas_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
