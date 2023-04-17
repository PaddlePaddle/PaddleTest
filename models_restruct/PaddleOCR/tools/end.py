# encoding: utf-8
"""
执行case后：获取log中的数值
"""
import os
import sys
import re
import json
import glob
import shutil
import math
import argparse
import logging
import platform
import yaml
import wget
import paddle
import allure
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("ce")


class PaddleOCR_End(object):
    """
    回收类
    """

    def __init__(self):
        """
        初试化
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("#### self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)  # 所有和yaml相关的变量与此拼接
        self.model = os.path.splitext(os.path.basename(self.rd_yaml_path))[0]
        self.category = re.search("/(.*?)/", self.rd_yaml_path).group(1)
        self.TRAIN_LOG_PATH = os.path.join("logs", self.reponame, self.qa_yaml_name, "train_multi.log")
        self.EVAL_LOG_PATH = os.path.join("logs", self.reponame, self.qa_yaml_name, "eval_pretrained.log")
        self.branch = os.environ["branch"]

    def update_json(self, filename, value):
        """
        update_json
        """
        # 1.读取原始json文件
        with open(filename, "r") as f:
            content = json.load(f)
        logger.info("#### content: {}".format(content))

        # 2.更新字典dict
        value_dict = {self.model: value}
        content.update(value_dict)
        logger.info("#### value_dict: {}".format(value_dict))

        # 3.写入
        with open(filename, "w") as f_new:
            json.dump(content, f_new)

    def collect_data_value(self):
        """
        回收之前下载的数据
        """
        if self.step == "train" and os.environ.get("UPDATA_BASE_VALUE") is True:
            # parse kpi
            train_loss = self.getdata(self.TRAIN_LOG_PATH, "loss")
            logger.info("#### train_loss: {}".format(train_loss))
            self.update_json("tools/train.json", train_loss)
        elif self.step == "eval" and os.environ.get("UPDATA_BASE_VALUE") is True:
            # eval acc
            pretrained_yaml_path = os.path.join(os.getcwd(), "tools/ocr_pretrained.yaml")
            pretrained_yaml = yaml.load(open(pretrained_yaml_path, "rb"), Loader=yaml.Loader)
            if self.model in pretrained_yaml[self.category].keys():
                if self.category == "det" or self.category == "kie":
                    if self.model == "det_r18_vd_ct":
                        eval_acc = self.getdata(self.EVAL_LOG_PATH, "f_score")
                    else:
                        eval_acc = self.getdata(self.EVAL_LOG_PATH, "hmean")
                elif self.category == "e2e":
                    eval_acc = self.getdata(self.EVAL_LOG_PATH, "f_score_e2e")
                elif self.category == "sr":
                    eval_acc = self.getdata(self.EVAL_LOG_PATH, "psnr_avg")
                else:
                    eval_acc = self.getdata(self.EVAL_LOG_PATH, "acc")
                logger.info("#### eval_acc: {}".format(eval_acc))
                self.update_json("tools/eval.json", eval_acc)
        else:
            pass

    def config_report_enviorement_variable(self):
        """
        generate report_enviorement_variable dict
        """
        logger.info("config report_enviorement_dict start")
        report_enviorement_dict = {}
        python_version = platform.python_version()
        paddle_version = paddle.__version__
        paddle_commit = paddle.version.commit
        os.chdir(self.reponame)
        models_commit = os.popen("git rev-parse HEAD").read().replace("\n", "")
        os.chdir("..")

        report_enviorement_dict["python_version"] = python_version
        report_enviorement_dict["paddle_version"] = paddle_version
        report_enviorement_dict["paddle_commit"] = paddle_commit
        report_enviorement_dict["model_repo_name"] = self.reponame
        report_enviorement_dict["model_branch"] = self.branch
        report_enviorement_dict["models_commit"] = models_commit

        if os.path.exists("result/environment.properties"):
            os.remove("result/environment.properties")
        with open("result/environment.properties", "w") as f:
            for key, value in report_enviorement_dict.items():
                f.write(str(key) + "=" + str(value) + "\n")

    def plot_paddle_compare_value(self, data1, data2, value, keyworld="prim"):
        """
        plot_paddle_compare_value
        """

        ydata1 = data1
        xdata1 = list(range(0, len(ydata1)))
        ydata2 = data2
        xdata2 = list(range(0, len(ydata2)))

        # plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xdata1, ydata1, label="paddle_dygraph2static_baseline_" + value, color="b", linewidth=2)
        ax.plot(xdata2, ydata2, label="paddle_dygraph2static_prim_" + value, color="r", linewidth=2)

        # set the legend
        ax.legend()
        # set the limits
        ax.set_xlim([0, len(xdata1)])
        if "loss" in value:
            ax.set_ylim([0, math.ceil(np.percentile(ydata1, 99.99))])
        else:
            ax.set_ylim([0, math.ceil(max(ydata1))])
        if "loss" in value:
            ax.set_xlabel("iteration")
        else:
            ax.set_xlabel("epoch")
        ax.set_ylabel(value)
        ax.grid()
        ax.set_title("PaddleOCR_DB")

        # display the plot
        # plt.show()
        if not os.path.exists("picture"):
            os.makedirs("picture")
        plt.savefig("picture/" + keyworld + "_dygraph2static_" + value + ".png")

    def get_paddle_data(self, filepath, kpi):
        """
        get_paddle_data(
        """
        data_list = []
        f = open(filepath, encoding="utf-8", errors="ignore")
        for line in f.readlines():
            if kpi + ":" in line and line.startswith("20"):
                if "current" in line:
                    pass
                else:
                    regexp = r"%s:(\s*\d+(?:\.\d+)?)" % kpi
                    r = re.findall(regexp, line)
                    # 如果解析不到符合格式到指标，默认值设置为-1
                    kpi_value = float(r[0].strip()) if len(r) > 0 else -1
                    data_list.append(kpi_value)
        return data_list

    def get_traning_curve(self, tag1, tag2, keyworld="prim"):
        """
        get_traning_curve
        """
        if "dygraph2static" in self.step:
            print("self.step:{}".format(self.step))
            filepath_baseline = os.path.join(
                "logs/PaddleOCR/config^benchmark^icdar2015_resnet50_FPN_DBhead_polyLR/",
                "train_" + tag1 + ".log",
            )
            filepath_prim = os.path.join(
                "logs/PaddleOCR/config^benchmark^icdar2015_resnet50_FPN_DBhead_polyLR/", "train_" + tag2 + ".log"
            )

            # loss
            data_baseline = self.get_paddle_data(filepath_baseline, "loss")
            data_prime = self.get_paddle_data(filepath_prim, "loss")
            logger.info("Get data successfully!")
            self.plot_paddle_compare_value(data_baseline, data_prime, "train_loss", keyworld)
            logger.info("Plot figure successfully!")

            # hmeans
            data_baseline_hmeans = self.get_paddle_data(filepath_baseline, "hmean")
            data_prime_hmeans = self.get_paddle_data(filepath_prim, "hmean")
            logger.info("Get data successfully!")
            self.plot_paddle_compare_value(data_baseline_hmeans, data_prime_hmeans, "eval_hmeans", keyworld)
            logger.info("Plot figure successfully!")

    def build_end(self):
        """
        执行准备过程
        """
        # 进入repo中
        logger.info("build collect data  value start")
        ret = 0
        ret = self.collect_data_value()
        if ret:
            logger.info("build collect_data_value failed")
            return ret
        logger.info("build collect_data_value end")
        self.get_traning_curve("dygraph2static_baseline", "dygraph2static_prim", "prim")
        self.get_traning_curve("dygraph2static_amp", "dygraph2static_amp_prim", "amp")


def run():
    """
    执行入口
    """
    model = PaddleOCR_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
