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
import tarfile
import yaml
import wget
import paddle
import allure
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("ce")


class Paddle3D_End(object):
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

    def getdata1(self, filename, delimiter1, delimiter2):
        """
        get_data
        """
        data = []
        with open(filename, "rt") as f:
            for line in f:
                pattern = re.compile(delimiter1 + "(.+)" + delimiter2)
                #          pattern=re.compile('loss:(.+), ')
                result = pattern.findall(line)
                if len(result) > 0:
                    # print(float(result[0]))
                    data.append(float(result[0]))
        return data[-1]

    def getdata(self, filename, kpi):
        """
        get_data
        """
        kpi_value = -1
        f = open(filename, encoding="utf-8", errors="ignore")
        delimiter_list = [":", "=", " "]
        for line in f.readlines():
            # if kpi + ":" in line:
            if any(kpi + delimiter in line for delimiter in delimiter_list):
                # regexp = r"%s:(\s*\d+(?:\.\d+)?)" % kpi
                regexp = r"%s[:= ](\s*\d+(?:\.\d+)?)" % kpi
                r = re.findall(regexp, line)
                # 如果解析不到符合格式到指标，默认值设置为-1
                kpi_value = float(r[0].strip()) if len(r) > 0 else -1
        f.close()
        return kpi_value

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
            # train loss
            # if self.category == "det":
            #    train_loss = self.getdata(self.TRAIN_LOG_PATH, "loss:", ", loss_shrink_maps")
            # elif self.category == "table":
            #    train_loss = self.getdata(self.TRAIN_LOG_PATH, "loss:", ", horizon_bbox_loss")
            # else:
            #    train_loss = self.getdata(self.TRAIN_LOG_PATH, "loss:", ", avg_reader_cost")
            # parse kpi
            train_loss = self.getdata(self.TRAIN_LOG_PATH, "total_loss")
            logger.info("#### train_loss: {}".format(train_loss))
            self.update_json("tools/train.json", train_loss)
        elif self.step == "eval" and os.environ.get("UPDATA_BASE_VALUE") is True:
            # eval acc
            # kiit
            if (
                self.category == "smoke"
                or self.model == "pointpillars_xyres16_kitti_cyclist_pedestrian"
                or self.model == "centerpoint_pillars_016voxel_kitti"
            ):
                eval_acc = self.getdata(self.EVAL_LOG_PATH, "AP_R11@25%")
            elif self.model == "pointpillars_xyres16_kitti_car":
                eval_acc = self.getdata(self.EVAL_LOG_PATH, "AP_R11@50%")
            # nuscenes dataset
            elif self.category == "centerpoint":
                eval_acc = self.getdata(self.EVAL_LOG_PATH, "mAP")
            elif self.category == "squeezesegv3":
                eval_acc = self.getdata(self.EVAL_LOG_PATH, "Acc avg")
            else:
                pass
            logger.info("#### eval_acc: {}".format(eval_acc))
            self.update_json("tools/eval.json", eval_acc)

    def plot_paddle_compare_value(self, data1, data2, data3, data4, value):
        """
        plot_paddle_compare_value
        """

        ydata1 = data1
        xdata1 = list(range(0, len(ydata1)))

        ydata2 = data2
        xdata2 = list(range(0, len(ydata2)))

        ydata3 = data3
        xdata3 = list(range(0, len(ydata3)))

        ydata4 = data4
        xdata4 = list(range(0, len(ydata4)))

        # plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xdata1, ydata1, label="paddle_amp_" + value, color="b", linewidth=1)
        ax.plot(xdata2, ydata2, label="paddle_dygraph2static_amp_" + value, color="r", linewidth=1)
        ax.plot(xdata3, ydata3, label="paddle_dygraph2static_amp_prim_" + value, color="g", linewidth=1)
        ax.plot(xdata4, ydata4, label="paddle_dygraph2static_amp_prim_cinn_" + value, color="y", linewidth=1)

        # set the legend
        ax.legend()
        # set the limits
        ax.set_xlim([0, len(xdata1)])
        if "loss" in value:
            ax.set_ylim([0, math.ceil(np.percentile(ydata1, 99.99))])
        else:
            ax.set_ylim([0, math.ceil(max(ydata1))])
        if "loss" in value:
            ax.set_xlabel("step/100 iters")
        else:
            ax.set_xlabel("epoch")
        ax.set_ylabel(value)
        ax.grid()
        ax.set_title("Paddle3D_PETRV2")

        # display the plot
        # plt.show()
        if not os.path.exists("picture"):
            os.makedirs("picture")
        plt.savefig("picture/" + "Paddle3D_PETRV2_" + value + ".png")

    def get_paddle_data(self, filepath, kpi):
        """
        get_paddle_data(
        """
        data_list = []
        f = open(filepath, encoding="utf-8", errors="ignore")
        i = 0
        for line in f.readlines():
            if kpi + "=" in line and line.startswith("20") and i % 10 == 0:
                if "current" in line:
                    pass
                else:
                    regexp = r"%s=(\s*\d+(?:\.\d+)?)" % kpi
                    r = re.findall(regexp, line)
                    # 如果解析不到符合格式到指标，默认值设置为-1
                    kpi_value = float(r[0].strip()) if len(r) > 0 else -1
                    data_list.append(kpi_value)
            i = i + 1
        return data_list

    def get_traning_curve(self):
        """
        get_traning_curve
        """
        if "dygraph2static_amp_prim_cinn" in self.step:
            print("self.step:{}".format(self.step))
            print("os.getcwd():{}".format(os.getcwd()))
            if not os.path.exists("logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp"):
                os.makedirs("logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp")

            wget.download(
                "https://paddle-qa.bj.bcebos.com/logs/Paddle3D/\
configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp/train_amp.log",
                "logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp",
            )
            wget.download(
                "https://paddle-qa.bj.bcebos.com/logs/Paddle3D/\
configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp/train_dygraph2static_amp.log",
                "logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp",
            )
            wget.download(
                "https://paddle-qa.bj.bcebos.com/logs/Paddle3D/\
configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp/train_dygraph2static_amp_prim.log",
                "logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp",
            )
            #             wget.download(
            #                 "https://paddle-qa.bj.bcebos.com/logs/Paddle3D/\
            # configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp/train_dygraph2static_amp_prim_cinn.log",
            #                 "logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp",
            #             )

            filepath_amp = os.path.join(
                "logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp/",
                "train_amp.log",
            )
            filepath_dygraph2static_amp = os.path.join(
                "logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp/",
                "train_dygraph2static_amp.log",
            )
            filepath_dygraph2static_amp_prim = os.path.join(
                "logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp/",
                "train_dygraph2static_amp_prim.log",
            )
            filepath_dygraph2static_amp_cinn = os.path.join(
                "logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp/",
                "train_dygraph2static_amp_prim_cinn.log",
            )

            # total_loss
            data_amp = self.get_paddle_data(filepath_amp, "total_loss")

            data_dygraph2static_amp = self.get_paddle_data(filepath_dygraph2static_amp, "total_loss")

            data_dygraph2static_amp_prim = self.get_paddle_data(filepath_dygraph2static_amp_prim, "total_loss")

            data_dygraph2static_amp_prim_cinn = self.get_paddle_data(filepath_dygraph2static_amp_cinn, "total_loss")

            logger.info("Get data successfully!")
            self.plot_paddle_compare_value(
                data_amp,
                data_dygraph2static_amp,
                data_dygraph2static_amp_prim,
                data_dygraph2static_amp_prim_cinn,
                "train_loss",
            )
            logger.info("Plot figure successfully!")
            # log
            shutil.copytree(
                "logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp", "picture", dirs_exist_ok=True
            )
        else:
            if os.path.exists("bos_new.tar.gz") is False and os.getenv("bce_whl_url"):
                bce_whl_url = os.getenv("bce_whl_url")
                wget.download(bce_whl_url)
                tf = tarfile.open("bos_new.tar.gz")
                tf.extractall(os.getcwd())
            log_name = os.listdir("logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp")[0]
            log_path = os.path.join("logs/Paddle3D/configs^petr^petrv2_vovnet_gridmask_p4_800x320_dn_amp", log_name)
            cmd = "python BosClient.py %s paddle-qa/" % (log_path)
            os.system(cmd)

            # # hmeans
            # data_baseline_hmeans = self.get_paddle_data(filepath_baseline, "hmean")
            # data_prime_hmeans = self.get_paddle_data(filepath_prim, "hmean")
            # logger.info("Get data successfully!")
            # self.plot_paddle_compare_value(data_baseline_hmeans, data_prime_hmeans, "eval_hmeans", keyworld)
            # logger.info("Plot figure successfully!")

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
        if "amp" in self.step:
            self.get_traning_curve()


def run():
    """
    执行入口
    """
    print("This is Paddle3D_End start!")
    model = Paddle3D_End()
    model.build_end()
    return 0


if __name__ == "__main__":
    run()
