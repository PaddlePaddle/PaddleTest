# encoding: utf-8
"""
分析对比绘图
"""
import re
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("ce")


def plot_loss_from_log_files(log_file_names, image_file_name):
    """
    用于绘制loss对比图
    """
    # 初始化列表
    losses_list = []  # 用于存储每个 log 文件中的训练损失
    labels = []  # 用于存储每个 log 文件的名称

    # 遍历每个 log 文件
    for log_file_name in log_file_names:
        # 读取 log 文件的内容
        with open(log_file_name, "r") as f:
            lines = f.read().splitlines()
            losses = []

        # 解析 log 文件中的每一行
        for line in lines:
            if "[Train][Epoch" in line and "[Avg]" in line:
                # 如果是带有损失值的行，则提取损失值并添加到列表中
                items = line.split()
                for i, arg in enumerate(items):
                    if arg == "loss:":
                        loss = items[i + 1][:-1]  # 提取损失值（去掉最后的逗号）
                        losses.append(float(loss))

        # 将每个 log 文件中的损失值列表和文件名保存到相应的列表中
        losses_list.append(losses)
        labels.append(os.path.basename(log_file_name))

    # 绘制损失随 epoch 变化的曲线
    for i, losses in enumerate(losses_list):
        plt.plot(losses, label=labels[i])

    # 添加图例
    plt.legend()

    # 保存图像文件
    plt.savefig(os.path.join(".", f"{image_file_name}.png"))


if __name__ == "__main__":
    # plt_dy2st(log_list, model_name)
    plot_loss_from_log_files(["test_ldc2d_unsteady_Re10_cuda11.6.log", "train_single.log"], "test")
