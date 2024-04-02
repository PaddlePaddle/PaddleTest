# encoding: utf-8
"""
分析对比绘图
"""
import re
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger("ce")


def plt_dy2st(log_list, model_name):
    """
    分析对比绘图
    """
    plt.figure(figsize=(16, 9))
    i = 1
    show_type = ["G_idt_A_loss", "G_idt_B_loss", "avg ips"]
    for show in show_type:
        plt.subplot(1, len(show_type), i)  # 表示第i张图片，下标只能从1开始，不能从0，
        task_type = log_list
        # task_type = ["cycle.log", "cycle_2.log"]
        for task in task_type:
            fp = open(task)
            temp = []
            for line in fp.readlines():
                if show == "train loss":
                    if "loss_pixel" in line:
                        temp.append(float(re.findall(r"loss_pixel: (.+?) ", line)[0]))
                if show == "G_idt_A_loss":
                    if "G_idt_A_loss" in line:
                        temp.append(float(re.findall(r"G_idt_A_loss: (.+?) ", line)[0]))
                if show == "G_idt_B_loss":
                    if "G_idt_B_loss" in line:
                        temp.append(float(re.findall(r"G_idt_B_loss: (.+?) ", line)[0]))
                if show == "psnr":
                    if "Metric psnr:" in line:
                        temp.append(float(re.compile(r"(?<=psnr: )\d+\.?\d*").findall(line)[0]))
                if show == "ssim":
                    if "Metric ssim:" in line:
                        temp.append(float(re.compile(r"(?<=ssim: )\d+\.?\d*").findall(line)[0]))
                if show == "avg ips":
                    if "ips" in line and "Epoch" in line and "logger" not in line:
                        temp.append(float(re.compile(r"(?<=ips: )\d+\.?\d*").findall(line)[0]))
            fp.close()
            if show == "avg ips":
                tmp = np.mean(np.array(temp[:]))
                for j, val in enumerate(temp):
                    temp[j] = tmp
                logger.info("####task is {}".format(task))
                logger.info("####avg ips is {}".format(tmp))

            plt.plot(temp[:], label=task.split("/")[-1])
            plt.legend()  # 显示图例

            if "loss" in show or "ips" in show:
                plt.xlabel("iter")
            else:
                plt.xlabel("epoch")
            plt.title(show)
        i += 1
    plt.suptitle(model_name)
    plt.savefig(model_name)


if __name__ == "__main__":
    # plt_dy2st(log_list, model_name)
    plt_dy2st(["./train_dy2st_prim.log", "./train_dy2st.log"], "cyclegan")
