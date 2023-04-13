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
    show_type = ["train avg loss", "eval avg loss", "eval metric(top1 acc)", "avg ips"]
    # show_type = ['train avg loss','eval metric(top1 acc)','eval avg loss','lr','avg ips']
    # show_type = ['train avg loss']
    for show in show_type:
        plt.subplot(1, len(show_type), i)  # 表示第i张图片，下标只能从1开始，不能从0，
        task_type = log_list
        # task_type=['train_dy2st_prim.log', 'train_dy2st.log', 'train_dy2st_prim_cinn.log', 'train_dy2st_cinn.log']
        for task in task_type:
            fp = open(task)
            temp = []
            for line in fp.readlines():
                if show == "train avg loss":
                    if "Eval" not in line and "Avg" in line and "Epoch" in line and "logger" not in line:
                        try:
                            temp.append(float(re.compile(r"(?<=loss: )\d+\.?\d*").findall(line)[0]))
                        except:
                            temp.append(0)
                if show == "lr":
                    if "lr" in line and "Epoch" in line and "logger" not in line:
                        temp.append(float(re.compile(r"(?<=lr: )\d+\.?\d*").findall(line)[0]))
                # if show == 'eval loss':
                #     if 'loss: ' in line and 'Train' not in line and 'Avg' not in line and 'Epoch'  in line:
                #         temp.append(float(re.findall(r"loss: (.+?),",line)[0]))
                if show == "eval metric(top1 acc)":
                    if "metric:" in line and "final" not in line and "Epoch" in line and "logger" not in line:
                        temp.append(float(re.compile(r"(?<=metric: )\d+\.?\d*").findall(line)[0]))
                if show == "eval avg loss":
                    if "Avg" in line and "Train" not in line and "Epoch" in line and "logger" not in line:
                        temp.append(float(re.findall(r"loss: (.+?),", line)[0]))
                if show == "avg ips":
                    if "ips" in line and "Epoch" in line and "logger" not in line:
                        temp.append(float(re.compile(r"(?<=ips: )\d+\.?\d*").findall(line)[0]))
            fp.close()
            if show == "avg ips":
                try:
                    tmp = np.mean(np.array(temp[:]))
                except:
                    logger.info("avg ips err")
                for j, val in enumerate(temp):
                    temp[j] = tmp
                logger.info("####  task is {}".format(task))
                logger.info("####  avg ips is {}".format(tmp))
            if show == "eval metric(top1 acc)":
                logger.info("####  task is {}".format(task))
                try:
                    tmp = np.max(np.array(temp[:]))
                    logger.info("####  eval metric(top1 acc) is {}".format(tmp))
                except:
                    logger.info("eval metric(top1 acc) err")

            plt.plot(temp[:], label=task.split("/")[-1])
            plt.legend()  # 显示图例
            if show == "lr" or show == "avg ips":
                plt.xlabel("iter")
            else:
                plt.xlabel("epoch")
            plt.title(show)
        i += 1
    # plt.show()
    plt.suptitle(model_name)
    plt.savefig(model_name)


if __name__ == "__main__":
    # plt_dy2st(log_list, model_name)
    plt_dy2st(["./train_dy2st_prim.log", "./train_dy2st.log"], "CAE")
