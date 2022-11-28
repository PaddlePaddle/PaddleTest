# encoding: utf-8
"""
获取gan小数据集
"""
import os
import sys
import shutil


class PaddleGAN_small_data(object):
    """
    准备小数据集
    """

    def __init__(self, data_org, data_target, num):
        """
        初始化变量
        """
        self.data_org = data_org
        self.data_target = data_target
        self.num = num

    def endswith_flag(self, value=None):
        """
        判断后缀
        """
        if (
            value.endswith(".jpg")
            or value.endswith(".jpeg")
            or value.endswith(".png")
            or value.endswith(".bmp")
            or value.endswith(".mp4")
            or value.endswith(".PNG")
        ):
            return 1
        else:
            return 0

    def copy_deep_data(self, data_org, data_target):
        """
        递归从原始文件夹复制制定后缀的文件到
        """
        for index, value in enumerate(os.listdir(data_org)):  # 支持4个层级
            data_org_path = os.path.join(data_org, value)
            data_target_path = os.path.join(data_target, value)
            if self.endswith_flag(value):
                if index < self.num:  # 复制50张
                    if os.path.exists(data_target_path) is True:
                        print("#### already have :", data_target_path)
                        os.remove(data_target_path)
                    shutil.copy(data_org_path, data_target_path)
            elif os.path.isdir(os.path.join(data_org, value)):
                if os.path.exists(os.path.join(data_target, value)) is False:
                    os.makedirs(os.path.join(data_target, value))
                self.copy_deep_data(os.path.join(data_org, value), os.path.join(data_target, value))
            else:
                print("#### other data", os.path.join(data_org, value))
            # print('####data_org_path', data_org_path)
            # print('####data_target_path', data_target_path)
            # input()

    def run(self):
        """
        执行入口
        """
        # 执行复制程序
        self.copy_deep_data(self.data_org, self.data_target)


if __name__ == "__main__":
    data_org = "big_data/PaddleGAN/RainH"
    data_target = "small_data/PaddleGAN/RainH"
    PaddleGAN = PaddleGAN_small_data(data_org, data_target, num=20)
    PaddleGAN.run()
