# encoding: utf-8
"""
获取clas小数据集
"""
import os
import sys
import shutil


class PaddleClas_small_data(object):
    """
    准备小数据集
    """

    def __init__(self, data_org, data_target, num, direct_copy, extra_path, max_file, split_flag):
        """
        初始化变量
        """
        self.num = num
        self.data_org = data_org
        self.data_target = data_target
        self.direct_copy = direct_copy
        self.extra_path = extra_path
        self.max_file = max_file
        self.split_flag = split_flag

    def image_endswith_flag(self, value=None):
        """
        判断后缀
        """
        if (
            value.endswith(".JPG")
            or value.endswith(".jpg")
            or value.endswith(".jpeg")
            or value.endswith(".png")
            or value.endswith(".bmp")
            or value.endswith(".mp4")
            or value.endswith(".PNG")
        ):
            return 1
        else:
            return 0

    def txt_endswith_flag(self, value=None):
        """
        判断后缀
        """
        if value.endswith(".xml") or value.endswith(".txt"):
            return 1
        else:
            return 0

    def copy_deep_image(self, data_org, data_target):
        """
        递归从原始文件夹复制制定后缀的文件到
        """
        for index, value in enumerate(os.listdir(data_org)):  # 支持4个层级
            data_org_path = os.path.join(data_org, value)
            data_target_path = os.path.join(data_target, value)
            # print('###data_org_path', data_org_path)
            # print('###data_target_path', data_target_path)
            # print('###index', index)
            # print('###value', value)
            # input()
            if index >= self.max_file:  # 增加对文件夹个数的限制,防止数据结构归于复杂的
                break
            else:
                if self.image_endswith_flag(value):
                    if index < self.num:  # 复制50张
                        if os.path.exists(data_target_path) is True:
                            print("#### already have :", data_target_path)
                            os.remove(data_target_path)
                        shutil.copy(data_org_path, data_target_path)
                elif os.path.isdir(os.path.join(data_org, value)):
                    # print('####data_org', data_org)
                    # print('####value', value)
                    # print('####len', len(os.listdir(os.path.join(data_org,value))))
                    # input()
                    if os.path.exists(os.path.join(data_target, value)) is False:
                        os.makedirs(os.path.join(data_target, value))
                    self.copy_deep_image(os.path.join(data_org, value), os.path.join(data_target, value))
                else:
                    print("#### other data", os.path.join(data_org, value))
                # print('####data_org_path', data_org_path)
                # print('####data_target_path', data_target_path)
                # input()

    def copy_deep_txt(self, data_org, data_target):
        """
        递归从原始文件夹复制制定后缀的文件到
        """
        for index, value in enumerate(os.listdir(data_org)):  # 支持4个层级
            data_org_path = os.path.join(data_org, value)
            data_target_path = os.path.join(data_target, value)
            # print('###data_org_path', data_org_path)
            # print('###data_target_path', data_target_path)
            # print('###index', index)
            # print('###value', value)
            # input()
            if self.txt_endswith_flag(value):
                if value in self.direct_copy:
                    if os.path.exists(data_target_path) is True:
                        print("#### already have :", data_target_path)
                        os.remove(data_target_path)
                    shutil.copy(data_org_path, data_target_path)
                else:
                    f_w = open(data_target_path, "w", encoding="utf-8")
                    with open(data_org_path, "r", encoding="utf-8") as f:
                        for line in f.readlines():
                            # print('###data', os.path.join(self.data_target, self.extra_path, \
                            #   line.split(self.split_flag)[0].strip()))
                            if os.path.exists(
                                os.path.join(self.data_target, self.extra_path, line.split(self.split_flag)[0].strip())
                            ):
                                f_w.write(line)
                                # print('###data', os.path.join(self.data_target, \
                                #   self.extra_path, line.split(self.split_flag)[0].strip()))
                                # print('###line', line)
                                # print('###line', line.split(" ")[0].strip())
                                # print('####self.data_org', self.data_org)
                                # print('####self.data_target', self.data_target)
                                # print('####data_org_path', data_org_path)
                                # print('####data_target_path', data_target_path)
                                # print('###line', line.split(" ")[0].strip())
                                # print('@@@@')
                                # input()
                    f_w.close()
                    # print('@@@@')
                    # input()
            elif os.path.isdir(os.path.join(data_org, value)):
                if os.path.exists(os.path.join(data_target, value)) is False:
                    # 在这里没有文件限制，所以还是会生成所有的文件夹，但是不会复制图片
                    os.makedirs(os.path.join(data_target, value))
                self.copy_deep_txt(os.path.join(data_org, value), os.path.join(data_target, value))
            else:
                if self.image_endswith_flag(value):
                    pass
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
        self.copy_deep_image(self.data_org, self.data_target)
        self.copy_deep_txt(self.data_org, self.data_target)


# txt的处理，直接copy，或者筛选copy，都以list区分配在图片下面
# 对于图片的递归，增加文件夹个数的限制
if __name__ == "__main__":

    # VeRI-Wild
    # direct_copy = []
    # num=50
    # max_file = 1000
    # extra_path="images"
    # split_flag = " "

    # VeRi
    # direct_copy = ["list_color.txt", "list_type.txt", "test_label.xml", "train_label.xml"]
    # num=100
    # max_file = 1000
    # extra_path = ""
    # split_flag = "\t"

    # text_image_orientation
    # direct_copy = ["label_list.txt"]
    # num=100
    # max_file = 1000
    # extra_path = ""
    # split_flag = " "

    # safety_helmet
    # direct_copy = []
    # num=500
    # max_file = 1000
    # extra_path = ""
    # split_flag = " "

    # pa100k
    # direct_copy = []
    # num=500
    # max_file = 1000
    # extra_path = ""
    # split_flag = "\t"

    # CIFAR10
    # direct_copy = []
    # num=100
    # max_file = 1000
    # extra_path = ""
    # split_flag = " "

    # SOP
    # direct_copy = []
    # num=1000
    # max_file = 1000
    # extra_path = ""
    # split_flag = " "

    direct_copy = []
    num = 1000
    max_file = 1000
    extra_path = ""
    split_flag = " "

    data_org = "big_data/PaddleClas/SOP"
    data_target = "small_data_new_clas/PaddleClas/SOP"
    PaddleClas = PaddleClas_small_data(data_org, data_target, num, direct_copy, extra_path, max_file, split_flag)
    PaddleClas.run()
