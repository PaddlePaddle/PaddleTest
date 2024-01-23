#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

##
# -----------------------------------------------------------------------------
# 执行分布式策略组合组件
# 执行shell 命令组件 ，或者是使用launch 模块
# 结果校验器，正则匹配组件

import os
import re
import shutil
import subprocess
import numpy


class Runner(object):
    """
    执行器
    初始化环境
    配置策略，case组合
    启动测试进程
    校验结果
    """
    def __init__(self,  case_file, gpus, expect, checker=None, python_interpreter="python", logdir="./test_log",):
        """
        初始化函数，用于创建测试对象

        Args:
            case_file (str): 测试用例文件路径
            gpus (str): 使用的GPU编号，例如"0,1,2,3"
            expect : 测试期望结果，用于断言
            checker (Optional[Callable]): 自定义的校验器，主要是为了自定义正则表达式
            python_interpreter (str): Python解释器路径，默认为"python"
            logdir (str): 日志输出路径，默认为"./test_log"

        Returns:
            None
        """
        """
        checker: 自定义的校验器,主要是为了自定义正则表达式
        """
        self.logdir = logdir
        self.shell_script = f"{python_interpreter} -m paddle.distributed.launch --gpus={gpus} --log_dir {logdir} {case_file}"
        self.expect = expect
        self.python_interpreter = python_interpreter
        self.checker = checker
        Initializer(logdir=self.logdir)
    def run(self):
        l = Launcher(self.shell_script)
        l.launch()
        if self.checker is None:
            c = Checker(self.expect)
        else:
            c = self.checker
        c.check()



class Initializer(object):
    """
    初始化模块
    """
    def __init__(self, logdir="./test_log"):
        self.logdir = logdir
        self.initialize_logdir()

    def initialize_logdir(self):
        """
        初始化日志目录
        """
        if os.path.exists(self.logdir):
            # 如果目录存在，删除并重新创建
            shutil.rmtree(self.logdir)
            os.makedirs(self.logdir)
        else:
            # 如果目录不存在，直接创建
            os.makedirs(self.logdir)


class Launcher(object):
    """
    执行shell 命令组件
    """

    def __init__(self, command):
        self.command = command

    def launch(self):
        try:
            print(f"Executing shell command: {self.command}")
            result = subprocess.run(self.command, shell=True, check=True, capture_output=True, text=True)
            print(f"Shell command output:\n{result.stdout}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error executing shell command: {e}")
            raise e


class Checker(object):
    def __init__(self, result, logdir="./test_log", pattern=r'loss is: (\d+\.\d+)'):
        self.expected_result = result
        self.logdir = logdir
        self.pattern = pattern

    def check(self):
        """
        校验模块
        """
        if not isinstance(self.expected_result, dict):
            raise TypeError("result must be a dict")
            # 检查logdir是否存在
        if not os.path.exists(self.logdir):
            raise FileNotFoundError(f"Log directory '{self.logdir}' not found")
        else:
            print(f"Log directory '{self.logdir}' exists")
        # 检查result是否包含多卡信息
        for key in self.expected_result.keys():
            if not key.startswith("gpu") or not key[3:].isdigit():
                raise ValueError(f"Invalid key in result: {key}")

        # 处理日志信息
        result = self.regex_check(gpu_nums=len(self.expected_result.keys()))
        self.compare(result, self.expected_result)



    def regex_check(self, gpu_nums=0):
        """
        正则匹配校验器, 可以独立使用
        """
        result = dict()
        if gpu_nums <= 0:
            raise ValueError("gpu_nums must be positive")
        # 遍历gpu_nums的数量，对应log文件
        for i in range(gpu_nums):
            result["gpu" + str(i)] = []
            log_file_path = os.path.join(self.logdir, f"workerlog.{i}")
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as file:
                    log = file.read()
                    # 使用正则表达式提取loss的值
                    # 使用正则表达式提取所有loss的值
                    matches = re.findall(self.pattern, log)

                    if matches:
                        for match in matches:
                            loss_value = float(match)
                            result["gpu" + str(i)].append(loss_value)
                            # print(f"Extracted loss value from {log_file_path}: {loss_value}")
                    else:
                        print(f"No match found in {log_file_path}")
                        raise ValueError("No match found in log file")
        return result

    def compare(self, result, expect):
        """
        比较结果
        """
        if isinstance(result, dict) and isinstance(expect, dict):
            for key in result.keys():
                print(f"checking {key} result ...")
                numpy.testing.assert_equal(result.get(key), expect.get(key))
                print("pass")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Distributed Testing Script")
    # parser.add_argument("--gpus", default="1", help="GPUs to use gpu_number")
    # args = parser.parse_args()
    #######获取日志信息，用来设置expect######
    # c = Checker(1)
    # res = c.regex_check(args.gpus)
    # print(res)
    ####################################
    pass