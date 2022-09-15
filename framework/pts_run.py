#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
多线程执行器
"""
import sys
import os
import queue
import time
import platform
import threading
import argparse
import wget


class Erwin(object):
    """
    多线程执行器
    """

    def __init__(self, case_dict, thread_num=2):
        self.case_queue = queue.Queue()
        self.thread_pool = self.create_pool(thread_num)
        self.report_dir = os.sep.join([self.get_cur_dir(), "report"])
        self.case_dict = case_dict
        self.ignore_list = self.get_ignore_list()
        self.case_list()

    def get_ignore_list(self):
        """
        忽略Case集合
        """
        return dict({})

    def get_cur_dir(self):
        """
        获取当前执行目录
        """
        dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
        return dirname

    def case_list(self):
        """
        获取case执行列表
        """
        for case_dir, modules_list in self.case_dict.items():
            for module in modules_list:
                path = os.sep.join([case_dir, module])
                for case in os.listdir(path):
                    if case.startswith("test") and case.endswith("py") and case not in self.ignore_list:
                        case_info = dict({"path": path, "case": case, "case_dir": case_dir})
                        self.case_queue.put(case_info)

    def run(self):
        """
        执行函数
        """
        for p in self.thread_pool:
            p.start()
        self.case_queue.join()

    def runner(self):
        """
        线程执行器
        """
        while True:
            case_info = self.case_queue.get()
            self.run_test(case_info)
            self.case_queue.task_done()

    def run_test(self, case_info):
        """
        单个case执行
        """
        path = case_info["path"]
        case = case_info["case"]
        print("case: {}".format(case))
        # TODO 执行case
        if case_info["case_dir"] == "api":
            dirname = "api"
        else:
            dirname = path.split(os.sep)[-1]
        if platform.system() == "Windows":
            os.system(
                "cd {} && python.exe -m pytest {} --alluredir={}".format(path, case, self.report_dir + os.sep + dirname)
            )
        else:
            os.system(
                "cd {} && {} -m pytest {} --alluredir={}".format(
                    path, args.interpreter, case, self.report_dir + os.sep + dirname
                )
            )

    def create_pool(self, thread_num):
        """
        池子
        """
        thread_pool = []
        for i in range(thread_num):
            thread = threading.Thread(target=self.runner)
            thread.daemon = True
            thread_pool.append(thread)
        return thread_pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module", type=str, required=True, help="choose module -> op_function | jit | external_api_function"
    )
    parser.add_argument("--interpreter", type=str, help="python interpreter", required=True)
    args = parser.parse_args()

    # prepare env
    os.environ["FLAGS_call_stack_level"] = "2"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["FLAGS_USE_STANDALONE_EXECUTOR"] = "1"
    os.environ["FLAGS_CONVERT_GRAPH_TO_PROGRAM"] = "1"
    # download allure and unzip
    # wget.download("https://paddle-qa.cdn.bcebos.com/PTS/allure-2.17.3.tgz")
    # os.system("mkdir allure && tar -xf allure-2.17.3.tgz -C allure --strip-components 1")

    start = time.time()
    if args.module == "op_function":
        case_dict = {
            "api": [
                "linalg",
                "nn",
                "optimizer",
                "loss",
                "incubate",
                "fft",
                "device",
                "paddlebase",
                "distribution",
                "utils",
            ]
        }
        worker = Erwin(case_dict, thread_num=4)
        worker.run()
    elif args.module == "jit":
        case_dict = {"e2e": ["jit"]}
        worker = Erwin(case_dict, thread_num=4)
        worker.run()
    elif args.module == "external_api_function":
        case_dict = {"e2e": ["custom_op"]}
        worker = Erwin(case_dict, thread_num=1)
        worker.run()
    end = time.time()
    print("running time: {} s".format(end - start))
