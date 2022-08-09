#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
logger base
"""

import logging
import os


class Logger:
    """
    logger base
    """

    def __init__(self, loggername):
        """
        initialize
        """
        # 创建一个logger
        # self.logger = logging.getLogger(loggername)
        self.logger = logging.Logger(loggername)
        self.logger.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        log_path = os.getcwd() + "/"  # 指定文件输出路径，注意logs是个文件夹，一定要加上/，不然会导致输出路径错误，把logs变成文件名的一部分了
        logname = log_path + "out.log"  # 指定输出的日志文件名
        fh = logging.FileHandler(logname, encoding="utf-8")  # 指定utf-8格式编码，避免输出的日志文本乱码
        fh.setLevel(logging.DEBUG)
        # 创建一个handler，用于将日志输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s-%(name)s-[%(levelname)s] ===> %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_log(self):
        """
        get logger object
        """
        return self.logger


logger = Logger("weaktrans")


if __name__ == "__main__":
    # test
    Logger("log").get_log().debug("Hello Logger %s" % "!")
    Logger("log1").get_log().info(
        "Hello Logger %s" % "info"
    )  # 如果不指定name则返回root对象，多次使用相同的name调用getLogger方法返回同一个logger对象。
