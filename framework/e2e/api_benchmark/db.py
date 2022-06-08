#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
db
"""

import os
import socket
import platform
from datetime import datetime
import paddle
import torch
import pymysql


class DB(object):
    """DB class"""

    def __init__(self):
        self.db = pymysql.connect(
            # 手动填写内容
        )
        self.cursor = self.db.cursor()

    def timestamp(self):
        """
        时间戳控制
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def reader(self, path="./log/"):
        """
        数据读取器
        """
        data = dict()
        for i in os.listdir(path):
            with open(path + i) as case:
                res = case.readline()
                api = i.split(".")[0]
                data[api] = res
        return data

    def save(self):
        """更新job状态"""
        retry = 3
        for i in range(retry):
            sql = "update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format(
                self.timestamp(), "saving", self.job_id
            )
            try:
                self.cursor.execute(sql)
                self.db.commit()
                break
            except Exception:
                # 防止超时失联
                self.db.ping(True)
                continue
        # 插入数据
        data = self.reader()
        for k, v in data.items():
            sql = (
                "insert into `cases`(`jid`, `case_name`, `result`, `create_time`) "
                "values ('{}', '{}', '{}', '{}')".format(self.job_id, k, v, self.timestamp())
            )
            try:
                self.cursor.execute(sql)
                self.db.commit()
            except Exception as e:
                print(e)

        # 更新job状态
        sql = "update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format(
            self.timestamp(), "done", self.job_id
        )
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(e)

    def error(self):
        """错误配置"""
        retry = 3
        for i in range(retry):
            sql = "update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format(
                self.timestamp(), "error", self.job_id
            )
            try:
                self.cursor.execute(sql)
                self.db.commit()
                break
            except Exception as e:
                # 防止超时失联
                self.db.ping(True)
                print(e)
                continue

    def init_mission(self, mode, place="cpu", card=None):
        """init mission"""
        sql = (
            "insert into `jobs` (`create_time`, `update_time`, `status`, `paddle_commit`, "
            "`paddle_version`, `torch_version`, `mode`, `hostname`, `place`, `card`, `system`) "
            "values ('{}','{}','{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}' );".format(
                self.timestamp(),
                self.timestamp(),
                "running",
                paddle.__git_commit__,
                paddle.__version__,
                torch.__version__,
                mode,
                socket.gethostname(),
                place,
                card,
                platform.platform(),
            )
        )
        try:
            self.cursor.execute(sql)
            self.job_id = self.db.insert_id()
            self.db.commit()
        except Exception as e:
            print(e)
