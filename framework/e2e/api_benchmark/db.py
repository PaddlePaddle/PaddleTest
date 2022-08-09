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
import json
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
                "insert into `case`(`jid`, `case_name`, `api`, `result`, `create_time`) "
                "values ('{}', '{}', '{}', '{}', '{}')".format(
                    self.job_id, k, json.loads(v).get("api"), v, self.timestamp()
                )
            )
            try:
                self.cursor.execute(sql)
                self.db.commit()
            except Exception as e:
                print(e)

        # 更新job状态
        sql = "update `job` set `update_time`='{}', `status`='{}' where id='{}';".format(
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

    def init_mission(self, framework, mode, place, cuda, cudnn, card=None):
        """init mission"""
        if framework == "paddle":
            version = paddle.__version__
            snapshot = {
                "os": platform.platform(),
                "card": card,
                "cuda": paddle.version.cuda(),
                "cudnn": paddle.version.cudnn(),
            }
        elif framework == "torch":
            version = torch.__version__
            snapshot = {"os": platform.platform(), "card": card}

        sql = (
            "insert into `job` (`framework`, `status`, `mode`, `commit`, `version`, "
            "`hostname`, `place`, `system`, `cuda`, `cudnn`, `snapshot`,`create_time`, "
            "`update_time`) values ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', "
            "'{}', '{}', '{}', '{}', '{}');".format(
                framework,
                "running",
                mode,
                paddle.__git_commit__,
                version,
                socket.gethostname(),
                place,
                platform.system(),
                cuda,
                cudnn,
                json.dumps(snapshot),
                self.timestamp(),
                self.timestamp(),
            )
        )
        try:
            self.cursor.execute(sql)
            self.job_id = self.db.insert_id()
            self.db.commit()
        except Exception as e:
            print(e)
