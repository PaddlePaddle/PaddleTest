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
import yaml
import paddle
import pymysql
from utils.logger import logger


class DB(object):
    """DB class"""

    def __init__(self, storage="storage.yaml"):
        host, port, user, password, database = self.load_storge(storage)
        self.db = pymysql.connect(host=host, port=port, user=user, password=password, database=database, charset="utf8")
        self.cursor = self.db.cursor()

    def load_storge(self, storage):
        """
        解析storage.yaml的内容添加到self.db
        """
        with open(storage, "r") as f:
            data = yaml.safe_load(f)
        tmp_dict = data.get("PRODUCTION").get("mysql").get("api_benchmark")
        host = tmp_dict.get("host")
        port = tmp_dict.get("port")
        user = tmp_dict.get("user")
        password = tmp_dict.get("password")
        database = tmp_dict.get("db_name")
        return host, port, user, password, database

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
        """更新job状态, retry 重试保持链接"""
        retry = 3
        for i in range(retry):
            sql = "update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format(
                self.timestamp(), "saving", self.job_id
            )
            try:
                self.cursor.execute(sql)
                self.db.commit()
                logger.get_log().info('开始写入数据库: 日志位置"./log"')
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
            logger.get_log().info("数据库录入完毕")
        except Exception as e:
            logger.get_log().info("数据库录入失败")
            logger.get_log().error(e)

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

    def init_mission(
        self,
        id,
        framework,
        wheel_link,
        mode,
        place,
        cuda,
        cudnn,
        routine,
        enable_backward,
        python,
        yaml_info,
        card=None,
        comment=None,
    ):
        """init mission"""
        if framework == "paddle":
            version = paddle.__version__
            snapshot = {
                "os": platform.platform(),
                "card": card,
                "cuda": paddle.version.cuda(),
                "cudnn": paddle.version.cudnn(),
                "comment": comment,
            }
        elif framework == "torch":
            import torch

            version = torch.__version__
            snapshot = {"os": platform.platform(), "card": card, "cuda": cuda, "cudnn": cudnn, "comment": comment}

        if routine == 1 and id == 0:
            sql = (
                "insert into `job` (`framework`, `wheel_link`, `status`, `mode`, `commit`, `version`, "
                "`hostname`, `place`, `system`, `cuda`, `cudnn`, `snapshot`,`create_time`, "
                "`update_time`, `routine`, `comment`, `enable_backward`, `python`, "
                "`yaml_info`) values ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', "
                "'{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}');".format(
                    framework,
                    wheel_link,
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
                    routine,  # routine例行标记
                    comment,
                    enable_backward,
                    python,
                    yaml_info,
                )
            )
            try:
                self.cursor.execute(sql)
                self.job_id = self.db.insert_id()
                self.db.commit()
            except Exception as e:
                print(e)
        else:
            sql = (
                "update `job` set `framework`='{}', `wheel_link`='{}', `status`='{}', `mode`='{}', "
                "`commit`='{}', `version`='{}', `hostname`='{}', `place`='{}',"
                "`system`='{}', `cuda`='{}', `cudnn`='{}', `snapshot`='{}', `update_time`='{}', "
                "`routine`='{}', `comment`='{}', `enable_backward`='{}', `python`='{}', "
                "`yaml_info`='{}' where id='{}';".format(
                    framework,
                    wheel_link,
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
                    routine,
                    comment,
                    enable_backward,
                    python,
                    yaml_info,
                    id,
                )
            )
            try:
                self.cursor.execute(sql)
                self.job_id = id
                self.db.commit()
            except Exception as e:
                print(e)
