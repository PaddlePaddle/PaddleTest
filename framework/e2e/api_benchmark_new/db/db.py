#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
db object
"""

import json
import traceback
from datetime import datetime
import yaml
import pymysql

# from utils.logger import logger

ACCURACY = "%.6g"


class DB(object):
    """DB class"""

    def __init__(self, storage="storage.yaml"):
        self.storage = storage
        host, port, user, password, database = self.load_storge()
        self.db = pymysql.connect(host=host, port=port, user=user, password=password, database=database, charset="utf8")
        self.cursor = self.db.cursor()

    def load_storge(self):
        """
        解析storage.yaml的内容添加到self.db
        """
        with open(self.storage, "r") as f:
            data = yaml.safe_load(f)
        tmp_dict = data.get("Config").get("api_benchmark").get("MYSQL")
        # tmp_dict = data.get("Config").get("api_benchmark").get("DEV")
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

    def insert(self, table, data):
        """插入数据"""
        id = -1
        sql_table = "`" + table + "`"
        ls = [(k, data[k]) for k in data if data[k] is not None]
        keys = ",".join(("`" + i[0] + "`") for i in ls)
        values = ",".join("%r" % i[1] for i in ls)

        sql = "INSERT INTO {table}({keys}) VALUES ({values})".format(table=sql_table, keys=keys, values=values)
        # sql = 'insert %s (' % table + ','.join(('`' + i[0] + '`') for i in ls) + \
        #       ') values (' + ','.join('%r' % i[1] for i in ls) + ')'
        try:
            self.cursor.execute(sql)
            id = self.db.insert_id()
            self.db.commit()
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        return id

    def update(self, table, data, data_condition):
        """按照data_condition 更新数据"""
        sql_table = "`" + table + "`"
        sql = (
            "UPDATE %s SET " % sql_table
            + ",".join("%s=%r" % (("`" + k + "`"), data[k]) for k in data)
            + " WHERE "
            + " AND ".join("%s=%r" % (("`" + k + "`"), data_condition[k]) for k in data_condition)
        )

        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(traceback.format_exc())
            print(e)

    def update_by_id(self, table, data, id):
        """按照id 更新数据"""
        sql_table = "`" + table + "`"
        sql = (
            "UPDATE %s SET " % sql_table
            + ",".join("%s=%r" % (("`" + k + "`"), data[k]) for k in data)
            + " WHERE "
            + "%s=%r" % ("`id`", id)
        )

        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(traceback.format_exc())
            print(e)

    def select(self, table, condition_list):
        """按照condition_list 查询数据"""
        results = []
        sql_table = "`" + table + "`"
        sql = "SELECT * FROM %s " % sql_table + " WHERE " + " AND ".join("%s" % k for k in condition_list)

        try:
            self.cursor.execute(sql)
            res = self.cursor.fetchall()

            index_list = self.show_list(table=table)
            for row in res:
                tmp = {}
                for i, v in enumerate(row):
                    tmp[index_list[i]] = row[i]
                results.append(tmp)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        return results

    def select_by_id(self, table, id):
        """按照id 查询数据"""
        results = []
        sql_table = "`" + table + "`"
        sql = "SELECT * FROM %s " % sql_table + " WHERE " + "%s=%r" % ("`id`", id)

        try:
            self.cursor.execute(sql)
            res = self.cursor.fetchall()

            tmp = {}
            index_list = self.show_list(table=table)
            for row in res:
                for i, v in enumerate(row):
                    tmp[index_list[i]] = row[i]
                results.append(tmp)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        return results

    def insert_case(self, jid, data_dict, create_time):
        """向case表中录入数据"""
        data_dict["result"]["forward"] = ACCURACY % data_dict["result"]["forward"]
        data_dict["result"]["total"] = ACCURACY % data_dict["result"]["total"]
        data_dict["result"]["backward"] = ACCURACY % data_dict["result"]["backward"]
        data_dict["result"]["best_total"] = ACCURACY % data_dict["result"]["best_total"]
        data = {
            "jid": jid,
            "case_name": data_dict["case_name"],
            "api": data_dict["result"]["api"],
            "result": json.dumps(data_dict["result"]),
            "create_time": create_time,
        }
        retry = 3
        # case_id = 0
        for i in range(retry):
            case_id = self.insert(table="case", data=data)
            # self.insert_case(jid=job_id, data_dict=case, create_time=create_time)
            if case_id == -1:
                print("db ping again~~~")
                self.db.ping(True)
                continue
            else:
                break
            # # self.ci_update_job(id=job_id, status="error", update_time=time_now)
            # print(traceback.format_exc())
            # print(e)
            # # 防止超时失联
            # self.db.ping(True)
            # continue

        # return case_id

    def show_list(self, table):
        """返回table中的列list"""
        results = []
        sql_table = "`" + table + "`"
        sql = "SHOW COLUMNS from {}".format(sql_table)
        try:
            self.cursor.execute(sql)
            results = [column[0] for column in self.cursor.fetchall()]
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        return results


if __name__ == "__main__":
    # db = DB(storage="storage.yaml")
    db = DB(storage="storage.yaml")

    # table = 'job'
    # data = {
    #     'framework': 'paddle', 'commit': 'aaabbbccc',
    #     'system': 'Darwin', 'cuda': '1121', 'cudnn': '2233',
    #     'update_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # }
    # id = db.insert(table='job', data=data)
    # print('id is: ', id)

    # table = 'job'
    # data = {
    #     'framework': 'paddle11', 'commit': '132aaabbbccc',
    #     'system': 'Darwin', 'cuda': '1121', 'cudnn': '2233',
    #     'update_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # }
    # data_condition = {
    #     'id': '60'
    # }
    # db.update(table='job', data=data, data_condition=data_condition)

    # table = 'job'
    # data = {
    #     'framework': 'paddle22', 'commit': '33323aaabbbccc',
    #     'system': 'Darwin', 'cuda': '1121', 'cudnn': '2233',
    #     'update_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # }
    # id = 60
    # db.update_by_id(table='job', data=data, id=id)

    # table = 'job'
    # condition_list = ['id < 60', 'framework = "paddle11"']
    # res = db.select(table='job', condition_list=condition_list)
    # print('res is: ', res)

    table = "case"
    condition_list = ["jid = 71"]
    res = db.select(table=table, condition_list=condition_list)
    print("res is: ", res)

    # table = "job"
    # id = 56
    # res = db.select_by_id(table="job", id=id)
    # print("res is: ", res)

    # table = "job"
    # res = db.show_list(table="job")
    # print("res is: ", res)
