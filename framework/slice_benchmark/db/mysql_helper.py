#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import pymysql
import time

class MySQLHelper(object):
    """
    MySQLHelper基类，用于与MySQL数据库交互的基本功能
    """
    def __init__(self, host, user, password, db_name, port):
        # super().__init__(logger)
        self.host = host
        self.user = user
        self.password = password
        self.db_name = db_name
        self.port = port
        self.conn = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.db_name, port=self.port)
        self.cursor = self.conn.cursor()

    def execute_query(self, query):
        try:
            self.cursor.execute(query)
            columns = [col[0] for col in self.cursor.description]
            result = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            return result
        except Exception as e:
            print(f"执行查询时出错：{e}")

    def execute_update(self, query):
        try:
            self.cursor.execute(query)
            self.conn.commit()
            print("Update successful")
        except Exception as e:
            print(f"Error updating:{e}")
            self.conn.rollback()


    def execute_insert(self, query):
        try:
            self.cursor.execute(query)
            self.conn.commit()
            # print("Insert successful")
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting:{e}")

    def execute_delete(self, query):
        try:
            self.cursor.execute(query)
            self.conn.commit()
            print("Delete successful")
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting:{e}")

    def __del__(self):
        self.cursor.close()
        self.conn.close()


class DB(object):
    """
    DB基类，用于与数据库交互的基本功能
    """
    @classmethod
    def init_db(cls, host, user, password, db_name, port):
        """
        初始化数据库连接
        :param host: 数据库主机地址
        :param user: 数据库用户名
        :param password: 数据库密码
        :param db_name: 数据库名称
        """
        cls.db = MySQLHelper(host=host, user=user, password=password, db_name=db_name, port=port)

    @classmethod
    def select(cls):
        """
        执行SELECT查询操作
        :return: 查询结果
        """
        query = f"SELECT * FROM {cls.table}"
        result = cls.db.execute_query(query)
        return result

    @classmethod
    def select_by_id(cls, id):
        """
        根据ID执行SELECT查询操作
        :param id: 要查询的ID
        :return: 查询结果
        """
        query = f"SELECT * FROM {cls.table} WHERE id = {id}"
        result = cls.db.execute_query(query)
        return result
    
    @classmethod
    def insert(cls, data):
        """
        执行INSERT操作
        :param data: 插入的数据，字典形式
        :return: 插入数据的ID
        """
        columns = ', '.join(data.keys())
        values = ', '.join([f"'{value}'" for value in data.values()])
        query = f"INSERT INTO {cls.table} ({columns}) VALUES ({values})"
        try:
            cls.db.execute_insert(query)
            last_id_query = "SELECT LAST_INSERT_ID()"
            last_id = cls.db.execute_query(last_id_query)[0]['LAST_INSERT_ID()']
            return last_id
        except:
            return None

    @classmethod
    def update(cls, data, condition):
        """
        执行UPDATE操作
        :param data: 更新的数据，字典形式
        :param condition: 更新条件
        :return: 更新操作影响的行数
        """
        set_values = ', '.join([f"{key}='{value}'" for key, value in data.items()])
        query = f"UPDATE {cls.table} SET {set_values} WHERE {condition}"
        try:
            cls.db.execute_update(query)
            affected_rows = cls.db.cursor.rowcount
            return affected_rows
        except:
            return -1

    @classmethod
    def delete(cls, condition):
        """
        执行DELETE操作
        :param condition: 删除条件
        :return: 删除操作影响的行数
        """
        query = f"DELETE FROM {cls.table} WHERE {condition}"
        try:
            cls.db.execute_delete(query)
            affected_rows = cls.db.cursor.rowcount
            return affected_rows
        except:
            return -1


class SliceBenchmarkDB(object):
    """
    db
    """
    def __init__(self, host, user, password, db_name, port):
        """
        初始化数据库连接
        :param host: 数据库主机地址
        :param user: 数据库用户名
        :param password: 数据库密码
        :param db_name: 数据库名称
        :param port: 数据库端口
        :param max_retries: 最大重试次数
        """
        self.host = host
        self.user = user
        self.password = password
        self.db_name = db_name
        self.port = port
        self.db = None
        self.max_retries = 3
        self.connect_db()

    def connect_db(self):
        """
        建立数据库连接
        """
        self.db = MySQLHelper(host=self.host, user=self.user, password=self.password, db_name=self.db_name,
                              port=self.port)

    def execute_with_reconnect(self, query, query_type):
        """
        执行数据库操作，失败时重试
        :param query: SQL 查询
        :param query_type: 查询类型 (select, insert, update, delete)
        :return: 查询结果或操作影响的行数
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if query_type == 'select':
                    return self.db.execute_query(query)
                elif query_type == 'insert':
                    self.db.execute_insert(query)
                    last_id_query = "SELECT LAST_INSERT_ID()"
                    return self.db.execute_query(last_id_query)[0]['LAST_INSERT_ID()']
                elif query_type == 'update' or query_type == 'delete':
                    self.db.execute_update(query) if query_type == 'update' else self.db.execute_delete(query)
                    return self.db.cursor.rowcount
            except Exception as e:
                print(f"Error executing {query_type} query: {e}")
                retries += 1
                time.sleep(2)  # 等待 2 秒后重试
                print(f"等待2秒后重新建立链接, 第{retries}次尝试")
                self.connect_db()  # 重新建立连接
        return -1

    def select(self, table):
        query = f"SELECT * FROM {table}"
        return self.execute_with_reconnect(query, 'select')

    def select_by_id(self, table, id):
        query = f"SELECT * FROM {table} WHERE id = {id}"
        return self.execute_with_reconnect(query, 'select')
    
    def select_by_tid(self, table, tid):
        """
        根据tid查询数据
        """
        query = f"SELECT * FROM {table} WHERE tid = {tid}"
        return self.execute_with_reconnect(query, 'select')
    
    def select_by_desc(self, table, description, status):
        """
        根据description和status查询数据
        """
        query = f"SELECT * FROM {table} WHERE description = '{description}' AND status = '{status}'"
        return self.execute_with_reconnect(query, 'select')
    
    def select_by_condition(self, table, condition):
        """
        根据ID执行SELECT查询操作
        :param id: 要查询的ID
        :return: 查询结果
        """
        query = f"SELECT * FROM {table} WHERE {condition}"
        return self.execute_with_reconnect(query, 'select')

    def insert(self, table, data):
        columns = ', '.join(data.keys())
        values = ', '.join([f"'{value}'" for value in data.values()])
        query = f"INSERT INTO {table} ({columns}) VALUES ({values})"
        return self.execute_with_reconnect(query, 'insert')

    def update(self, table, data, condition):
        set_values = ', '.join([f"{key}='{value}'" for key, value in data.items()])
        query = f"UPDATE {table} SET {set_values} WHERE {condition}"
        return self.execute_with_reconnect(query, 'update')

    def delete(self, table, condition):
        query = f"DELETE FROM {table} WHERE {condition}"
        return self.execute_with_reconnect(query, 'delete')


# Example usage:
if __name__ == "__main__":
    import yaml
    db_config = "/paddle/baidu/paddle/PTSTools/Uploader/apibm_config.yml"
    with open(db_config, encoding="utf-8") as f:
        db_config = yaml.load(f, Loader=yaml.FullLoader)
    print(db_config)
    # exit(0)
    db = SliceBenchmarkDB(**db_config['Config']['slice_benchmark']['MYSQL'])
    # exit(0)
    # Insert example
    env_info = "环境示例"
    data = {"comment": "调试slice demo任务", "env_info": "环境示例", "status": "running", "commit": "aaaaaaaaaaabbbbbbb", }
    db.insert(table='slice_job', data=data)

    # Update example
    # update_query = "UPDATE table_name SET column1 = 'value1' WHERE id = 1"
    # db.execute_update(update_query)
    #
    # # Insert example
    # insert_query = "INSERT INTO slice_job (column1, column2) VALUES ('value1', 'value2')"
    # db.execute_insert(insert_query)
    #
    # # Delete example
    # delete_query = "DELETE FROM table_name WHERE id = 1"
    # db.execute_delete(delete_query)

    # db = JellyDB(host="10.99.15.144", user="rdsroot", password="work@paddle123", db_name="jelly", port=8120)
    # task = db.select_by_condition(
    #     table="afl", 
    #     condition=f"description = 'PaddleX CE 动转静模型测试【PaddleX单卡动转静SOT训练】' "
    #     "AND afl_result = '总计新增报错case有1个, 其中有1个进入二分定位流程, 确信定位到的commit有0个, 请点击详情查看结果'")
    # print(task)

