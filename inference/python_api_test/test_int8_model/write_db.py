"""
write to db
"""
# coding=utf-8

import os
import sys
import yaml
import pymysql


db_info = {
    "host": "",
    "port": 1,
    "user": "",
    "password": "",
    "database": "",
}


def get_db_info():
    """
    get db info
    """
    with open("db_info.yaml", "r") as fin:
        file_date = yaml.load(fin.read(), Loader=yaml.Loader)
        db_info["host"] = file_date["host"]
        db_info["port"] = int(file_date["port"])
        db_info["user"] = file_date["user"]
        db_info["password"] = file_date["password"]
        db_info["database"] = file_date["database"]


def write(res):
    """
    write to db
    """
    get_db_info()
    db = pymysql.connect(
        host=db_info["host"],
        port=db_info["port"],
        user=db_info["user"],
        password=db_info["password"],
        database=db_info["database"],
    )
    cursor = db.cursor()

    # cases
    sql_str = "insert into SlimResult \
                        (task_dt, \
                         model_name, batch_size, fp_mode, use_trt, use_mkldnn, \
                         ips, ips_unit, cpu_mem, gpu_mem, \
                         frame, frame_branch, frame_commit, frame_version, \
                         docker_image, python_version, cuda_version, cudnn_version, trt_version, \
                         device_type, thread_num, jingdu, jingdu_unit) \
                        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    val = []
    for item in res:
        val.append(
            (
                item["task_dt"],
                item["model_name"],
                item["batch_size"],
                item["fp_mode"],
                item["use_trt"],
                item["use_mkldnn"],
                item["ips"],
                item["ips_unit"],
                item["cpu_mem"],
                item["gpu_mem"],
                item["frame"],
                item["frame_branch"],
                item["frame_commit"],
                item["frame_version"],
                item["docker_image"],
                item["python_version"],
                item["cuda_version"],
                item["cudnn_version"],
                item["trt_version"],
                item["device"],
                item["thread_num"],
                item["jingdu"],
                item["jingdu_unit"],
            )
        )
    cursor.executemany(sql_str, val)
    db.commit()
    db.close()


if __name__ == "__main__":
    pass
