"""
get gsb
"""
# code: utf-8

import os
import sys
import datetime
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


def get_gsb(task_dt):
    """
    get original data and compute gsb
    """
    task_dt_today = datetime.date.today()
    task_dt_today = datetime.datetime.strptime(task_dt, "%Y-%m-%d")
    oneday = datetime.timedelta(days=1)
    task_dt_last = task_dt_today + oneday
    print(task_dt_today, task_dt_last)
    gsb = {
        "task_dt": task_dt,
        "gpu": {
            "env": "",
            "table_title": [],
            "metric": [],
            "frame": [],
            "value": {},
        },
        "cpu": {
            "env": "",
            "table_title": [],
            "metric": [],
            "frame": [],
            "value": {},
        },
        "slim": {
            "env": "",
            "table_title": [],
            "metric": [],
            "frame": [],
            "value": {},
        },
    }

    get_db_info()

    db = pymysql.connect(
        host=db_info["host"],
        port=db_info["port"],
        user=db_info["user"],
        password=db_info["password"],
        database=db_info["database"],
    )
    cursor = db.cursor(pymysql.cursors.DictCursor)

    sql_str = "select * from InferResult where device_type not like '%CPU%' and task_dt >= '{}' and task_dt < '{}'".format(
        task_dt_today,
        task_dt_last
    )
    cursor.execute(sql_str)
    res_infer = cursor.fetchall()
    select_compute(res_infer, gsb, "gpu")
    sql_str = "select * from InferResult where device_type like '%CPU%' and task_dt >= '{}' and task_dt < '{}'".format(
        task_dt_today,
        task_dt_last
    )
    cursor.execute(sql_str)
    res_infer = cursor.fetchall()
    select_compute(res_infer, gsb, "cpu")
    sql_str = "select * from SlimResult where task_dt >= '{}' and task_dt < '{}'".format(
       task_dt_today,
       task_dt_last
    )
    cursor.execute(sql_str)
    res_slim = cursor.fetchall()
    select_compute(res_slim, gsb, "slim")

    db.commit()
    cursor.close()
    db.close()
    return gsb


def select_compute(db_res, gsb, main_clas):
    """
    select and compute
    """
    if len(db_res) < 1:
        return
    for item in db_res:
        # item is dict
        model_name = item["model_name"]
        mode = ""
        if item["use_trt"] == 1:
            mode = "trt"
        elif item["use_mkldnn"] == 1:
            mode = "mkldnn"
        else:
            mode = "native"
        precision = item["fp_mode"]
        bs = item["batch_size"]
        frame = item["frame"]
        ips = item["ips"]
        cpu_mem = item["cpu_mem"]
        gpu_mem = item["gpu_mem"]
        device_type = item["device_type"]
        docker_image = item["docker_image"]
        paddle_commit = item["frame_commit"]
        # frame
        if frame not in gsb[main_clas]["frame"]:
            gsb[main_clas]["frame"].append(frame)
        # value
        if mode not in gsb[main_clas]["value"].keys():
            gsb[main_clas]["value"].setdefault(mode, {})
        if precision not in gsb[main_clas]["value"][mode].keys():
            gsb[main_clas]["value"][mode].setdefault(precision, {})
        if bs not in gsb[main_clas]["value"][mode][precision].keys():
            gsb[main_clas]["value"][mode][precision].setdefault(bs, {})
        if frame not in gsb[main_clas]["value"][mode][precision][bs].keys():
            gsb[main_clas]["value"][mode][precision][bs].setdefault(
                frame, {"ips": {"value": {}}, "cpu_mem": {"value": {}}, "gpu_mem": {"value": {}}}
            )
        if model_name not in gsb[main_clas]["value"][mode][precision][bs][frame]["ips"]["value"].keys():
            gsb[main_clas]["value"][mode][precision][bs][frame]["ips"]["value"][model_name] = ips
        if model_name not in gsb[main_clas]["value"][mode][precision][bs][frame]["cpu_mem"]["value"].keys():
            gsb[main_clas]["value"][mode][precision][bs][frame]["cpu_mem"]["value"][model_name] = cpu_mem
        if model_name not in gsb[main_clas]["value"][mode][precision][bs][frame]["gpu_mem"]["value"].keys():
            gsb[main_clas]["value"][mode][precision][bs][frame]["gpu_mem"]["value"][model_name] = gpu_mem

    # env
    gsb[main_clas]["env"] = "device_type:{}<br>docker_image:{}<br>paddle_commit:{}".format(
        device_type,
        docker_image,
        paddle_commit
    )
    # metric
    gsb[main_clas]["metric"] = ["ips", "cpu_mem", "gpu_mem"]
    # table_title
    gsb[main_clas]["table_title"] = ["模式", "precision", "batch_size"]
    for metric in gsb[main_clas]["metric"]:
        for frame in gsb[main_clas]["frame"]:
            if frame == "paddle":
                continue
            item = "{}-{}".format(metric, frame)
            gsb[main_clas]["table_title"].append(item)
    
    for mode in gsb[main_clas]["value"].keys():
        for precision in gsb[main_clas]["value"][mode].keys():
            for bs in gsb[main_clas]["value"][mode][precision].keys():
                frames = gsb[main_clas]["value"][mode][precision][bs].keys()
                if "paddle" not in frames:
                    continue
                metrics = gsb[main_clas]["value"][mode][precision][bs]["paddle"].keys()
                for metric in metrics:
                    models = gsb[main_clas]["value"][mode][precision][bs]["paddle"][metric]["value"].keys()
                    for frame in frames:
                        if frame == "paddle":
                            continue
                        _gsb = ""
                        g = 0
                        s = 0
                        b = 0
                        for model in models:
                            if model not in gsb[main_clas]["value"][mode][precision][bs][frame][metric]["value"].keys():
                                continue
                            v_paddle = gsb[main_clas]["value"][mode][precision][bs]["paddle"][metric]["value"][model]
                            v_frame = gsb[main_clas]["value"][mode][precision][bs][frame][metric]["value"][model]
                            if (v_frame <= 0) or (v_paddle <= 0):
                                continue
                            gap = (v_paddle - v_frame) / v_frame
                            if gap < -0.05:
                                b += 1
                            elif gap > 0.05:
                                g += 1
                            else:
                                s += 1
                        if (g > 0) or (s > 0) or (b > 0):
                            _gsb = "{}:{}:{}".format(g, s, b)
                            gsb[main_clas]["value"][mode][precision][bs][frame][metric].setdefault(
                                "gsb", _gsb
                            )


if __name__ == "__main__":
    task_dt = "2022-12-07"
    get_gsb(task_dt)
