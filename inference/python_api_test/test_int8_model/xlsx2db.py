"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


import datetime
import subprocess
import argparse
import yaml
import numpy as np
import pymysql
import pandas as pd


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--now_excel_name", type=str, default="tipc_benchmark_paddle.xlsx", help="output excel file name"
    )
    parser.add_argument("--task_link", type=str, default="task_link", help="task link url")
    parser.add_argument("--db_info", type=str, default="db_info.yaml", help="db info yaml")
    parser.add_argument("--branch", type=str, default="develop", help="Paddle branch")
    parser.add_argument("--build_num", type=int, default=0, help="build number")
    parser.add_argument("--build_id", type=int, default=0, help="build id")
    parser.add_argument("--paddle_path", type=str, default="./", help="Paddle repo path")
    return parser.parse_args()


def main():
    """
    main func
    """
    select_columns = [
        "日期",
        "环境",
        "version",
        "device_name",
        "model_name",
        "repo",
        "precision",
        "l3_cache",
        "unit",
        "精度",
        "avg_cost",
        "HBM_used",
    ]
    head_columns = ["model_name", "repo", "precision", "l3_cache"]
    now_df = pd.read_excel(args.now_excel_name)[select_columns]
    # 相同config保留最后一次数据，重跑机制决定最后一次数据(最优或首次diff<5%)
    now_df = now_df.drop_duplicates(head_columns, keep="last")
    now_df.to_excel("tipc_benchmark_paddle_xpu.xlsx")

    db_dict = yaml.load(open(args.db_info, "r", encoding="utf-8"), Loader=yaml.FullLoader)

    conn = pymysql.connect(
        host=db_dict["host"],
        user=db_dict["user"],
        passwd=db_dict["passwd"],
        db=db_dict["db"],
        port=db_dict["port"],
        charset="utf8",
    )
    cursor = conn.cursor()

    sql = (
        "insert into InferXPUInt8Result(task_time,build_num,task_link,branch,commit_id,commit_time,"
        "model_name,repo,infer_precision,l3_cache,unit,accuracy,avg_cost,HBM_used,device_type) "
        "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )

    time = str(datetime.datetime.now())
    task_link = f"http://{args.task_link}/viewLog.html?buildId={args.build_id}"
    one_line = now_df.iloc[0]
    # print(one_line)
    commit_id = one_line["version"].split("/")[-1]
    print(commit_id)
    # print(backend)
    device_name = one_line["device_name"]

    repo_directory = args.paddle_path
    commit_sha = commit_id
    git_command = f"git --git-dir={repo_directory}/.git log -1 --pretty=format:%ad --date=iso {commit_sha}"
    try:
        commit_time = subprocess.check_output(git_command, shell=True, text=True)
        time_1 = datetime.datetime.strptime(commit_time, "%Y-%m-%d %H:%M:%S %z")
        print("Commit Time:", commit_time.strip())
        print("Commit Time:", time_1)
    except subprocess.CalledProcessError as e:
        print("Error:", e)

    data_list = []
    for i, row in now_df.iterrows():
        data_list.append(
            (
                time,
                args.build_num,
                task_link,
                args.branch,
                commit_id,
                time_1,
                row["model_name"],
                row["repo"],
                row["precision"],
                row["l3_cache"],
                row["unit"],
                None if np.isnan(row["精度"]) else row["精度"],
                None if np.isnan(row["avg_cost"]) else row["avg_cost"],
                None if np.isnan(row["HBM_used"]) else row["HBM_used"],
                device_name,
            )
        )

    cursor.executemany(sql, data_list)
    # print(data_list)
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    args = parse_args()
    main()
