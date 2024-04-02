"""
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""

import time
import argparse
import paddle
import pandas as pd


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./result.txt", help="tipc benchmark log path")
    parser.add_argument(
        "--paddle_output_name", type=str, default="tipc_benchmark_paddle.xlsx", help="paddle output excel file name"
    )
    parser.add_argument(
        "--docker_name",
        type=str,
        default="registry.baidubce.com/device/paddle-xpu:ubuntu18-x86_64-gcc82",
        help="docker name",
    )
    return parser.parse_args()


def log_split(file_name: str) -> list:
    """
    log split
    """
    log_list = []
    with open(file_name, "r") as f:
        log_lines = f.read().split("\n\n")
        for log_line in log_lines:
            log_list.append(log_line)

    return log_list


def process_log(log_list: list) -> dict:
    """
    process log to dict
    """
    output_dict = {}

    for log_line in log_list.split("\n"):

        if "model_name" in log_line:
            output_dict["model_name"] = log_line.split(" : ")[-1].strip()
            continue
        if "repo" in log_line:
            output_dict["repo"] = log_line.split(" : ")[-1].strip()
            continue
        if "avg_cost" in log_line:
            output_dict["avg_cost"] = log_line.split(" : ")[-1].strip()
            continue
        if "device_name" in log_line:
            output_dict["device_name"] = log_line.split(" : ")[-1].strip()
            continue
        if "HBM_used" in log_line:
            output_dict["HBM_used"] = log_line.split(" : ")[-1].strip()
            continue
        if "l3_cache" in log_line:
            output_dict["l3_cache"] = log_line.split(" : ")[-1].strip()
            continue
        if "precision" in log_line:
            output_dict["precision"] = log_line.split(" : ")[-1].strip()
            continue
        if "unit" in log_line:
            output_dict["unit"] = log_line.split(" : ")[-1].strip()
            continue
        if "jingdu" in log_line:
            output_dict["jingdu"] = log_line.split(" : ")[-1].strip()
            continue
        else:
            continue
    if "device_name" not in output_dict.keys() and "model_name" in output_dict.keys():
        output_dict["device_name"] = "CPU"
        output_dict["gpu_mem"] = "0"
    return output_dict


def data_process(env, paddle_version, output_total_list):
    """
    data process func
    """
    log_time = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    data_merging_logs_paddle = []
    for output_dict_num, _ in enumerate(output_total_list):
        output_dict = output_total_list[output_dict_num]
        log_dict = {}
        log_dict["日期"] = log_time
        log_dict["环境"] = env
        log_dict["version"] = paddle_version
        log_dict["device_name"] = output_dict["device_name"]
        log_dict["model_name"] = output_dict["model_name"]
        log_dict["repo"] = output_dict["repo"]
        log_dict["precision"] = output_dict["precision"]
        log_dict["l3_cache"] = output_dict["l3_cache"]
        log_dict["unit"] = output_dict["unit"]
        log_dict["精度"] = output_dict["jingdu"]
        log_dict["avg_cost"] = output_dict["avg_cost"]
        log_dict["HBM_used"] = output_dict["HBM_used"]

        data_merging_logs_paddle.append(log_dict)

    return data_merging_logs_paddle


def main(args, result_path):
    """
    main
    """
    # create empty DataFrame
    env = args.docker_name
    paddle_commit = paddle.__git_commit__
    paddle_tag = paddle.__version__
    paddle_version = paddle_tag + "/" + paddle_commit
    columns_list = [
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
    origin_df = pd.DataFrame(columns=columns_list)

    origin_df_paddle = origin_df.copy()

    log_list = log_split(result_path)

    dict_list = []
    for one_model_log in log_list:
        output_total_list = process_log(one_model_log)
        if "model_name" in output_total_list.keys():
            dict_list.append(output_total_list)

    output_excl_list_paddle = data_process(env, paddle_version, dict_list)
    for one_log in output_excl_list_paddle:
        origin_df_paddle = pd.concat([origin_df_paddle, pd.Series(one_log).to_frame().T], ignore_index=True)

    origin_df_paddle.sort_values(by=["repo", "model_name", "precision", "l3_cache"], inplace=True)
    origin_df_paddle.to_excel(args.paddle_output_name)


if __name__ == "__main__":
    args = parse_args()
    result_path = args.result_path
    main(args, result_path)
