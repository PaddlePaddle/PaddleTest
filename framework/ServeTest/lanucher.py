#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ServeTest
"""
import argparse
import os.path
from utils import *
from logger import base_logger
from case_loader import load_yaml_case
from core import *


def parse_args():
    """
    解析命令行参数。

    Args:
        无

    Returns:
        argparse.Namespace: 解析后的命令行参数对象。

    Raises:
        argparse.ArgumentError: 如果命令行参数不符合要求，抛出该异常。

    """
    parser = argparse.ArgumentParser(
        description="FD Case Launcher - 启动并执行指定测试用例"
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="FastDeploy 服务 URL，例如 http://localhost:8000/v1/chat/completions"
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="任务名，用于后续做入库标识"
    )

    parser.add_argument(
        "--case",
        type=str,
        required=True,
        help="测试用例文件路径，YAML 格式，例如 ./cases/test_case1.yaml"
    )

    parser.add_argument(
        "--executor", "-exe",
        type=str,
        required=True,
        default=None,
        help="执行器类型，需要在route中注册"
    )

    parser.add_argument(
        "--request_template", "-rt",
        type=str,
        required=True,
        default=None,
        help="请求模板文件路径（可选），例如 ./template/request_template.json"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="请求超时时间，单位秒，默认 60 秒"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="并发请求数量（仅非 baseline 模式有效），默认 4"
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="是否启用 baseline 模式，添加该参数则为 True，默认为 False"
    )

    args = parser.parse_args()

    # 参数校验
    if not args.url.startswith("http://") and not args.url.startswith("https://"):
        parser.error("参数 --url 必须是 http 或 https 开头的 URL")

    if not os.path.isfile(args.case):
        parser.error(f"指定的测试用例文件不存在: {args.case}")

    return args


def check(args):
    """
    检查并输出参数信息。

    Args:
        args (argparse.Namespace): 包含命令行参数的命名空间对象。

    Raises:
        ValueError: 如果指定的执行器不存在。

    """
    if args.baseline:
        base_logger.info(f"基线模式已启用")
    else:
        base_logger.info(f"非录入基线模式")
    base_logger.info(f"FastDeploy 服务地址: {args.url}")
    base_logger.info(f"任务名: {args.name}")
    base_logger.info(f"测试用例路径: {args.case}")
    base_logger.info(f"请求超时时间: {args.timeout} 秒")

    if args.request_template:
        base_logger.info(f"使用请求模板: {args.request_template}")
    else:
        base_logger.info(f"未指定请求模板，将使用默认构造逻辑")

    if args.executor not in ROUTE.keys():
        raise ValueError(f"{args.executor}执行器不存在，请指定正确的执行器类型")


def main():
    """
    主函数

    Args:
        无

    Returns:
        无

    """
    args = parse_args()
    # 入参检查
    check(args)
    # 加载测试用例
    case_data = load_yaml_case(args.case)
    # 请求结果获取s
    res_list = response(args)

    # 根据入参路由解析类, 接口固定
    exe = ROUTE[args.executor](res_list, args, case_data)
    exe.run()


if __name__ == "__main__":
    main()
