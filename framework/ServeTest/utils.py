#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ServeTest
"""
import sys
import requests
import copy
from config.request_template import *
import concurrent.futures
from logger import base_logger
from case_loader import load_yaml_case


def build_request_payload(template_name: str, case_data: dict, payload: dict = None) -> dict:
    """
    基于模板构造请求 payload，按优先级依次合并：
    template < payload 参数 < case_data，后者会覆盖前者的同名字段。

    :param template_name: 模板变量名，例如 "TOKEN_LOGPROB"
    :param case_data: 单条测试数据，必须包含 messages 字段
    :param payload: 可选的覆盖字段（用于 CLI/动态传参）
    :return: 构造后的完整请求 payload dict
    """
    try:
        template = globals()[template_name]
    except KeyError:
        base_logger.error(f"[ERROR] 请求模板 `{template_name}` 不存在于 request_template.py 中")
        sys.exit(1)

    if "messages" not in case_data:
        base_logger.error("[ERROR] case_data 缺少必填字段 'messages'")
        sys.exit(1)

    # 深拷贝模板
    final_payload = copy.deepcopy(template)

    # 合并 payload 参数（优先级 2）
    if payload:
        for k, v in payload.items():
            final_payload[k] = v

    # 合并 case_data（优先级最高）
    for k, v in case_data.items():
        final_payload[k] = v

    return final_payload


def send_request(url, payload, timeout=600, stream=False):
    """
    向指定URL发送POST请求，并返回响应结果。

    Args:
        url (str): 请求的目标URL。
        payload (dict): 请求的负载数据，应该是一个字典类型。
        timeout (int, optional): 请求的超时时间，默认为600秒。
        stream (bool, optional): 是否以流的方式下载响应内容，默认为False。

    Returns:
        response: 请求的响应结果，如果请求失败则返回None。

    Raises:
        None

    """
    headers = {
        "Content-Type": "application/json",
    }
    base_logger.info("🔄 正在请求模型接口...")

    try:
        res = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=timeout
        )
        base_logger.info("🟢 接收响应中...\n")
        return res
    except requests.exceptions.Timeout:
        base_logger.error(f"❌ 请求超时（超过 {timeout} 秒）")
        return None
    except requests.exceptions.RequestException as e:
        base_logger.error(f"❌ 请求失败：{e}")
        return None


def response(args):
    """
    发送请求并处理响应。

    Args:
        args (argparse.Namespace): 包含命令行参数的对象。

    Returns:
        list: 包含所有请求的响应的列表。

    Raises:
        SystemExit: 如果测试用例格式错误或加载 YAML 测试用例失败，则退出程序。

    """
    try:
        case = load_yaml_case(args.case)
        case_data = case.get("cases")
        global_settings = case.get("global")
        if not isinstance(case_data, list):
            base_logger.error("测试用例格式错误，应为 YAML 列表")
            sys.exit(33)
        else:
            base_logger.info(f"测试用例数量: {len(case_data)}")
    except Exception as e:
        base_logger.error(f"加载 YAML 测试用例失败: {e}")
        sys.exit(33)

    template_name = args.request_template
    if args.concurrency > 1:
        base_logger.info(f"开启并发模式，最大线程数: {args.concurrency}")
        stream = True
    else:
        base_logger.info(f"开启串行模式，最大线程数: 1")
        stream = False

    def request_worker(index):
        """
        向指定索引的工作节点发送请求。
        Args:
            index (int): 工作节点的索引。
        Returns:
            tuple: 包含索引和请求响应的元组。
        """
        case = case_data[index]
        payload = build_request_payload(template_name, case, global_settings.get("payload"))
        base_logger.debug(payload)
        res = send_request(args.url, payload, timeout=args.timeout, stream=stream)
        return index, res  # ✅ 带索引返回，确保后续排序

    # 设置并发线程数
    max_workers = min(args.concurrency or 4, len(case_data))

    # 提交任务（按索引）
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(request_worker, i): i for i in range(len(case_data))}

        result_dict = {}
        for future in concurrent.futures.as_completed(futures):
            idx, resp = future.result()
            result_dict[idx] = resp

    # ✅ 最后按顺序返回 response 列表
    req_list = [result_dict[i] for i in range(len(case_data))]
    return req_list
