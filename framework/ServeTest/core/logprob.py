#!/bin/env python3
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
ServeTest
"""
import os
import time
import sys
import json
from pathlib import Path
from .inferface import ResponseHandler
from logger import base_logger

class LogProbHandler(ResponseHandler):
    """
    LogProbHandler类，用于处理响应数据并保存为JSON Lines文件。
    """
    def __init__(self, res_list, args, case):
        """
        初始化LogProbHandler类的实例。
        """
        super().__init__(args.name)
        self.res_list = res_list
        self.args = args
        self.case = case

    def save(self, data, path):
        """
        将数据保存为JSON Lines文件。
        Args:
            data (list): 要保存的数据列表，每个元素为一个字典，表示一条记录。
            path (str): 文件保存路径。
        Returns:
            None
        Raises:
            None
        """
        save_as_jsonl(data, path)

    def compare(self, data, baseline_path):
        """
        将传入的数据与基准路径中的数据进行比较。
        Args:
            data (list): 要比较的数据列表。
            baseline_path (str): 基准数据的路径。
        Returns:
            None
        """
        compare_token_data_with_jsonl(data, baseline_path)

    def run(self):
        """
        运行测试。
        Args:
            无
        Returns:
            无
        Raises:
            无
        """
        if self.args.baseline:
            # 创建保存目录
            save_dir = os.path.join("baseline_output", get_timestamp_dir_name())
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            print(f"📁 创建基线保存目录：{save_dir}")
            for idx, res in enumerate(self.res_list, 1):
                token_data = parse_openai_stream(res, show_token_detail=False)  # show_token_detail=True 调试用
                save_path = os.path.join(save_dir, f"baseline_{idx}.jsonl")
                save_as_jsonl(token_data, save_path)
                print(f"⭐️ Baseline[{idx}] 保存成功：{save_path}")
            print(f"📁 基线文件全部保存成功，保存目录：{save_dir}")
        else:
            path = self.case["global"]["baseline"]

            for idx, res in enumerate(self.res_list, 1):
                token_data = parse_openai_stream(res, show_token_detail=False)
                try:
                    self.compare(token_data, os.path.join(path, f"baseline_{idx}.jsonl"))
                except Exception as e:
                    fail_info = {
                        "case_index": idx,
                        "baseline_path": os.path.join(path, f"baseline_{idx}.jsonl"),
                        "error": str(e)
                    }
                    self.failed_cases.append(fail_info)
                    print(f"[❌] Case {idx} 比较失败：{e}")

            # 汇总写入 result.log
            with open(self.result_log_path, "w", encoding="utf-8") as f:
                f.write(f"# 测试结果记录 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## 启动参数（args）：\n")
                for k, v in vars(self.args).items():
                    f.write(f"- {k}: {v}\n")
                f.write("\n")

                if self.failed_cases:
                    f.write(f"## ❌ 共计 {len(self.failed_cases)} 个失败用例\n")
                    for fail in self.failed_cases:
                        f.write(f"- Case {fail['case_index']} | Baseline: {fail['baseline_path']}\n")
                        f.write(f"  错误信息: {fail['error']}\n")
                else:
                    f.write("✅ 所有用例通过，无失败记录\n")

            print(f"📄 测试记录已写入：{self.result_log_path}")
            if len(self.failed_cases) > 0:
                print(f"❗️ 有 {len(self.failed_cases)} 个失败用例，请检查result.log")
                sys.exit(33)
            else:
                print("全部用例通过")


def parse_openai_stream(response, show_token_detail=True):
    """
    解析OpenAI API返回的数据流。

    Args:
        response (requests.Response): OpenAI API的响应对象。
        show_token_detail (bool): 是否打印每个token的详细信息。默认为True。

    Returns:
        list: 包含解析结果的列表，每个元素为一个字典，包含token的索引、文本、token_id、logprob和top_k信息。

    """
    output_text = ""
    token_index = 1
    token_results = []

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        raw = line.removeprefix("data: ").strip()
        if raw == "[DONE]":
            break

        try:
            chunk = json.loads(raw)
            choice = chunk.get("choices", [{}])[0]

            delta = choice.get("delta", {})
            token = delta.get("content", "")
            token_ids = delta.get("token_ids", [])
            token_bytes = delta.get("bytes", None)
            logprobs = choice.get("logprobs", {})
            logprob = None
            top_logprobs = []

            if logprobs and "content" in logprobs and logprobs["content"]:
                logprob_item = logprobs["content"][0]
                logprob = logprob_item.get("logprob")
                top_logprobs = logprob_item.get("top_logprobs", [])

            # 打印每个 token 的信息
            if token or token_ids:
                decoded = decode_token(token, token_bytes)
                output_text += decoded

                result = {
                    "index": token_index,
                    "text": decoded,
                    "token_id": token_ids[0] if token_ids else None,
                    "logprob": logprob,
                    "top_k": []
                }

                for cand in top_logprobs:
                    top_token = decode_token(cand.get("token", ""), cand.get("bytes"))
                    top_lp = cand.get("logprob", None)
                    result["top_k"].append({
                        "token": top_token,
                        "logprob": top_lp
                    })

                token_results.append(result)

                if show_token_detail:
                    base_logger.debug(f"[{token_index}] 文字：{decoded}")
                    if result["token_id"] is not None:
                        base_logger.debug(f"    Token ID: {result['token_id']}")
                    if result["logprob"] is not None:
                        base_logger.debug(f"    logprob: {result['logprob']:.3f}")
                    if result["top_k"]:
                        base_logger.debug("    Top-K:")
                        for i, cand in enumerate(result["top_k"], 1):
                            base_logger.debug(f"        {i}. {cand['token']}: {cand['logprob']:.3f}")

                token_index += 1

        except Exception as e:
            base_logger.error(f"[⚠️ 解析异常] {e}, 行内容: {line}")
            continue

    base_logger.info("🟩 最终文本输出：")
    base_logger.info(output_text)
    return token_results


def decode_token(token: str, byte_list=None) -> str:
    """
    尝试用 bytes 还原 token，fallback 到安全字符串显示。
    对于无法打印的字符，返回转义串，并注明是 fallback。
    """
    if byte_list:
        try:
            return bytes(byte_list).decode('utf-8')
        except Exception:
            return f"<InvalidBytes: {repr(bytes(byte_list))}>"

    try:
        if token.strip() == "":
            return repr(token)  # 控制符、空格、换行等
        decoded = token.encode("utf-8", "replace").decode("utf-8")
        if decoded == "�":
            return f"�  # 非法字符或编码不完整"
        return decoded
    except Exception:
        return f"<Unprintable: {repr(token)}>"


def save_as_jsonl(token_data, path):
    """
    将 token 数据保存为 JSONL 文件。

    Args:
        token_data (list): 包含 token 数据的列表，其中每个元素都是字典格式。
        path (str): 保存 JSONL 文件的路径。

    Returns:
        None

    """
    with open(path, "w", encoding="utf-8") as f:
        for item in token_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    base_logger.info(f"✅ 已保存为 JSONL 文件：{os.path.abspath(path)}")


def compare_token_data_with_jsonl(token_data, jsonl_path):
    """
    比对内存中的 token_data 与 jsonl 文件中的每一项内容，按 index 对应逐项比对。
    打印字段级不一致项，支持定位回归问题或调试。
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl_items = [json.loads(line) for line in f if line.strip()]

    success = True
    total = min(len(token_data), len(jsonl_items))
    base_logger.info(f"🔍 开始逐项比对，共 {total} 项...")

    for i in range(total):
        a = token_data[i]
        b = jsonl_items[i]
        if a != b:
            success = False
            base_logger.error(f"❌ 第 {i+1} 项不一致：")

            all_keys = set(a.keys()).union(set(b.keys()))
            for key in all_keys:
                a_val = a.get(key)
                b_val = b.get(key)
                if a_val != b_val:
                    base_logger.debug(f"  🔸 字段 '{key}' 不一致：")
                    base_logger.debug(f"     内存中 : {json.dumps(a_val, ensure_ascii=False)}")
                    base_logger.debug(f"     JSONL中: {json.dumps(b_val, ensure_ascii=False)}")

    if len(token_data) != len(jsonl_items):
        success = False
        base_logger.info(f"\n⚠️ 数量不一致：内存 {len(token_data)} vs JSONL {len(jsonl_items)}")

    if success:
        base_logger.info("✅ 全部一致")
        assert True
    else:
        base_logger.error("❗存在不一致项，请检查字段差异")
        assert False, "存在不一致项"


def get_timestamp_dir_name():
    """
    获取当前时间戳的目录名称。

    Args:
        无

    Returns:
        str: 以当前时间戳命名的目录名称，格式为 "baseline_YYYYMMDD_HHMMSS"。

    """
    return time.strftime("baseline_%Y%m%d_%H%M%S")