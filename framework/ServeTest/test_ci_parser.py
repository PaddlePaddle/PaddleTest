import json
import os
import pytest
import re

import requests
from utils import *


HOST = os.environ.get("HOST")
if not HOST:
    HOST = "127.0.0.1"
PORT = os.environ.get("FD_API_PORT")
if not PORT:
    raise ValueError("Please set FD_API_PORT environment variable.")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"


def send_request(url, payload, timeout=600):
    """
    发送请求到指定的URL，并返回响应结果。
    """
    headers = {
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        print("🟢 接收响应中...\n")
        return res
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时（超过 {timeout} 秒）")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败：{e}")
        return None


def get_stream_chunks(response):
    """解析流式返回，生成chunk List[dict]"""
    chunks = []

    if response.status_code == 200:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    line = line[len("data: "):]

                if line.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except Exception as e:
                    print(f"解析失败: {e}, 行内容: {line}")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("返回内容：", response.text)

    return chunks


def test_reasoning_parser():
    payload = {
        "model": "null",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "解释一下温故而知新",
                    },
                ],
             },
        ],
        "stream": True,
        "temperature": 1.0,
        "seed": 21,
        "top_p": 0,
        "stop": ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"],
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    print("fastdeploy answer is :")

    try:
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        # for idx, chunk in enumerate(chunks):
        #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, ensure_ascii=False)}")
        reasoning_result = "".join([x['choices'][0]['delta']['reasoning_content'] for x in chunks])
        result = "".join([x['choices'][0]['delta']['content'] for x in chunks])
    except Exception as e:
        print(f"解析失败: {e}")
        # 打印log/worklog.0
        if os.path.exists('log/workerlog.0'):
            with open('log/workerlog.0', 'r') as file:
                log_contents = file.read()
                print("################# workerlog.0 ##################", log_contents)
                pytest.fail(f"解析失败: {e}")
    print("\nresult:\n", result)
    if os.getenv("BASELINE") == "1":
        with open(f"/MODELDATA/baseline_parser_result_{TEST_BRANCH}.txt", "w", encoding="utf-8") as f:
            f.writelines(result)
        with open(f"/MODELDATA/baseline_parser_reason_{TEST_BRANCH}.txt", "w", encoding="utf-8") as f:
            f.writelines(reasoning_result)
    # 对比baseline
    if os.getenv("AGILE_COMPILE_BRANCH") == "release/online/20251131":
        with open("/MODELDATA/baseline_parser_result_1131.txt", "r", encoding="utf-8") as f:
            baseline = f.read()
        with open("/MODELDATA/baseline_parser_reason_1131.txt", "r", encoding="utf-8") as f:
            baseline_reason = f.read()
    else:
        with open(f"/MODELDATA/baseline_parser_result_{TEST_BRANCH}.txt", "r", encoding="utf-8") as f:
            baseline = f.read()
        with open(f"/MODELDATA/baseline_parser_reason_{TEST_BRANCH}.txt", "r", encoding="utf-8") as f:
            baseline_reason = f.read()
    print("reasoning_result:\n", reasoning_result)
    print("result:\n", result)
    assert reasoning_result == baseline_reason, f"思考结果与baseline存在diff，" \
                                                f"result: {reasoning_result}\n baseline: {baseline_reason}"
    assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"


if __name__ == '__main__':
    test_reasoning_parser()
