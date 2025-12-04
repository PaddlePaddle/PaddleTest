import os
import pytest
from utils import *

import requests


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


def recursive_diff(a, b, path=""):
    """
    递归比较两个不规则嵌套数组（list）
    - a, b: 任意层嵌套的列表
    - path: 当前路径标识
    返回: 差异信息列表
    """
    diffs = []

    # 两者都为 list，继续深入比较
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append({
                "path": path,
                "msg": f"长度不同: {len(a)} vs {len(b)}",
                "a": a,
                "b": b
            })
            # 继续对比重叠部分
            for i in range(min(len(a), len(b))):
                diffs += recursive_diff(a[i], b[i], f"{path}[{i}]")
        else:
            for i in range(len(a)):
                diffs += recursive_diff(a[i], b[i], f"{path}[{i}]")

    else:
        # 不是列表，直接比较值
        if a != b:
            diffs.append({
                "path": path,
                "msg": f"值不同: {a} vs {b}"
            })

    return diffs


def test_text_to_image_diff():
    payload = {
        "model": "null",
        "messages": [
            {
                "role": "user",
                "content": "一张长椅，静静地放置在户外环境中。长椅的木质表面呈现出自然的纹理，给人一种复古而温馨的感觉。"
                           "背景是一片宁静的公园景色，绿树成荫，小径通幽。",
             },
        ],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "temperature": 0.7,
        "seed": 21,
        "top_p": 0,
        "stop": ["停止生成"],
        "disable_chat_template": False,
        "return_token_ids": True,
        "height": 256,
        "width": 256
    }

    print("fastdeploy answer is :")
    completion_token_ids = []
    url = ""
    result = ""

    try:
        response = send_request(URL, payload, timeout=1200)
        chunks = get_stream_chunks(response)
        # for idx, chunk in enumerate(chunks):
        #         print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")
        for chunk in chunks[:-1]:
            delta = chunk["choices"][0].get("delta", {})
            multimodal_content = delta.get("multimodal_content", [])
            if multimodal_content:
                if multimodal_content[0].get("type", "text") == "image":
                    url = multimodal_content[0].get("url", None)
                    if not url:
                        print(f"got url error, multimodal_content[0] is {multimodal_content[0]}")

                    completion_token_ids = multimodal_content[0].get("completion_token_ids", None)

                    if completion_token_ids:
                        with open("img_token_file", "w") as f:
                            print(completion_token_ids, file=f, end="")
                else:
                    res = multimodal_content[0].get("text", None)
                    # print(res, end="", flush=True)
                    result += res
        print("#####completion_token_ids:\n", completion_token_ids)
        print("#####url:\n", url)
        print("#####result:\n", result)

    except Exception as e:
        print(f"解析失败: {e}")
        # 打印log/worklog.0
        if os.path.exists('log/workerlog.0'):
            with open('log/workerlog.0', 'r') as file:
                log_contents = file.read()
                print("################# workerlog.0 ##################", log_contents)
                pytest.fail(f"解析失败: {e}")

    # with open("./baseline_t2i_tokens.txt", "w", encoding="utf-8") as f:
    #     json.dump(completion_token_ids, f)
    #
    # with open("./baseline_t2i.txt", "w", encoding="utf-8") as f:
    #     f.writelines(result)

    with open("./baseline_t2i.txt", "r", encoding="utf-8") as f:
        baseline = f.read()

    with open("/cot/baseline_t2i_tokens.txt", "r", encoding="utf-8") as f:
        baseline_tokens = json.load(f)

    diffs = recursive_diff(baseline_tokens, completion_token_ids)

    if not diffs:
        print("✅ completion_token_ids 完全一致")
    else:
        print(f"❌ 共发现 {len(diffs)} 处差异：")
        for i, d in enumerate(diffs[:10]):
            print(f"\n[{i + 1}] 路径: {d['path']}")
            print(f"👉 {d['msg']}")
            if "a" in d and isinstance(d["a"], list):
                print(f"baseline 子数组: {d['a']}")
            if "b" in d and isinstance(d["b"], list):
                print(f"current 子数组:  {d['b']}")

    assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"
    # assert url, "got url error"
    assert not diffs, f"与baseline存在diff，diffs: {diffs[:10]}"


if __name__ == '__main__':
    test_text_to_image_diff()
