import json
import os
import pytest

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


def test_diff():
    # 如果将chat_template拼接在输入中，需要把disable_chat_template设为True
    # 注：sft后模型模版，预训练模型没法识别这些特殊token
    text = (
        "<|im_start|>system\n"
        "<global_setting>\n"
        "think_mode=True\n"
        "</global_setting><|im_end|>\n"
        "\n"
        "<|im_start|>user\n"
        "写巴黎圣母院，模仿《滕王阁序》<|im_end|>\n"
        "\n"
        "<|im_start|>assistant\n"
        "<think>"
    )

    # tokenizer_config.json需要配置chat_template
    # text = "写巴黎圣母院，模仿《滕王阁序》"
    payload = {
        "messages": [
            {"role": "user", "content": text},
        ],
        "stream": True,
        "max_tokens": 64,
        "temperature": 1.0,
        "seed": 21,
        "top_p": 0,
        "disable_chat_template": True
    }

    print("fastdeploy answer is :")

    response = send_request(URL, payload)
    chunks = get_stream_chunks(response)
    # for idx, chunk in enumerate(chunks):
    #         print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")
    try:
        result = "".join([x['choices'][0]['delta']['content'] for x in chunks])
    except Exception as e:
        print(f"解析失败: {e}")
        # 打印log/worklog.0
        if os.path.exists('log/worklog.0'):
            with open('log/worklog.0', 'r') as file:
                log_contents = file.read()
                print(log_contents)
                pytest.fail(f"解析失败: {e}")
    print("\nresult:\n", result)
    # 对比baseline
    with open("./baseline.txt", "r", encoding="utf-8") as f:
        baseline = f.read()
    assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"
    # with open("./baseline.txt", "w", encoding="utf-8") as f:
    #     f.writelines(result)


if __name__ == '__main__':
    test_diff()
