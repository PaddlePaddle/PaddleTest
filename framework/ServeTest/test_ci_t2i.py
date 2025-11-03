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


def test_text_to_image_diff():
    payload = {
        "model": "null",
        "messages": [
            {
                "role": "user",
                "content": "一个杯子，造型简约而现代，线条流畅，具有一定的艺术美感。",
             },
        ],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "temperature": 0.7,
        "seed": 22,
        "top_p": 1.0,
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
        response = send_request(URL, payload)
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
        # print("#####completion_token_ids:\n", completion_token_ids)
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

    # with open("./baseline_t2i.txt", "w", encoding="utf-8") as f:
    #     f.writelines(result)

    with open("./baseline_t2i.txt", "r", encoding="utf-8") as f:
        baseline = f.read()

    assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"


if __name__ == '__main__':
    test_text_to_image_diff()
