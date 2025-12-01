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


def get_token_list(response):
    """解析 response 中的 token 文本列表"""
    token_list = []

    try:
        content_logprobs = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"解析失败：{e}")
        return []

    for token_info in content_logprobs:
        token = token_info.get("token")
        if token is not None:
            token_list.append(token)

    print(f"Token List:{token_list}")
    return token_list


def test_text_diff():
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
        "metadata": {
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        }
    }

    print("fastdeploy answer is :")

    try:
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        # for idx, chunk in enumerate(chunks):
        #         print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")
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
    # 对比baseline
    with open("./baseline.txt", "r", encoding="utf-8") as f:
        baseline = f.read()
    assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"
    # with open("./baseline.txt", "w", encoding="utf-8") as f:
    #     f.writelines(result)


def test_picture_diff():
    payload = {
        "model": "null",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "bos://nlp-sr-text2img/luobin06/dataset/doc_images/ChineseDocVQA/4e5278fdb82c881c69122c09f902e029.png",
                        },
                        "tokenizer_options": {
                            "resolution": 4096,
                            "version": "v1"
                        }
                    },
                    {"type": "text", "text": "哪个银行？"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "bos://nlp-sr-text2img/luobin06/dataset/doc_images/ChineseDocVQA/4e5278fdb82c881c69122c09f902e029.png",
                        },
                        "tokenizer_options": {
                            "resolution": 4096,
                            "version": "v1"
                        }
                    },
                    {"type": "text", "text": "哪个银行？"},
                ],
             },
        ],
        "stream": True,
        "temperature": 1.0,
        "seed": 21,
        "top_p": 0,
        "stop": ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"],
        "metadata": {
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        }
    }

    print("fastdeploy answer is :")

    try:
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        # for idx, chunk in enumerate(chunks):
        #         print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")
        result = "".join([x['choices'][0]['delta']['content'] for x in chunks])
    except Exception as e:
        print(f"解析失败: {e}")
        # 打印log/worklog.0
        if os.path.exists('log/workerlog.0'):
            with open('log/workerlog.0', 'r') as file:
                log_contents = file.read()
                print("################# workerlog.0 ##################", log_contents)
                pytest.fail(f"解析失败: {e}")
        print("chunks：", chunks)
    print("\nresult:\n", result)
    # 对比baseline
    with open("./baseline_pic.txt", "r", encoding="utf-8") as f:
        baseline = f.read()
    # with open("./baseline_pic.txt", "w", encoding="utf-8") as f:
    #     f.writelines(result)
    assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"


def test_chat_usage_stream():
    """测试流式chat usage"""
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
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "temperature": 1.0,
        "seed": 21,
        "top_p": 0,
        "stop": ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"],
        "metadata": {
            "min_tokens": 10,
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        },
        "max_tokens": 50,
    }

    response = send_request(url=URL, payload=payload)
    chunks = get_stream_chunks(response)
    result = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    print("Response:", result)
    assert result != "", "结果为空"
    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_chat_usage_non_stream():
    """测试非流式chat usage"""
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
        "stream": False,
        "temperature": 1.0,
        "seed": 21,
        "top_p": 0,
        "stop": ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"],
        "metadata": {
            "min_tokens": 10,
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        },
        "max_tokens": 50,
    }

    response = send_request(url=URL, payload=payload).json()
    usage = response["usage"]
    result = response["choices"][0]["message"]["content"]
    assert result != "", "结果为空"
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], "completion_tokens大于max_tokens"
    assert payload["metadata"]["min_tokens"] <= usage["completion_tokens"], "completion_tokens小于min_tokens"
    assert usage["total_tokens"] == total_tokens, "total_tokens不等于prompt_tokens + completion_tokens"


def test_non_chat_usage_stream():
    """测试completions接口 流式"""
    payload = {
        "model": "null",
        "prompt": "你好，你是谁？",
        "max_tokens": 50,
        "stream": True,
        # "temperature": 1.0,
        "seed": 566,
        # "top_p": 0,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "metadata": {
            "min_tokens": 10,
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        },
    }
    completion_url = URL.replace("chat/completions", "completions")

    response = send_request(url=completion_url, payload=payload)
    chunks = get_stream_chunks(response)

    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], f"completion_tokens大于max_tokens, usage: {usage}"
    assert (
        payload["metadata"]["min_tokens"] <= usage["completion_tokens"]
    ), f"completion_tokens小于min_tokens, usage: {usage}"
    assert (
        usage["total_tokens"] == total_tokens
    ), f"total_tokens不等于prompt_tokens + completion_tokens, usage: {usage}"


def test_non_chat_usage_non_stream():
    """测试completions接口 非流式"""
    payload = {
        "model": "null",
        "prompt": "你好，你是谁？",
        "max_tokens": 50,
        "stream": False,
        # "temperature": 1.0,
        "seed": 566,
        # "top_p": 0,
        "metadata": {
            "min_tokens": 10,
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        },
    }
    completion_url = URL.replace("chat/completions", "completions")

    response = send_request(url=completion_url, payload=payload)
    print(response.text)
    chunks = get_stream_chunks(response)

    usage = chunks[-1]["usage"]
    total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
    assert payload["max_tokens"] >= usage["completion_tokens"], f"completion_tokens大于max_tokens, usage: {usage}"
    assert (
        payload["metadata"]["min_tokens"] <= usage["completion_tokens"]
    ), f"completion_tokens小于min_tokens, usage: {usage}"
    assert (
        usage["total_tokens"] == total_tokens
    ), f"total_tokens不等于prompt_tokens + completion_tokens, usage: {usage}"


def test_stop_sequence():
    """测试stop"""
    payload = {
        "stream": False,
        "stop": ["。"],
        "messages": [
            {
                "role": "user",
                "content": "你要严格按照我接下来的话输出，输出冒号后面的内容，请输出：这是第一段。果冻这是第二段啦啦啦啦啦。",
            },
        ],
        "max_tokens": 20,
        "top_p": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "metadata": {
            "min_tokens": 10,
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        },
    }

    resp = send_request(URL, payload).json()
    # print(f"{json.dumps(resp, indent=2, ensure_ascii=False)}")
    content = resp["choices"][0]["message"]["content"]
    token_list = get_token_list(resp)
    print("token列表:", token_list)
    print("截断输出:", content)
    assert "第二段" not in content
    assert "第二段" not in token_list
    assert "。" in token_list, "没有找到。符号"


def test_stop_sequence1():
    """测试stop"""
    payload = {
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": "你要严格按照我接下来的话输出，输出冒号后面的内容，请输出：这是第一段。果冻这是第二段啦啦啦啦啦。",
            },
        ],
        "max_tokens": 20,
        "top_p": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "metadata": {
            "min_tokens": 10,
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        },
    }

    resp = send_request(URL, payload).json()
    # print(f"{json.dumps(resp, indent=2, ensure_ascii=False)}")
    content = resp["choices"][0]["message"]["content"]
    # token_list = get_token_list(resp)
    # print("token列表:", token_list)
    print("不截断输出:", content)
    assert "第二段" in content


def test_stop_sequence2():
    """测试stop"""
    payload = {
        "stream": False,
        "stop": ["这是第二段啦啦"],
        "messages": [
            {
                "role": "user",
                "content": "你要严格按照我接下来的话输出，输出冒号后面的内容，请输出：这是第一段。果冻这是第二段啦啦啦啦啦。",
            },
        ],
        "max_tokens": 20,
        "top_p": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "metadata": {
            "min_tokens": 10,
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        },
    }

    resp = send_request(URL, payload).json()
    # print(f"{json.dumps(resp, indent=2, ensure_ascii=False)}")
    content = resp["choices"][0]["message"]["content"]
    token_list = get_token_list(resp)
    print("token列表:", token_list)
    print("截断输出:", content)
    assert "啦啦啦" not in content


def test_non_stream_with_logprobs():
    """
    测试非流式响应开启 logprobs 后，返回的 token 概率信息是否正确。
    """
    payload = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "logprobs": True,
        "top_logprobs": 5,
        "seed": 21,
        "metadata": {
            "min_tokens": 1,
            "chat_template_kwargs": {
                "options": {
                    "thinking_mode": "close",
                },
            },
            "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
        },
    }

    response = send_request(URL, payload)
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["message"]["content"] == "<response>牛顿"
    assert resp_json["choices"][0]["logprobs"]["content"][0]["token"] == "<response>"
    assert resp_json["choices"][0]["logprobs"]["content"][0]["logprob"] == -3.2186455882765586e-06
    assert resp_json["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0] == {
        "token": "<response>",
        "logprob": -3.2186455882765586e-06,
        "bytes": [60, 114, 101, 115, 112, 111, 110, 115, 101, 62],
        "top_logprobs": None,
    }

    assert resp_json["usage"]["prompt_tokens"] == 50
    assert resp_json["usage"]["completion_tokens"] == 3
    assert resp_json["usage"]["total_tokens"] == 53


if __name__ == '__main__':
    test_non_stream_with_logprobs()
