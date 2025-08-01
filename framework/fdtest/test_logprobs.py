from core import *
import requests
import json

# 测试非流式响应开启 logprobs 后，返回的 token 概率信息是否正确
def test_unstream_logprobs():
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }
    payload = build_request_payload(TEMPLATE, data)
    req = send_request(URL, payload)
    print(json.dumps(req.json(), indent=2, ensure_ascii=False))
    req = req.json()
    assert req["choices"][0]["message"]["content"] == "牛顿的"
    assert req["choices"][0]["logprobs"]["content"][0]["token"] == "牛顿"
    assert req["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.031025361269712448
    assert req["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0] == {
        "token": "牛顿",
        "logprob": -0.031025361269712448,
        "bytes": None,
        "top_logprobs": None
    }
    assert req["usage"] == {
        "prompt_tokens": 22,
        "total_tokens": 25,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 0}
    }

# 测试非流式响应关闭 logprobs 后，返回中是否不包含 logprobs 字段
def test_unstream_unlogprobs():
    data = {
        "stream": False,
        "logprobs": False,
        "top_logprobs": None,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }
    payload = build_request_payload(TEMPLATE, data)
    req = send_request(URL, payload)
    print(json.dumps(req.json(), indent=2, ensure_ascii=False))
    req = req.json()
    assert req["choices"][0]["message"]["content"] == "牛顿的"
    assert req["choices"][0]["logprobs"] is None
    assert req["usage"] == {
        "prompt_tokens": 22,
        "total_tokens": 25,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 0}
    }

# 测试流式响应开启 logprobs 后，返回的首个 token 的概率信息是否正确
def test_stream_logprobs():
    data = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }
    payload = build_request_payload(TEMPLATE, data)
    req = send_request(URL, payload)
    result_dic = {}
    for line in req.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ")
        if decoded == "[DONE]":
            break
        chunk = json.loads(decoded)
        content = chunk["choices"][0]["delta"].get("content")
        if content:
            result_dic = chunk
            print(json.dumps(result_dic, indent=2, ensure_ascii=False))
            break

    assert result_dic["choices"][0]["delta"]["content"] == "牛顿"
    assert result_dic["choices"][0]["logprobs"]["content"][0]["token"] == "牛顿"
    assert result_dic["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.031025361269712448
    assert result_dic["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0] == {
        "token": "牛顿",
        "logprob": -0.031025361269712448
    }

# 测试流式响应关闭 logprobs 后，确认响应中不包含 logprobs 字段
def test_stream_unlogprobs():
    data = {
        "stream": True,
        "logprobs": False,
        "top_logprobs": None,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }
    payload = build_request_payload(TEMPLATE, data)
    req = send_request(URL, payload)
    result_dic = {}
    for line in req.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ")
        if decoded == "[DONE]":
            break
        chunk = json.loads(decoded)
        content = chunk["choices"][0]["delta"].get("content")
        if content:
            result_dic = chunk
            print(json.dumps(result_dic, indent=2, ensure_ascii=False))
            break

    assert result_dic["choices"][0]["delta"]["content"] == "牛顿"
    assert result_dic["choices"][0]["logprobs"] is None

# 主程序入口，运行测试
if __name__ == '__main__':
    test_unstream_logprobs()
    test_unstream_unlogprobs()
    test_stream_logprobs()
    test_stream_unlogprobs()
