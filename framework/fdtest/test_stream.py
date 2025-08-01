from core import *
import requests
import json

def test_stream_and_not_stream():
    # 测试接口在 stream 模式和非 stream 模式下返回的内容是否一致

    # 发送 stream=True 的请求，解析流式响应
    data = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 100,
    }
    payload = build_request_payload(TEMPLATE, data)
    req = send_request(URL, payload)

    # 解析流式响应内容
    resp_chunks = []
    for line in req.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                decoded = decoded[len("data: "):]
            if decoded == "[DONE]":
                break
            resp_chunks.append(json.loads(decoded))

    # 拼接最终生成内容
    final_content = "".join([
        chunk["choices"][0]["delta"]["content"]
        for chunk in resp_chunks
        if "choices" in chunk and "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]
    ])
    print(final_content)

    # 发送 stream=False 的请求，获取完整响应
    data["stream"] = False
    payload = build_request_payload(TEMPLATE, data)
    req = send_request(URL, payload)
    print(json.dumps(req.json(), indent=2, ensure_ascii=False))
    req = req.json()

    # 对比 stream 与非 stream 响应内容是否一致
    assert final_content == req["choices"][0]["message"]["content"]

if __name__ == '__main__':
    test_stream_and_not_stream()  # 运行测试函数
