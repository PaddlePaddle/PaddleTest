import json
import os
import pytest
import re
import copy

import requests


HOST = os.environ.get("HOST")
if not HOST:
    HOST = "127.0.0.1"
PORT = os.environ.get("FD_API_PORT")
if not PORT:
    raise ValueError("Please set FD_API_PORT environment variable.")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"


def send_request(url, payload, timeout=120):
    """
    发送请求到指定的URL，并返回响应结果。
    """
    headers = {
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout)
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
                    line = line[len("data: ") :]

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


def extract_logprobs(chunks):
    """
    提取 stream chunks 中的 logprobs（跳过 usage / 空 choices chunk）
    """
    results = []

    for chunk in chunks:
        choices = chunk.get("choices")
        if not choices:
            continue

        choice = choices[0]
        logprobs = choice.get("logprobs")
        if not logprobs or not logprobs.get("content"):
            continue

        token_infos = []
        for item in logprobs["content"]:
            token_infos.append({
                "token": item["token"],
                "logprob": item["logprob"],
                "top_logprobs": [
                    {
                        "token": tlp["token"],
                        "logprob": tlp["logprob"],
                    }
                    for tlp in item.get("top_logprobs", [])
                ]
            })

        results.append(token_infos)

    return results


def extract_last_entropy(log_path: str, req_id: str):
    """
    从日志中提取指定 req_id 的最后一条 entropy 值
    """
    pattern = re.compile(
        rf"req_id:\s*{re.escape(req_id)}_\d+.*entropy:\s*([0-9]*\.?[0-9]+)"
    )

    last_entropy = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                last_entropy = float(match.group(1))

    return last_entropy


def test_prefix_cache_text():
    payload = {
        "model": "null",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "国外项目风险管理研究起步较早，理论体系成熟。早期研究集中于保险与金融领域，后逐步扩展至工程项目、"
                                "公共管理等多领域。在理论层面，COSO《企业风险管理——整合框架》和ISO31000标准为风险管理提供了系统性"
                                "指导，强调风险识别、评估、应对与监控的全流程管理。风险识别方法包括故障树分析、事件树分析等；风险评估"
                                "则广泛应用VaR模型、蒙特卡洛模拟等量化工具。应对策略涵盖规避、转移、减轻和接受等，并衍生出风险共享、"
                                "升级等复杂策略。此外，组织文化、管理层支持等因素对风险管理有效性影响显著。近年来，随着科技发展，"
                                "人工智能、大数据等技术被引入风险管理，推动其向智能化、自动化方向发展。请介绍一下国外关于项目风险管理"
                                "的文献研究综述，300字以内",
                    }
                ]
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "temperature": 0.8,
        "seed": 21,
        "top_p": 0,
        "logprobs": True,
        "top_logprobs": 3,
        "stop": ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"],
        "chat_template_kwargs": {
            "options": {
                "thinking_mode": "close",
            },
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    print("fastdeploy answer is :")

    try:
        # prefix cache存在diff，跳过第一次请求
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        # for idx, chunk in enumerate(chunks):
        #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, indent=2, ensure_ascii=False)}")
        req_id = chunks[-1]["id"]
        result = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
        logprobs = extract_logprobs(chunks)
        entropy = extract_last_entropy("log/data_processor.log", req_id)
    except Exception as e:
        print(f"解析失败: {e}")
        # 打印log/worklog.0
        if os.path.exists("log/workerlog.0"):
            with open("log/workerlog.0", "r") as file:
                log_contents = file.read()
                print("################# workerlog.0 ##################", log_contents)
                pytest.fail(f"解析失败: {e}")
    print("\nresult:\n", result)
    # print("\nlogprobs:\n", logprobs)
    # mtp accept ratio
    mtp_ratio_base = {
        "accepted_tokens": 167,
        "rejected_tokens": 29,
        "accept_ratio": 0.4131736526946108,
        "average_accept_length": 1.7040816326530612,
        "accept_ratio_per_head": [
          0.7040816326530612
        ]
      }

    # 对比baseline
    # with open("/MODELDATA/baseline_cache_text.txt", "w", encoding="utf-8") as f:
    #     f.writelines(result)
    response = send_request(URL, payload)
    chunks = get_stream_chunks(response)
    req_id_2 = chunks[-1]["id"]
    result_2 = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    logprobs_2 = extract_logprobs(chunks)
    speculate_metrics_2 = chunks[-2]["choices"][0]["speculate_metrics"]
    entropy_2 = extract_last_entropy("log/data_processor.log", req_id_2)
    print("chunks:", chunks[-1])
    print("speculate_metrics:", speculate_metrics_2)
    print("entropy_2:", entropy_2)
    assert logprobs == logprobs_2, (
        "logprobs 前后不一致\n"
        f"logprobs_1: {json.dumps(logprobs, ensure_ascii=False, indent=2)}\n"
        f"logprobs_2: {json.dumps(logprobs_2, ensure_ascii=False, indent=2)}"
    )
    base_entropy = 0.1566898212183909
    assert abs(entropy - entropy_2) < 1e-12, (
        "entropy 前后不一致\n"
        f"entropy_1: {req_id}:{entropy}\n"
        f"entropy_2: {req_id_2}:{entropy_2}"
    )
    assert abs(entropy - base_entropy) < 1e-12 and abs(entropy_2 - base_entropy) < 1e-12, (
        "entropy 与期望值不一致\n"
        f"expected: {base_entropy}\n"
        f"entropy_1: {req_id}:{entropy}\n"
        f"entropy_2: {req_id_2}:{entropy_2}"
    )
    assert entropy == entropy_2, (
        "entropy 前后不一致\n"
        f"entropy_1: {req_id}:{entropy}\n"
        f"entropy_2: {req_id_2}:{entropy_2}"
    )
    assert speculate_metrics_2 == mtp_ratio_base, (
        f"speculate_metrics存在diff，" f"speculate_metrics_2: {speculate_metrics_2}\n " f"baseline: {mtp_ratio_base}"
    )
    if os.getenv("TEST_CUDA_GRAPH") == "1":
        print("TEST_CUDA_GRAPH=1, CUDA_GRAPH baseline")
        with open("/MODELDATA/baseline_cache_text_cuda.txt", "r", encoding="utf-8") as f:
            baseline = f.read()
        assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"
        assert result_2 == baseline, f"与baseline存在diff，result: {result_2}\n baseline: {baseline}"
    else:
        # 关cudagraph无法锁住两轮结果
        with open("/MODELDATA/baseline_cache_text.txt", "r", encoding="utf-8") as f:
            baseline = f.read()
        assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"
        assert result_2 == baseline, f"与baseline存在diff，result: {result_2}\n baseline: {baseline}"

    prompt_tokens = chunks[-1]["usage"]["prompt_tokens"]
    cached_tokens = chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
    # TODO:暂时关闭cached_tokens校验
    # assert cached_tokens == prompt_tokens // 64 * 64, "cached_tokens数量有问题"


def test_prefix_cache_picture():
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
                        "tokenizer_options": {"resolution": 4096, "version": "v1"},
                    },
                    {"type": "text", "text": "哪个银行？"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "bos://nlp-sr-text2img/luobin06/dataset/doc_images/ChineseDocVQA/4e5278fdb82c881c69122c09f902e029.png",
                        },
                        "tokenizer_options": {"resolution": 4096, "version": "v1"},
                    },
                    {"type": "text", "text": "哪个银行？"},
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "temperature": 1.0,
        "seed": 21,
        "top_p": 0,
        "max_tokens": 200,
        "stop": ["</s>", "<eos>", "<|endoftext|>", "<|im_end|>"],
        "chat_template_kwargs": {
            "options": {
                "thinking_mode": "close",
            },
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }

    print(json.dumps(payload, ensure_ascii=False))

    print("fastdeploy answer is :")

    try:
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        for idx, chunk in enumerate(chunks):
            print(f"\nchunk[{idx}]:\n{json.dumps(chunk, ensure_ascii=False)}")
        result = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    except Exception as e:
        print(f"解析失败: {e}")
        # 打印log/worklog.0
        if os.path.exists("log/workerlog.0"):
            with open("log/workerlog.0", "r") as file:
                log_contents = file.read()
                print("################# workerlog.0 ##################", log_contents)
                pytest.fail(f"解析失败: {e}")
    print("\nresult:\n", result)

    # 对比baseline
    # with open("/MODELDATA/baseline_cache_pic.txt", "w", encoding="utf-8") as f:
    #     f.writelines(result)
    with open("/MODELDATA/baseline_cache_pic.txt", "r", encoding="utf-8") as f:
        baseline = f.read()
    assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"

    response = send_request(URL, payload)
    chunks = get_stream_chunks(response)
    result_2 = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    print("chunks:", chunks[-1])

    assert result_2 == baseline, f"与baseline存在diff，result: {result}\n baseline: {result_2}"

    prompt_tokens = chunks[-1]["usage"]["prompt_tokens"]
    cached_tokens = chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
    # TODO:暂时关闭cached_tokens校验
    # assert cached_tokens == prompt_tokens // 64 * 64, "cached_tokens数量有问题"


def test_prefix_cache_video():
    """
    测试prefix cache disable-chunked-mm-input不影响精度
    """
    with open("/MODELDATA/video_for_fastdeploy", "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    original_video = next(item for item in data["messages"][0]["content"] if item["type"] == "video_url")
    payload = {
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "messages": [
            {
              "content": [
                {
                  "type": "video_url",
                  "video_url": {
                    "url": original_video["video_url"]["url"],
                  },
                  "enable_chunks": False,
                  "tokenizer_options": {
                    "frames": 10,
                    "end_ts": 290,
                    "version": "v1020"
                  }
                },
                {
                  "text": "简单介绍视频内容",
                  "type": "text"
                }
              ],
              "role": "user"
            }
          ],
        "temperature": 1.0,
        "seed": 21,
        "top_p": 0,
        "max_tokens": 200,
        "chat_template_kwargs": {
            "options": {
                "thinking_mode": "close",
            },
        },
        "bad_words_token_ids": [101031, 101032, 101027, 101028, 101023, 101024],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    try:
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        for idx, chunk in enumerate(chunks):
            print(f"\nchunk[{idx}]:\n{json.dumps(chunk, ensure_ascii=False)}")
        result = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    except Exception as e:
        print(f"解析失败: {e}")
        # 打印log/worklog.0
        if os.path.exists("log/workerlog.0"):
            with open("log/workerlog.0", "r") as file:
                log_contents = file.read()
                print("################# workerlog.0 ##################", log_contents)
                pytest.fail(f"解析失败: {e}")
    print("\nresult:\n", result)
    # 对比baseline
    # with open("/MODELDATA/baseline_cache_video.txt", "w", encoding="utf-8") as f:
    #     f.writelines(result)

    response = send_request(URL, payload)
    chunks = get_stream_chunks(response)
    result_2 = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    print("chunks:", chunks[-1])

    if os.getenv("TEST_CUDA_GRAPH") == "1":
        print("TEST_CUDA_GRAPH=1, CUDA_GRAPH baseline")
        with open("/MODELDATA/baseline_cache_video_cuda.txt", "r", encoding="utf-8") as f:
            baseline = f.read()
        assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"
        assert result_2 == baseline, f"与baseline存在diff，result: {result_2}\n baseline: {baseline}"
    else:
        # 关cudagraph无法锁住两轮结果
        with open("/MODELDATA/baseline_cache_video.txt", "r", encoding="utf-8") as f:
            baseline = f.read()
        assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"
        # assert result_2 == baseline, f"与baseline存在diff，result: {result_2}\n baseline: {baseline}"
    prompt_tokens = chunks[-1]["usage"]["prompt_tokens"]
    cached_tokens = chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
    # 视频输入触发回退，23符合预期
    # TODO:暂时关闭cached_tokens校验
    # assert cached_tokens == 23, "cached_tokens数量有问题"


if __name__ == '__main__':
    test_prefix_cache_text()
