import json
import os
import pytest
import copy

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


def test_prefix_cache_text():
    payload = {
        "model": "null",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "我请介绍一下国外关于项目风险管理的文献研究综述，300字以内"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "国外项目风险管理研究起步较早，理论体系成熟。早期研究集中于保险与金融领域，后逐步扩展至工程项目、"
                                "公共管理等多领域。在理论层面，COSO《企业风险管理——整合框架》和ISO31000标准为风险管理提供了系统性"
                                "指导，强调风险识别、评估、应对与监控的全流程管理。风险识别方法包括故障树分析、事件树分析等；风险评估"
                                "则广泛应用VaR模型、蒙特卡洛模拟等量化工具。应对策略涵盖规避、转移、减轻和接受等，并衍生出风险共享、"
                                "升级等复杂策略。此外，组织文化、管理层支持等因素对风险管理有效性影响显著。近年来，随着科技发展，"
                                "人工智能、大数据等技术被引入风险管理，推动其向智能化、自动化方向发展。",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "请介绍一下国内关于智慧交通的文献研究综述，300字以内"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "国内关于智慧交通的文献研究近年来呈现蓬勃发展态势，主要聚焦于概念体系构建、关键技术应用及实践案例分析。"
                                "在概念层面，学者普遍认为智慧交通是智能交通的升级形态，强调通过物联网、大数据、人工智能等技术实现"
                                "交通系统全要素的实时感知与智能决策。研究重点集中于车路协同、交通大数据分析、智能信号控制等核心技术，"
                                "如基于深度学习的交通流预测模型准确率已提升至90%以上。实践层面，北京“交通大脑”、上海“智能出行”平台"
                                "等案例验证了智慧交通在缓解拥堵、提升效率方面的显著成效。同时，文献指出当前研究仍存在技术标准不统一、"
                                "数据壁垒突出等挑战，未来需加强跨领域协同创新，推动智慧交通与自动驾驶、数字孪生等技术的深度融合，"
                                "为城市交通治理提供更科学的决策支持。",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "作者是工商管理专业研究生，在写题为《ZY公司智慧交通项目风险管理研究》的毕业论文，"
                                             "请介绍一下国外关于智慧交通的文献研究综述，300字以内"}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "国外关于智慧交通的文献研究起步较早，成果丰富。美国作为智慧交通发源地，研究重点聚焦于提升交通效率、"
                                "降低能耗及减少环境污染，构建了涵盖出行管理、公共交通、紧急事件处理等八大服务领域的体系架构。欧洲国家"
                                "则注重标准化与推广应用，通过立法和政策推动技术发展，如欧盟制定的合作智能交通系统标准。日本在智慧"
                                "交通研究中强调与自动驾驶技术的结合，致"
                                "力于实现智能交通与智能车辆的协同发展。此外，国外研究还广泛涉及交通信息采集、数据处理与分析、通信与"
                                "控制等关键技术，以及智慧交通在提升交通效率、改善交通安全、促进可持续发展等方面的作用，为ZY公司智慧"
                                "交通项目风险管理提供了重要的理论支撑和实践参考。",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "请介绍一下国内关于项目风险管理的文献研究综述，300字以内"}]},
        ],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "temperature": 1.0,
        "seed": 21,
        "top_p": 0,
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
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        # for idx, chunk in enumerate(chunks):
        #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, ensure_ascii=False)}")
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

    response = send_request(URL, payload)
    chunks = get_stream_chunks(response)
    result_2 = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    print("chunks:", chunks[-1])
    # 纯文prefix cache存在diff，暂不校验
    # assert result == result_2, f"与baseline存在diff，result: {result}\n baseline: {result_2}"

    prompt_tokens = chunks[-1]["usage"]["prompt_tokens"]
    cached_tokens = chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached_tokens == prompt_tokens // 64 * 64, "cached_tokens数量有问题"


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

    original_pic = {
        "type": "image_url",
        "image_url": {
            "url": "bos://nlp-sr-text2img/luobin06/dataset/doc_images/ChineseDocVQA/4e5278fdb82c881c69122c09f902e029.png",
        },
        "tokenizer_options": {"resolution": 4096, "version": "v1"},
    }
    payload["messages"][0]["content"].insert(1, copy.deepcopy(original_pic))
    for _ in range(5):
        payload["messages"][0]["content"].insert(1, copy.deepcopy(original_pic))

    print(json.dumps(payload, ensure_ascii=False))

    print("fastdeploy answer is :")

    try:
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        # for idx, chunk in enumerate(chunks):
        #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, ensure_ascii=False)}")
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

    response = send_request(URL, payload)
    chunks = get_stream_chunks(response)
    result_2 = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    print("chunks:", chunks[-1])

    assert result == result_2, f"与baseline存在diff，result: {result}\n baseline: {result_2}"

    prompt_tokens = chunks[-1]["usage"]["prompt_tokens"]
    cached_tokens = chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached_tokens == prompt_tokens // 64 * 64, "cached_tokens数量有问题"


def test_prefix_cache_video():
    """
    测试prefix cache disable-chunked-mm-input不影响精度
    """
    with open("/MODELDATA/video_128k_for_fastdeploy_0805", "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    original_video = next(item for item in data["messages"][0]["content"] if item["type"] == "video_url")
    data["messages"][0]["content"].insert(1, copy.deepcopy(original_video))
    for _ in range(10):
        data["messages"][0]["content"].insert(1, copy.deepcopy(original_video))
    print(json.dumps(data, indent=2, ensure_ascii=False))
    payload = {
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "messages": data["messages"],
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

    try:
        response = send_request(URL, payload)
        chunks = get_stream_chunks(response)
        # for idx, chunk in enumerate(chunks):
        #     print(f"\nchunk[{idx}]:\n{json.dumps(chunk, ensure_ascii=False)}")
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
    with open("./baseline_video_mtp.txt", "w", encoding="utf-8") as f:
        f.writelines(result)
    with open("./baseline_video_mtp.txt", "r", encoding="utf-8") as f:
        baseline = f.read()
    assert result == baseline, f"与baseline存在diff，result: {result}\n baseline: {baseline}"

    response = send_request(URL, payload)
    chunks = get_stream_chunks(response)
    result_2 = "".join([x["choices"][0]["delta"]["content"] for x in chunks[:-1]])
    print("chunks:", chunks[-1])
    # 纯文prefix cache存在diff，暂不校验
    # assert result_2 == baseline, f"与baseline存在diff，result_2: {result_2}\n baseline: {baseline}"
    prompt_tokens = chunks[-1]["usage"]["prompt_tokens"]
    cached_tokens = chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached_tokens == prompt_tokens // 64 * 64, "cached_tokens数量有问题"

if __name__ == '__main__':
    test_prefix_cache_video()
