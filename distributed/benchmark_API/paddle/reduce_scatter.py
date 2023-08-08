# -*- coding: utf-8 -*-
import argparse
import yaml

import time

import paddle
import paddle.distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument("--case_name", default="reduce_scatter_legacy_sync")
args = parser.parse_args()
case_name = args.case_name

f = open("config.yaml", "rb")
yaml_config = yaml.load(f, Loader=yaml.FullLoader)["reduce_scatter"]


def get_res(case, config):

    is_legacy = yaml_config[case_name]["is_legacy"]
    sync_op = yaml_config[case_name]["sync_op"]
    use_calc_stream = yaml_config[case_name]["use_calc_stream"]
    is_tensor = yaml_config[case_name]["is_tensor"]

    # args
    warms = 5
    epochs = 20
    devices = 8
    b = 1024
    e = 134217728  # 128M

    dist.init_parallel_env()

    byte_to_test = []
    while b <= e:
        byte_to_test.append(b)
        b *= 2

    time_list = {case_name: {}}
    for b in byte_to_test:
        n_ele = b // 4 // devices

        if dist.get_rank() == 0:
            data = paddle.to_tensor([0] * n_ele, "float32")
        else:
            data = paddle.to_tensor([1] * n_ele, "float32")

        if is_legacy == True:
            tensor_list = [paddle.to_tensor([0] * n_ele, "float32") for i in range(devices)]
            # warmup
            for i in range(warms):
                dist.reduce_scatter(data, tensor_list, sync_op=sync_op)
            paddle.device.cuda.synchronize()  # 等待给定的 CUDA 设备上的计算完成
            # stats
            start = time.perf_counter()  # 返回当前的计算机系统时间
            for i in range(epochs):
                dist.reduce_scatter(data, tensor_list, sync_op=sync_op)
            paddle.device.cuda.synchronize()
            cost = (time.perf_counter() - start) / epochs
        else:
            if is_tensor == True:
                tensor = paddle.to_tensor([0] * n_ele * devices, "float32")
                # warmup
                for i in range(warms):
                    dist.stream.reduce_scatter(data, tensor, sync_op=sync_op, use_calc_stream=use_calc_stream)
                paddle.device.cuda.synchronize()
                # stats
                start = time.perf_counter()
                for i in range(epochs):
                    dist.stream.reduce_scatter(data, tensor, sync_op=sync_op, use_calc_stream=use_calc_stream)
                paddle.device.cuda.synchronize()
                cost = (time.perf_counter() - start) / epochs
            else:
                tensor_list = [paddle.to_tensor([0] * n_ele, "float32") for i in range(devices)]
                # warmup
                for i in range(warms):
                    dist.stream.reduce_scatter(
                        data, tensor_list, sync_op=sync_op, use_calc_stream=use_calc_stream
                    )
                paddle.device.cuda.synchronize()
                # stats
                start = time.perf_counter()
                for i in range(epochs):
                    dist.stream.reduce_scatter(
                        data, tensor_list, sync_op=sync_op, use_calc_stream=use_calc_stream
                    )
                paddle.device.cuda.synchronize()
                cost = (time.perf_counter() - start) / epochs

        # print(f'data: {b} B, time: {cost} s, algbw: {b/1_000_000_000/cost} GB/s')
        if b < 1048576: # 1MB
            num = str(b // 1024) + "KB"
        else:
            num = str(b // 1024 // 1024) + "MB"
        time_list[case_name][num] = {}
        time_list[case_name][num]["time"] = cost
        time_list[case_name][num]["algbw"] = b / 1_000_000_000 / cost
    # print(time_list)
    return time_list


if __name__ == "__main__":
    print(yaml_config[case_name])
    res = get_res(case_name, yaml_config)
    print(res)
