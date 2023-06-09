# -*- coding: utf-8 -*-
import argparse
import time

import paddle
import paddle.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--is_legacy", default=False, action='store_true', help="dist.stream/dist")
parser.add_argument("--sync_op", default=False, action='store_true', help="False/True")
parser.add_argument("--use_calc_stream", default=False, action='store_true', help="False/True")
parser.add_argument("--is_tensor", default=True, action='store_true', help="tensor_list/tensor")

args = parser.parse_args()
is_legacy = args.is_legacy
sync_op = args.sync_op
use_calc_stream = args.use_calc_stream
is_tensor = args.is_tensor

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

time_list = defaultdict(lambda: (dict))
for b in byte_to_test:
    n_ele = b // 4 // devices

    if dist.get_rank() == 0:
        data = paddle.to_tensor([0] * n_ele, 'float32')
    else:
        data = paddle.to_tensor([1] * n_ele, 'float32')

    if is_legacy == True:
        # tensor_list = [
        #     paddle.to_tensor([0] * n_ele, 'float32') for i in range(devices)
        # ]
        # warmup
        for i in range(warms):
            dist.all_reduce(data, sync_op=sync_op)
        paddle.device.cuda.synchronize() # 等待给定的 CUDA 设备上的计算完成
        # stats
        start = time.perf_counter() # 返回当前的计算机系统时间
        for i in range(epochs):
            dist.all_reduce(data, sync_op=sync_op)
        paddle.device.cuda.synchronize()
        cost = (time.perf_counter() - start) / epochs
    else:
        if is_tensor == True:
            tensor = paddle.to_tensor([0] * n_ele * devices, 'float32')
            # warmup
            for i in range(warms):
                dist.stream.all_reduce(data,
                                       sync_op=sync_op,
                                       use_calc_stream=use_calc_stream)
            paddle.device.cuda.synchronize()
            # stats
            start = time.perf_counter()
            for i in range(epochs):
                dist.stream.all_reduce(data,
                                       sync_op=sync_op,
                                       use_calc_stream=use_calc_stream)
            paddle.device.cuda.synchronize()
            cost = (time.perf_counter() - start) / epochs
        # else:
        #     tensor_list = [
        #         paddle.to_tensor([0] * n_ele, 'float32')
        #         for i in range(devices)
        #     ]
        #     # warmup
        #     for i in range(warms):
        #         dist.stream.all_gather(tensor_list,
        #                                data,
        #                                sync_op=sync_op,
        #                                use_calc_stream=use_calc_stream)
        #     paddle.device.cuda.synchronize()
        #     # stats
        #     start = time.perf_counter()
        #     for i in range(epochs):
        #         dist.stream.all_gather(tensor_list,
        #                                data,
        #                                sync_op=sync_op,
        #                                use_calc_stream=use_calc_stream)
        #     paddle.device.cuda.synchronize()
        #     cost = (time.perf_counter() - start) / epochs

    # print(f'data: {b}B, time: {cost} s, algbw: {b/1_000_000_000/cost} GB/s')
    temp_info = time_list[b]
    temp_info.update({'time': cost, 'algbw': b/1_000_000_000/cost})
print(time_list)