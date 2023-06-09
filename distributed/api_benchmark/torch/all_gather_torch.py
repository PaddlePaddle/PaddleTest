import time

import torch
import torch.distributed as dist

# args
warms = 5
epochs = 20
devices = 8
b = 1024
e = 134217728  # 128M

dist.init_process_group('nccl')
rank = dist.get_rank()

byte_to_test = []
while b <= e:
    byte_to_test.append(b)
    b *= 2

for b in byte_to_test:
    n_ele = b // 4 // devices

    if rank == 0:
        data = torch.tensor([0] * n_ele, dtype=torch.float32).to(rank)
    else:
        data = torch.tensor([1] * n_ele, dtype=torch.float32).to(rank)

    tensor_list = [
        torch.tensor([0] * n_ele, dtype=torch.float32).to(rank)
        for i in range(devices)
    ]
    # warmup
    for i in range(warms):
        dist.all_gather(tensor_list, data)
    torch.cuda.synchronize()
    # stats
    start = time.perf_counter()
    for i in range(epochs):
        dist.all_gather(tensor_list, data)
    torch.cuda.synchronize()
    cost = (time.perf_counter() - start) / epochs

    print(f'data: {b}B, time: {cost}s, algbw: {b/1_000_000_000/cost}GB/s')