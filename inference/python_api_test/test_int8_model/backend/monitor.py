"""
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import multiprocessing
import time
import sys
import os

try:
    import pynvml
    import psutil
    import GPUtil
except Exception as e:
    sys.stderr.write("Cannot import tensorrt, maybe it's not installed.\n")


class StatBase(object):
    """StatBase"""

    keys = ("cpu_memory.used", "gpu_memory.used")


class Monitor(StatBase):
    """Monitor"""

    def __init__(self, gpu_id=0, interval=0.1):
        """init"""
        self.result = {}
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
        self.gpu_id = gpu_id
        self.interval = interval

        self.stat_queue = multiprocessing.Queue()

    def start(self):
        """start"""
        # cpu stat
        pid = os.getpid()
        self.stat_worker = multiprocessing.Process(
            target=self.stat_func, args=(self.stat_queue, pid, self.gpu_id, self.interval)
        )
        self.stat_worker.start()

    def stop(self):
        """stop"""
        try:
            self.stat_worker.terminate()
            self.stat_worker.join(timeout=0.01)
        except Exception as e:
            print(e)
            return

        # cpu
        stat_result = {}
        if self.stat_queue.qsize() > 0:
            stat_result = {k: v for k, v in zip(StatBase.keys, self.stat_queue.get())}
        while not self.stat_queue.empty():
            item = {k: v for k, v in zip(StatBase.keys, self.stat_queue.get())}
            for k in StatBase.keys:
                stat_result[k] = max(stat_result[k], item[k])
        self.result["result"] = stat_result

    def stat_func(self, queue, pid, gpu_id, interval=0.0):
        """cpu stat function"""
        cpu_info = psutil.Process(pid)
        while True:
            # cpu info
            info = cpu_info.memory_full_info()
            cpu_mem = info.uss / 1024.0 / 1024.0

            # gpu info
            gpu_mem = 0
            gpus = GPUtil.getGPUs()
            if gpu_id is not None and len(gpus) > 0 and gpu_id < len(gpus):
                # gpu_percent = gpus[gpu_id].load
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem = meminfo.used / 1024.0 / 1024.0
            cpu_mem_use, gpu_mem_use = round(cpu_mem, 4), round(gpu_mem, 4)

            queue.put([cpu_mem_use, gpu_mem_use])
            time.sleep(interval)
        return

    def output(self):
        """output"""
        return self.result


if __name__ == "__main__":
    begin = time.time()
    monitor = Monitor(0)

    monitor.start()
    time.sleep(80)
    monitor.stop()

    print(monitor.output())
    end = time.time()
    print(end - begin)
