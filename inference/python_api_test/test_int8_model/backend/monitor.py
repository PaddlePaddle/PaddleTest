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
import subprocess
import time
import sys
import os
import signal
import cpuinfo

try:
    import pynvml
    import psutil
    import GPUtil
except Exception as e:
    sys.stderr.write("Cannot import pynvml, psutil, GPUtil, maybe it's not installed.\n")


class StatBase(object):
    """StatBase"""

    nvidia_smi_path = "nvidia-smi"
    gpu_keys = (
        "index",
        "uuid",
        "name",
        "timestamp",
        "memory.total",
        "memory.free",
        "memory.used",
        "utilization.gpu",
        "utilization.memory",
    )
    xpu_keys = (
        "pci_addr",
        "board_id",
        "dev_id",
        "sn",
        "temperature",
        "p1",
        "mem temperature",
        "p2",
        "power(mW)",
        "freq_0",
        "freq_1",
        "freq_2",
        "freq_3",
        "freq_4",
        "freq_5",
        "L3_used",
        "L3_size",
        "HBM_used",
        "HBM_size",
        "use_ratio",
        "firmware version",
        "model",
    )
    nu_opt = ",nounits"
    cpu_keys = ("cpu.util", "memory.util", "memory.used")


class Monitor(StatBase):
    """Monitor"""

    def __init__(self, gpu_id=0, use_gpu=True, xpu_id=0, use_xpu=False, interval=0.1):
        self.result = {}
        self.result["result"] = {}
        # in Paddle-Test int8 model test
        # we usually export CUDA_VISIBLE_DEVICES=global_gpu_id first.
        self.gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", gpu_id))
        self.use_gpu = use_gpu
        self.xpu_id = int(os.environ.get("XPU_VISIBLE_DEVICES", xpu_id))
        self.use_xpu = use_xpu
        self.interval = interval

        self.cpu_stat_q = multiprocessing.Queue()

    def start(self):
        """start"""
        cmd = "%s --id=%s --query-gpu=%s --format=csv,noheader%s -lms 50" % (
            StatBase.nvidia_smi_path,
            self.gpu_id,
            ",".join(StatBase.gpu_keys),
            StatBase.nu_opt,
        )
        xpu_cmd = f"while true; do xpu_smi -d{self.xpu_id} -m; sleep 0.05; done"
        # print(cmd)
        if self.use_gpu:
            self.gpu_stat_worker = subprocess.Popen(
                cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True, preexec_fn=os.setsid
            )

        if self.use_xpu:
            self.xpu_stat_worker = subprocess.Popen(
                xpu_cmd,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                close_fds=True,
                preexec_fn=os.setsid,
            )

        # cpu stat
        pid = os.getpid()
        self.cpu_stat_worker = multiprocessing.Process(
            target=self.cpu_stat_func, args=(self.cpu_stat_q, pid, self.interval)
        )
        self.cpu_stat_worker.start()

    def stop(self):
        """stop"""
        try:
            if self.use_gpu:
                os.killpg(self.gpu_stat_worker.pid, signal.SIGUSR1)
            if self.use_xpu:
                os.killpg(self.xpu_stat_worker.pid, signal.SIGUSR1)
            # os.killpg(p.pid, signal.SIGTERM)
            self.cpu_stat_worker.terminate()
            self.cpu_stat_worker.join(timeout=0.01)
        except Exception as e:
            print(e)
            return

        # gpu
        if self.use_gpu:
            lines = self.gpu_stat_worker.stdout.readlines()
            # print(lines)
            lines = [line.strip().decode("utf-8") for line in lines if line.strip() != ""]
            gpu_info_list = [{k: v for k, v in zip(StatBase.gpu_keys, line.split(", "))} for line in lines]
            if len(gpu_info_list) == 0:
                return
            result = gpu_info_list[0]
            for item in gpu_info_list:
                for k in item.keys():
                    if k not in ["name", "uuid", "timestamp"]:
                        result[k] = max(int(result[k]), int(item[k]))
                    else:
                        result[k] = max(result[k], item[k])
            self.result["result"]["gpu_memory.used"] = result["memory.used"]

        # xpu
        if self.use_xpu:
            lines = self.xpu_stat_worker.stdout.readlines()
            # print(lines)
            lines = [line.strip().decode("utf-8") for line in lines if line.strip() != ""]
            xpu_info_list = [{k: v for k, v in zip(StatBase.xpu_keys, line.split(" "))} for line in lines]
            if len(xpu_info_list) == 0:
                return
            result = xpu_info_list[0]
            for item in xpu_info_list:
                for k in item.keys():
                    if k not in ["pci_addr", "sn", "firmware version", "model"]:
                        result[k] = max(int(result[k]), int(item[k]))
                    else:
                        result[k] = max(result[k], item[k])
            self.result["XPU"] = result

        # cpu
        cpu_result = {}
        if self.cpu_stat_q.qsize() > 0:
            cpu_result = {k: v for k, v in zip(StatBase.cpu_keys, self.cpu_stat_q.get())}
        while not self.cpu_stat_q.empty():
            item = {k: v for k, v in zip(StatBase.cpu_keys, self.cpu_stat_q.get())}
            for k in StatBase.cpu_keys:
                cpu_result[k] = max(cpu_result[k], item[k])
        cpu_result["name"] = cpuinfo.get_cpu_info()["brand_raw"]
        self.result["result"]["cpu_memory.used"] = cpu_result["memory.used"]

    def output(self):
        """output"""
        return self.result

    def cpu_stat_func(self, q, pid, interval=0.01):
        """cpu stat function"""
        stat_info = psutil.Process(pid)
        while True:
            # pid = os.getpid()
            cpu_util, mem_util, mem_use = (
                stat_info.cpu_percent(),
                stat_info.memory_percent(),
                round(stat_info.memory_info().rss / 1024.0 / 1024.0, 4),
            )
            q.put([cpu_util, mem_util, mem_use])
            time.sleep(interval)
        return


if __name__ == "__main__":
    begin = time.time()
    # monitor = Monitor(0)
    monitor = Monitor(use_gpu=False, use_xpu=True, xpu_id=2)

    monitor.start()
    time.sleep(80)
    monitor.stop()

    print(monitor.output())
    end = time.time()
    print(end - begin)
