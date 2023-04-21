#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
md5获取
"""

import uuid
import hashlib
import cpuinfo
import psutil
import pynvml


class Snapshot(object):
    """
    md5获取
    """

    def __init__(self):
        self.tag = "api_benchmark"  # md5 扰乱

    def get_md5_id(self):
        """
        获取唯一标识码
        """
        message = self.status()

        # 创建 MD5 对象
        md5 = hashlib.md5()
        # 更新 MD5 对象的内容
        md5.update((message + self.tag).encode("utf-8"))
        # 获取 MD5 值
        md5_value = md5.hexdigest()
        # 输出结果
        print("md5_code: " + md5_value)
        return md5_value

    def status(self):
        """
        显示状态
        """
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        mac_address = ":".join(["{:02x}".format((uuid.getnode() >> ele) & 0xFF) for ele in range(0, 8 * 6, 8)][::-1])
        print(f"CPU使用率：{cpu_percent}%, 内存使用率：{memory_percent}%, MAC地址：{mac_address}")
        try:
            self.get_nvml()
        except Exception:
            print("显卡状态获取异常")

        info = cpuinfo.get_cpu_info()
        cpu_name = "_".join([info["brand_raw"], info["arch"], info["arch_string_raw"]])
        print(f"cpu名字: {cpu_name}")
        mem_total = round(psutil.virtual_memory().total / (1024 * 1024 * 1024), 2)
        print(f"内存总量: {mem_total}GB")
        return " | ".join([cpu_name, str(mem_total), mac_address])

    def get_nvml(self):
        """
        获取显卡信息
        """

        # 初始化pynvml
        pynvml.nvmlInit()
        # 获取第0个GPU的句柄
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # 获取GPU的型号名称
        name = pynvml.nvmlDeviceGetName(handle)
        # 获取GPU的利用率
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        # 打印结果
        print(f"GPU型号：{name.decode('utf-8')}")
        print(f"GPU利用率：{utilization}%")
        # 释放pynvml资源
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    """
    test
    """
    snapshot = Snapshot()
    q = snapshot.get_md5_id()
