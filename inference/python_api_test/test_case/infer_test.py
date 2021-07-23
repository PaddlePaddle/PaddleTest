# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
infer test
"""
import time
import os
import sys
import logging
import threading
from multiprocessing import Process


import psutil
import yaml
import pytest
import pynvml
import numpy as np
import paddle.inference as paddle_infer

from pynvml.smi import nvidia_smi

_gpu_mem_lists = []

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class InferenceTest(object):
    """
    python inference test base class
    """

    def __init__(self):
        """
        __init__
        """
        pass

    def load_config(self, **kwargs):
        """
        load model to create config
        Args:
            model_path(str) : uncombined model path
            model_file(str) : combined model's model file
            params_file(str) : combined model's params file
        Returns:
            None
        """
        model_path = kwargs.get("model_path", None)
        model_file = kwargs.get("model_file", None)
        params_file = kwargs.get("params_file", None)

        if model_path:
            assert os.path.exists(model_path)
            self.pd_config = paddle_infer.Config(model_path)
        elif model_file:
            assert os.path.exists(params_file)
            self.pd_config = paddle_infer.Config(model_file, params_file)
        else:
            raise Exception(f"model file path is not exist, [{model_path}] or [{model_file}] invalid!")

    def get_truth_val(self, input_data_dict: dict, device: str) -> dict:
        """
        get truth value calculated by target device kernel
        Args:
            input_data_dict(dict) : input data constructed as dictionary
        Returns:
            None
        """
        if device == "cpu":
            self.pd_config.disable_gpu()
        elif device == "gpu":
            self.pd_config.enable_use_gpu(1000, 0)
        else:
            raise Exception(f"{device} not support in current test codes")
        self.pd_config.switch_ir_optim(False)
        predictor = paddle_infer.create_predictor(self.pd_config)

        input_names = predictor.get_input_names()
        for i, input_data_name in enumerate(input_names):
            input_handle = predictor.get_input_handle(input_data_name)
            input_handle.copy_from_cpu(input_data_dict[input_data_name])

        predictor.run()

        output_data_dict = {}
        output_names = predictor.get_output_names()
        for _, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data = output_data.flatten()
            output_data_dict[output_data_name] = output_data
        return output_data_dict

    def config_test(self):
        """
        test config instance
        """
        assert isinstance(self.pd_config, paddle_infer.Config), "Paddle Inference Config created failed"

    def disable_gpu_test(self, input_data_dict: dict, repeat=20):
        """
        test disable_gpu() api
        Args:
            input_data_dict(dict) : input data constructed as dictionary
            repeat(int) : inference repeat time, set to catch gpu mem
        Returns:
            None
        """
        self.pd_config.disable_gpu()
        predictor = paddle_infer.create_predictor(self.pd_config)

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            cuda_visible_device = int(cuda_visible_devices.split(",")[0])
        else:
            cuda_visible_device = 0

        ori_gpu_mem = float(get_gpu_mem(cuda_visible_device)["used(MB)"])

        record_thread = threading.Thread(target=record_by_pid, args=(os.getpid(), cuda_visible_device))
        record_thread.setDaemon(True)
        record_thread.start()

        input_names = predictor.get_input_names()
        for i, input_data_name in enumerate(input_names):
            input_handle = predictor.get_input_handle(input_data_name)
            input_handle.copy_from_cpu(input_data_dict[input_data_name])

        for i in range(repeat):
            predictor.run()

        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()

        gpu_max_mem = max([float(i["used(MB)"]) for i in _gpu_mem_lists])
        assert abs(gpu_max_mem - ori_gpu_mem) < 1, "set disable_gpu(), but gpu activity found"

    def trt_fp32_bz1_test(self, input_data_dict: dict, output_data_dict: dict, repeat=5, delta=1e-5):
        """
        test enable_tensorrt_engine()
        batch_size = 1
        trt max_batch_size = 1
        precision_mode = paddle_infer.PrecisionType.Float32
        Args:
            input_data_dict(dict) : input data constructed as dictionary
            output_data_dict(dict) : output data constructed as dictionary
            repeat(int) : inference repeat time, set to catch gpu mem
            delta(float): difference threshold between inference outputs and thruth value
        Returns:
            None
        """
        self.pd_config.enable_use_gpu(1000, 0)
        self.pd_config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        predictor = paddle_infer.create_predictor(self.pd_config)

        input_names = predictor.get_input_names()
        for _, input_data_name in enumerate(input_names):
            input_handle = predictor.get_input_handle(input_data_name)
            input_handle.copy_from_cpu(input_data_dict[input_data_name])

        for i in range(repeat):
            predictor.run()

        output_names = predictor.get_output_names()
        for i, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data = output_data.flatten()
            output_data_truth_val = output_data_dict[output_data_name].flatten()
            for j, out_data in enumerate(output_data):
                assert (
                    abs(out_data - output_data_truth_val[j]) <= delta
                ), f"{out_data} - {output_data_truth_val[j]} > {delta}"

    def trt_fp32_bz1_multi_thread_test(self, input_data_dict: dict, output_data_dict: dict, repeat=5, delta=1e-5):
        """
        test enable_tensorrt_engine()
        batch_size = 1
        trt max_batch_size = 4
        thread_num = 5
        precision_mode = paddle_infer.PrecisionType.Float32
        多线程TensorRT预测器，max_batch_size=4,预测batch_size=1
        Args:
            input_data_dict(dict) : input data constructed as dictionary
            output_data_dict(dict) : output data constructed as dictionary
            repeat(int) : inference repeat time, set to catch gpu mem
            delta(float): difference threshold between inference outputs and thruth value
        Returns:
            None
        """
        self.pd_config.enable_use_gpu(1000, 0)
        self.pd_config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=4,
            min_subgraph_size=3,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        thread_num = 5
        predictors = paddle_infer.PredictorPool(self.pd_config, thread_num)
        for i in range(5):
            record_thread = threading.Thread(
                target=self.run_multi_thread_test_predictor, 
                args=(predictors.retrive(i), input_data_dict, output_data_dict, 5, 1e-5))
            
            record_thread.start()
            record_thread.join()
    
    def run_multi_thread_test_predictor(
        self, predictor, input_data_dict: dict, output_data_dict: dict, repeat=5, delta=1e-5
    ):
        """
        test paddle predictor
        多线程TensorRT预测器，max_batch_size=4,预测batch_size=1
        Args:
            predictor: paddle inference predictor
            repeat(int) : inference repeat time, set to catch gpu mem
        Returns:
            None
        """
        input_names = predictor.get_input_names()
        for _, input_data_name in enumerate(input_names):
            input_handle = predictor.get_input_handle(input_data_name)
            input_handle.copy_from_cpu(input_data_dict[input_data_name])

        for i in range(repeat):
            predictor.run()
        output_names = predictor.get_output_names()
        for i, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data = output_data.flatten()
            output_data_truth_val = output_data_dict[output_data_name].flatten()
            for j, out_data in enumerate(output_data):
                assert (
                    abs(out_data - output_data_truth_val[j]) <= delta
                ), f"{out_data} - {output_data_truth_val[j]} > {delta}"


def record_by_pid(pid: int, cuda_visible_device: int):
    """
    record_by_pid
    Args:
        pid(int) : pid of the process
        cuda_visible_device(int) : first gpu card of CUDA_VISIBLE_DEVICES
    Returns:
        gpu_max_mem(float): recorded max gpu mem
    """
    global _gpu_mem_lists

    while psutil.pid_exists(pid):
        gpu_mem = get_gpu_mem(cuda_visible_device)
        _gpu_mem_lists.append(gpu_mem)

    gpu_max_mem = max([float(i["used(MB)"]) for i in _gpu_mem_lists])
    return gpu_max_mem


def get_gpu_mem(gpu_id=0):
    """
    get gpu mem from gpu id
    Args:
        gpu_id(int): gpu id
    Returns:
        gpu_mem(dict): gpu infomartion
    """
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_utilization_info = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    gpu_mem = {}
    gpu_mem["total(MB)"] = gpu_mem_info.total / 1024.0 ** 2
    gpu_mem["free(MB)"] = gpu_mem_info.free / 1024.0 ** 2
    gpu_mem["used(MB)"] = gpu_mem_info.used / 1024.0 ** 2
    gpu_mem["gpu_utilization_rate(%)"] = gpu_utilization_info.gpu
    gpu_mem["gpu_mem_utilization_rate(%)"] = gpu_utilization_info.memory
    pynvml.nvmlShutdown()
    return gpu_mem
