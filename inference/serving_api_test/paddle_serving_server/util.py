"""
utils for test
"""
import os
import argparse
import base64
import subprocess
import pynvml


def kill_process(port, sleep_time=0):
    """kill process by port"""
    command = "kill -9 $(netstat -nlp | grep :" + str(port) + " | awk '{print $7}' | awk -F'/' '{{ print $1 }}')"
    os.system(command)
    # 解决端口占用
    os.system(f"sleep {sleep_time}")


def check_gpu_memory(gpu_id):
    """check gpu memory by gpu_id"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_used = mem_info.used / 1024 ** 2
    print(f"GPU-{gpu_id} memory used:", mem_used)
    return mem_used > 100


def count_process_num_on_port(port):
    """count process num"""
    command = "netstat -nlp | grep :" + str(port) + " | wc -l"
    count = eval(os.popen(command).read())
    print(f"port-{port} processes num:", count)
    return count


def check_keywords_in_server_log(words: str):
    """grep keywords in log"""
    p = subprocess.Popen(f"grep '{words}' stderr.log", shell=True)
    p.wait()
    return p.returncode == 0


def cv2_to_base64(image):
    """img to base64 str"""
    return base64.b64encode(image).decode("utf8")


def default_args():
    """default args"""
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.thread = 2
    args.port = 9292
    args.device = "cpu"
    args.gpu_ids = [""]
    args.op_num = 0
    args.op_max_batch = 32
    args.model = [""]
    args.workdir = "workdir"
    args.name = "None"
    args.use_mkl = False
    args.precision = "fp32"
    args.use_calib = False
    args.mem_optim_off = False
    args.ir_optim = False
    args.max_body_size = 512 * 1024 * 1024
    args.use_encryption_model = False
    args.use_multilang = False
    args.use_trt = False
    args.use_lite = False
    args.use_xpu = False
    args.product_name = None
    args.container_id = None
    args.gpu_multi_stream = False
    return args
