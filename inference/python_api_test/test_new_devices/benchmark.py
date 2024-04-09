"""
Inference benchmark
"""

import logging

import os
import time
import multiprocessing
import subprocess
import signal
import sys

import argparse
import numpy as np
import yaml
import psutil
import cpuinfo

from paddle.inference import Config
from paddle.inference import create_predictor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


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
    nu_opt = ",nounits"
    cpu_keys = ("cpu.util", "memory.util", "memory.used")


class Monitor(StatBase):
    """Monitor"""

    def __init__(self, use_gpu=False, gpu_id=0, interval=0.1):
        self.result = {}
        self.gpu_id = gpu_id
        self.use_gpu = use_gpu
        self.interval = interval

        self.cpu_stat_q = multiprocessing.Queue()

    def start(self):
        """start func"""
        cmd = "%s --id=%s --query-gpu=%s --format=csv,noheader%s -lms 50 > gpu_info.txt" % (
            StatBase.nvidia_smi_path,
            self.gpu_id,
            ",".join(StatBase.gpu_keys),
            StatBase.nu_opt,
        )
        # print(cmd)
        if os.path.exists("gpu_info.txt"):
            os.remove("gpu_info.txt")
        if self.use_gpu:
            self.gpu_stat_worker = subprocess.Popen(
                cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True, preexec_fn=os.setsid
            )

        # cpu stat
        pid = os.getpid()
        self.cpu_stat_worker = multiprocessing.Process(
            target=self.cpu_stat_func, args=(self.cpu_stat_q, pid, self.interval)
        )
        self.cpu_stat_worker.start()

    def stop(self):
        """stop monitor"""
        try:
            if self.use_gpu:
                os.killpg(self.gpu_stat_worker.pid, signal.SIGUSR1)
            # os.killpg(p.pid, signal.SIGTERM)
            self.cpu_stat_worker.terminate()
            self.cpu_stat_worker.join(timeout=0.01)
        except Exception as e:
            print(e)
            return

        # gpu
        if self.use_gpu:
            with open("gpu_info.txt", "r") as f:
                lines = f.readlines()
            # lines = self.gpu_stat_worker.stdout.readlines()
            # print(lines)
            lines = [line.strip() for line in lines if line.strip() != ""]
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
            self.result["gpu"] = result

        # cpu
        cpu_result = {}
        if self.cpu_stat_q.qsize() > 0:
            cpu_result = {k: v for k, v in zip(StatBase.cpu_keys, self.cpu_stat_q.get())}
        while not self.cpu_stat_q.empty():
            item = {k: v for k, v in zip(StatBase.cpu_keys, self.cpu_stat_q.get())}
            for k in StatBase.cpu_keys:
                cpu_result[k] = max(cpu_result[k], item[k])
        cpu_result["name"] = cpuinfo.get_cpu_info()["brand_raw"]
        self.result["cpu"] = cpu_result

    def output(self):
        """output func"""
        return self.result

    def cpu_stat_func(self, q, pid, interval=0.0):
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


def device_name(gpu_id):
    """get device name"""
    try:
        cmd = "nvidia-smi --id=%s --query-gpu=name --format=csv,noheader,nounits" % (gpu_id)
        return os.system(cmd)
    except Exception as e:
        return ""


def str2bool(v):
    """str2bool"""
    if v.lower() == "true":
        return True
    else:
        return False


def str2list(v):
    """str2list"""
    if len(v) == 0:
        return []

    return [list(map(int, item.split(","))) for item in v.split(":")]


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--type', required=True, choices=["cls", "shitu"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_shape", type=str2list, default=[])
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--inter_op_threads", type=int, default=1)
    parser.add_argument("--subgraph_size", type=int, default=3)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--backend_type", type=str, choices=["MLU", "NPU", "XPU", "DCU"], default="paddle")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--paddle_model_file", type=str, default="model.pdmodel")
    parser.add_argument("--paddle_params_file", type=str, default="model.pdiparams")
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--enable_openvino", type=str2bool, default=False)
    parser.add_argument("--enable_paddleort", type=str2bool, default=False)
    parser.add_argument("--enable_gpu", type=str2bool, default=False)
    parser.add_argument("--enable_pir", type=str2bool, default=False)
    parser.add_argument("--enable_trt", type=str2bool, default=False)
    parser.add_argument("--enable_dynamic_shape", type=str2bool, default=True)
    parser.add_argument("--enable_tune", type=str2bool, default=False)
    parser.add_argument("--gen_calib", type=str2bool, default=False)
    parser.add_argument("--enable_profile", type=str2bool, default=False)
    parser.add_argument("--enable_benchmark", type=str2bool, default=True)
    parser.add_argument("--save_result", type=str2bool, default=False)
    parser.add_argument("--return_result", type=str2bool, default=False)
    parser.add_argument("--enable_debug", type=str2bool, default=False)
    parser.add_argument("--enable_fd_paddle", type=str2bool, default=False)
    parser.add_argument("--enable_fd_trt", type=str2bool, default=False)
    parser.add_argument("--enable_fd_ort", type=str2bool, default=False)
    parser.add_argument("--enable_fd_openvino", type=str2bool, default=False)

    parser.add_argument("--config_file", type=str, default="config.yaml")
    parser.add_argument("--shape_range_file", type=str, default="shape_range.pbtxt")
    parser.add_argument("-sv", type=str, default="pytest args")
    parser.add_argument("-v", type=str, default="pytest args")
    parser.add_argument("-s", type=str, default="pytest args")
    args = parser.parse_args()
    return args


def get_backend(backend):
    """get backend"""
    if backend == "MLU":
        from backend_paddle_mlu import BackendPaddle

        backend = BackendPaddle()
    elif backend == "XPU":
        from backend_paddle_xpu import BackendPaddle

        backend = BackendPaddle()
    elif backend == "NPU":
        from backend_paddle_npu import BackendPaddle

        backend = BackendPaddle()
    elif backend == "DCU":
        from backend_paddle_dcu import BackendPaddle

        backend = BackendPaddle()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend


def parse_time(time_data, result_dict):
    """parse time"""
    time_data = np.sort(np.array(time_data))[25:75]
    # print(time_data)
    # print(f"方差：{np.var(time_data * 1000)}")
    # print(f"标准差：{np.std(time_data * 1000)}")
    # print(f"极差：{np.max(time_data * 1000) - np.min(time_data * 1000)}")
    if len(time_data) == 0:
        return result_dict
    percentiles = [50.0, 80.0, 90.0, 95.0, 99.0, 99.9]
    buckets = np.percentile(time_data, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b * 1000) for p, b in zip(percentiles, buckets)])
    # if result_dict["total"] == 0:
    result_dict["total"] = len(time_data)
    result_dict["result"] = {str(k): float(format(v * 1000, ".4f")) for k, v in zip(percentiles, buckets)}
    avg_cost = np.mean(time_data)
    result_dict["result"]["avg_cost"] = float(format(avg_cost * 1000, ".4f"))


def get_shape_str(shape_dict, idx=0):
    """get_shape_str"""
    shape_info = []
    try:
        for key, item in shape_dict.items():
            shape_str = ",".join([str(x) for x in item["shape"][idx]])
            shape_info.append(shape_str)
    except Exception as e:
        pass
    return ":".join(shape_info)


class BenchmarkRunner:
    """BenchmarkRunner"""

    def __init__(self, args):
        self.params = args
        self.out_diff = {}
        self.cpu_results = []
        self.warmup_times = 50
        self.run_times = 100
        self.time_data = []
        self.h2d_time = []
        self.d2h_time = []
        self.compute_time = []
        self.backend = None
        self.conf = None
        self.monitor = None
        self.result = None

    def preset(self, start_monitor=False):
        """preset func"""
        # if start_monitor:
        #     self.monitor = Monitor(self.conf.enable_gpu, self.conf.gpu_id)
        #     self.monitor.start()
        self.backend = get_backend(self.conf.backend_type)
        self.backend.load(self.conf)
        log.info("{}: {} model reload finish. ".format(self.conf.model_dir, self.conf.backend_type))
        self.time_data.clear()

    def get_cpu_results(self):
        """get_cpu_results"""
        if self.params.model_dir == "":
            print("need model_dir!!!")
            exit(8)
        self.input_dict = self.backend.input_tensors
        model_dir = self.params.model_dir
        print(model_dir)
        model_file = ""
        params_file = ""
        for file in os.listdir(model_dir):
            if file.endswith(".pdmodel"):
                model_file = f"{model_dir}/{file}"
            if file.endswith(".pdiparams"):
                params_file = f"{model_dir}/{file}"

        config = Config(model_file, params_file)
        # config.enable_memory_optim()
        config.switch_ir_optim(False)
        predictor = create_predictor(config)

        # copy img data to input tensor
        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()
        results = []

        for i, name in enumerate(input_names):
            # print(name)
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(self.input_dict[i])

        # do the inference
        predictor.run()

        # get out data from output tensor
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)

        # print(results)
        self.cpu_results = results

    def compare(self):
        """compare diff"""
        min_fp16 = 1e-4
        min_fp32 = 1e-15

        diff = 0
        sum_diff_0 = 0
        sum_diff_1 = 0
        sum_diff_2 = 0
        sum_diff_3 = 0
        sum = 0

        for i in range(len(self.results)):
            diff_0 = 0
            diff_1 = 0
            diff_2 = 0
            diff_3 = 0
            cpu_data = self.cpu_results[i].reshape(-1)
            xpu_data = self.results[i].reshape(-1)
            print(cpu_data.shape)
            sum += len(cpu_data)
            # for j in range(len(cpu_data)):
            for j, _ in enumerate(cpu_data):
                if xpu_data[j] < min_fp16 and cpu_data[j] < min_fp16:
                    diff = 0
                elif xpu_data[j] > min_fp16 and cpu_data[j] < min_fp32:
                    diff = 1
                else:
                    diff = abs(xpu_data[j] - cpu_data[j]) / abs(cpu_data[j])

                if diff < 0.1:
                    diff_0 += 1
                if diff < 0.01:
                    diff_1 += 1
                if diff < 0.001:
                    diff_2 += 1
                if diff < 0.0001:
                    diff_3 += 1

            print("output", i, ":")
            print("diff < 0.1 = %f " % (100.0 * diff_0 / len(cpu_data)))
            print("diff < 0.01 = %f " % (100.0 * diff_1 / len(cpu_data)))
            print("diff < 1e-3 = %f " % (100.0 * diff_2 / len(cpu_data)))
            print("diff < 1e-4 = %f " % (100.0 * diff_3 / len(cpu_data)))
            print("cosine similarity:", calc_cos_sim(xpu_data, cpu_data))

            self.out_diff["output_" + str(i) + "_diff_less_0.1"] = 100.0 * diff_0 / len(cpu_data)
            self.out_diff["output_" + str(i) + "_diff_less_0.01"] = 100.0 * diff_1 / len(cpu_data)

            sum_diff_0 += diff_0
            sum_diff_1 += diff_1
            sum_diff_2 += diff_2
            sum_diff_3 += diff_3

        print("Summary:")
        print("diff < 0.1 = %f " % (100.0 * sum_diff_0 / sum))
        print("diff < 0.01 = %f " % (100.0 * sum_diff_1 / sum))
        print("diff < 1e-3 = %f " % (100.0 * sum_diff_2 / sum))
        print("diff < 1e-4 = %f " % (100.0 * sum_diff_3 / sum))

        self.out_diff["sum_diff_less_0.1"] = 100.0 * sum_diff_0 / sum
        self.out_diff["sum_diff_less_0.01"] = 100.0 * sum_diff_1 / sum

        print(self.out_diff)

    def run(self):
        """run benchmark"""
        if self.conf.save_result:
            name_suffix = get_shape_str(self.conf.yaml_config["input_shape"], self.conf.test_num)
            output_save_path = os.path.join(
                self.conf.model_dir, "output", self.conf.model_dir.split("/")[-1] + name_suffix
            )
            if not os.path.exists(output_save_path):
                os.makedirs(output_save_path)
            output = self.backend.predict()
            import pickle

            with open(os.path.join(output_save_path, self.conf.precision + "_" + self.conf.backend_type), "wb") as f:
                pickle.dump(output, f)
            return

        if self.conf.return_result or self.conf.enable_tune or self.conf.gen_calib:
            output = self.backend.predict()
            self.results = output

        for i in range(self.warmup_times):
            self.backend.predict()

        run_count = 0
        min_run_time = 0
        self.backend.reset()
        while run_count < self.run_times:
            begin = time.time()
            self.backend.predict()
            local_time = time.time() - begin
            min_run_time += local_time
            run_count = run_count + 1
            self.time_data.append(local_time)

    def report(self, status=True):
        """report result"""
        perf_result = {}
        if self.monitor is not None:
            self.monitor.stop()
        parse_time(self.time_data, perf_result)

        print("##### benchmark result: #####")
        result = {}
        test_nums = len(self.conf.yaml_config["input_shape"]["0"]["shape"])
        name_suffix = "_" + str(self.conf.test_num) if test_nums > 1 else ""
        result["model_name"] = self.conf.model_dir.split("/")[-1] + name_suffix
        result["origin_name"] = self.conf.model_dir.split("/")[-1]
        result["status"] = "success" if status else "failure"
        result["detail"] = perf_result
        result["avg_cost"] = perf_result["result"]["avg_cost"] if perf_result else 0
        result["h2d_cost"] = float(format(np.mean(self.h2d_time), ".6f")) if len(self.h2d_time) > 0 else 0
        result["d2h_cost"] = float(format(np.mean(self.d2h_time), ".6f")) if len(self.d2h_time) > 0 else 0
        result["compute_cost"] = float(format(np.mean(self.compute_time), ".6f")) if len(self.compute_time) > 0 else 0
        result["stat"] = self.monitor.output() if self.monitor is not None else {}
        if self.conf.enable_gpu:
            result["device_name"] = (
                result["stat"]["gpu"]["name"]
                if result["stat"] and "gpu" in result["stat"]
                else device_name(self.conf.gpu_id)
            )
            result["gpu_mem"] = (
                result["stat"]["gpu"]["memory.used"] if result["stat"] and "gpu" in result["stat"] else 0
            )
        result["backend_type"] = self.conf.backend_type
        result["batch_size"] = self.conf.batch_size
        result["out_diff"] = self.out_diff
        result["precision"] = self.conf.precision
        result["cpu_threads"] = self.conf.cpu_threads
        result["cpu_mem"] = result["stat"]["cpu"]["memory.used"] if result["stat"] and "cpu" in result["stat"] else 0
        result["enable_mkldnn"] = self.conf.enable_mkldnn
        result["enable_gpu"] = self.conf.enable_gpu
        result["enable_pir"] = self.conf.enable_pir
        result["enable_trt"] = self.conf.enable_trt
        result["input_shape"] = get_shape_str(self.conf.yaml_config["input_shape"], self.conf.test_num)
        print(result)
        with open("result.txt", "a+") as f:
            f.write("model path: " + self.conf.model_dir + "\n")
            for key, val in result.items():
                f.write(key + " : " + str(val) + "\n")
            f.write("\n")
        self.result = result

    def test(self, conf, input_num=None):
        """test func"""
        self.conf = conf
        config_path = os.path.abspath(self.conf.model_dir + "/" + self.conf.config_file)
        if not os.path.exists(config_path):
            log.error("{} not found".format(config_path))
            sys.exit(1)
        try:
            fd = open(config_path)
        except Exception as e:
            raise ValueError("open config file failed.")
        yaml_config = yaml.load(fd, yaml.FullLoader)
        fd.close()
        self.conf.yaml_config = yaml_config

        if self.conf.save_result:
            test_num = len(self.conf.yaml_config["input_shape"]["0"]["shape"])
            for i in range(test_num):
                self.conf.test_num = i
                self.preset(False)
                self.run()
            return

        # benchmark loop with different input
        test_num = len(self.conf.yaml_config["input_shape"]["0"]["shape"])
        for i in range(test_num):
            # 对于具有多个输入shape的case，指定跑对应shape（diff case rerun）
            if input_num is not None and (i != input_num):
                continue
            self.conf.test_num = i
            try:
                self.preset()
                self.run()
                if "Pix2pix" not in self.conf.model_dir.split("/")[-1]:
                    self.get_cpu_results()
                    self.compare()
                self.report()
            except Exception as e:
                self.report(False)
                print(e)
                log.info("{}: {} benchmark failed!".format(self.conf.model_dir, self.conf.backend_type))
                raise Exception(e)


def calc_cos_sim(data1, data2):
    """calculate_cos_sim"""
    data1 = data1.reshape(-1)
    data2 = data2.reshape(-1)
    cos_sim = data1.dot(data2) / (np.linalg.norm(data1) * np.linalg.norm(data2))
    return cos_sim


def main():
    """main"""
    args = parse_args()
    runner = BenchmarkRunner(args)
    runner.test(args)


if __name__ == "__main__":
    main()
