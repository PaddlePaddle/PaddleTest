"""
XPU backend for Paddle inference
"""

import os
import sys
import time
import backend
import numpy as np
from common import getdtype

try:
    import paddle
    from paddle.inference import XpuConfig
    import paddle.inference as paddle_infer
except Exception as e:
    sys.stderr.write("Cannot import paddle, maybe paddle is not installed.\n")

paddle.disable_signal_handler()


class BackendPaddle(backend.Backend):
    """XPU backend"""

    def __init__(self):
        super(BackendPaddle, self).__init__()
        self.h2d_time = []
        self.compute_time = []
        self.d2h_time = []
        self.input_tensors = []

    def version(self):
        # paddle.version.commit
        return paddle.version.full_version

    def name(self):
        return "paddle"

    def load(self, config_arg, inputs=None, outpus=None):
        self.args = config_arg
        if os.path.exists(self.args.model_dir):
            model_file = os.path.join(self.args.model_dir + "/" + self.args.paddle_model_file)
            model_params = os.path.join(self.args.model_dir + "/" + self.args.paddle_params_file)
            config = paddle_infer.Config(model_file, model_params)
        else:
            raise ValueError(f"The model dir {self.args.model_dir} does not exists!")

        # enable memory optim
        config.enable_memory_optim()
        # config.disable_gpu()
        # config.set_cpu_math_library_num_threads(self.args.cpu_threads)
        # Enable to use Kunlun XPU
        print("Enable to use Kunlun XPU", self.args.gpu_id)
        config.enable_xpu()
        # ernie等模型暂时需要禁用该pass，待后续修复
        config.delete_pass("embedding_with_eltwise_add_xpu_fuse_pass")
        xpu_config = XpuConfig()
        xpu_config.device_id = self.args.gpu_id
        l3_size = 0
        xpu_config.l3_size = l3_size
        xpu_config.l3_autotune_size = 0
        config.set_xpu_config(xpu_config)
        self.predictor = paddle_infer.create_predictor(config)

        input_shape = self.args.yaml_config["input_shape"]
        if len(input_shape) <= 0:
            raise Exception("input shape is empty.")

        if "input_data" in self.args.yaml_config:
            input_file = self.args.yaml_config["input_data"]["data"][self.args.test_num]
            self.numpy_input = np.load(input_file, allow_pickle=True)

        # prepare input
        self.prepare_input()
        return self

    def prepare_input(self):
        """prepare input"""
        # set input tensor
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
            if "input_data" not in self.args.yaml_config:
                if self.args.yaml_config["input_shape"][str(i)]["shape"][self.args.test_num][0] == -1:
                    input_shape = [self.args.batch_size] + self.args.yaml_config["input_shape"][str(i)]["shape"][
                        self.args.test_num
                    ][1:]
                    dtype = self.args.yaml_config["input_shape"][str(i)]["dtype"][self.args.test_num]
                else:
                    input_shape = self.args.yaml_config["input_shape"][str(i)]["shape"][self.args.test_num]
                    dtype = self.args.yaml_config["input_shape"][str(i)]["dtype"][self.args.test_num]
                if hasattr(self.args, "test_data"):
                    fake_input = self.args.test_data[i].astype(getdtype(dtype))
                else:
                    fake_input = np.random.uniform(0, 1, size=input_shape).astype(getdtype(dtype))
                self.input_tensors.append(fake_input)
            else:
                if self.args.yaml_config["input_shape"][str(i)]["shape"][self.args.test_num][0] == -1:
                    real_input = np.expand_dims(self.numpy_input[i], 0).repeat(self.args.batch_size, axis=0)
                else:
                    real_input = np.tile(self.numpy_input[i], self.args.batch_size)
                self.input_tensors.append(real_input)

    def set_input(self):
        """set input"""
        # set input tensor
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(self.input_tensors[i])

    def set_output(self):
        """set output"""
        results = []
        # get out data from output tensor
        output_names = self.predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = self.predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            # print(np.std(output_data), np.mean(output_data))
            if self.args.return_result or self.args.save_result:
                results.append(output_data)
        if self.args.return_result or self.args.save_result:
            return results

    def reset(self):
        """reset func"""
        self.h2d_time.clear()
        self.d2h_time.clear()
        self.compute_time.clear()

    def warmup(self):
        # for i range(self.args.warmup):
        #     self.predictor.run()
        pass

    def predict(self, feed=None):
        self.set_input()
        self.predictor.run()
        output = self.set_output()
        if self.args.return_result or self.args.save_result:
            return output


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
    runner.report()
