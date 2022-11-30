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

import os

import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

paddle.disable_signal_handler()

class PaddleInferenceEngine(object):
    """
    Paddle Inference instance
    """

    def __init__(
        self,
        model_dir,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams",
        precision="fp32",
        use_trt=False,
        use_mkldnn=False,
        batch_size=1,
        device="CPU",
        min_subgraph_size=3,
        use_dynamic_shape=False,
        cpu_threads=1,
    ):
        """set AnalysisConfig, generate AnalysisPredictor
        Args:
            model_dir (str): root path of model.pdmodel and model.pdiparams.
            precision (str): mode of running(fp32/fp16/int8).
            use_trt (bool): whether use TensorRT or not.
            use_mkldnn (bool): whether use MKLDNN or not in CPU.
            batch_size (int): Batch size of infer sample.
            device (str): Choose the device you want to run, it can be: CPU/GPU, default is CPU.
            min_subgraph_size (int): min subgraph size in trt.
            use_dynamic_shape (bool): use dynamic shape or not.
            cpu_threads (int): num of thread when use CPU.
        """
        self.rerun_flag = False
        if device != "GPU" and use_trt:
            raise ValueError(
                "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".format(precision, device)
            )
        config = Config(os.path.join(model_dir, model_filename), os.path.join(model_dir, params_filename))
        if device == "GPU":
            # initial GPU memory(M), device ID
            config.enable_use_gpu(200, 0)
            # optimize graph and fuse op
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(cpu_threads)
            config.switch_ir_optim()
            if use_mkldnn:
                config.enable_mkldnn()
                if precision == "int8":
                    config.enable_mkldnn_int8({"conv2d", "depthwise_conv2d", "pool2d", "transpose2", "elementwise_mul"})
                if precision == "bf16":
                    config.enable_mkldnn_bfloat16()

                if precision == "bf16":
                    config.enable_mkldnn_bfloat16()

        if use_trt:
            if precision == "bf16":
                print("paddle trt does not support bf16, switching to fp16.")
                precision = "fp16"

            precision_map = {
                "int8": Config.Precision.Int8,
                "fp32": Config.Precision.Float32,
                "fp16": Config.Precision.Half,
            }
            assert precision in precision_map.keys()

            if use_dynamic_shape:
                dynamic_shape_file = os.path.join(model_dir, "dynamic_shape.txt")
                if os.path.exists(dynamic_shape_file):
                    config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
                    print("trt set dynamic shape done!")
                else:
                    # In order to avoid memory overflow when collecting dynamic shapes, it is changed to use CPU.
                    config.disable_gpu()
                    config.set_cpu_math_library_num_threads(10)
                    config.collect_shape_range_info(dynamic_shape_file)
                    print("Start collect dynamic shape...")
                    self.rerun_flag = True

            if not self.rerun_flag:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    max_batch_size=batch_size,
                    min_subgraph_size=min_subgraph_size,
                    precision_mode=precision_map[precision],
                    use_static=True,
                    use_calib_mode=False,
                )

        # enable shared memory
        config.enable_memory_optim()
        self.predictor = create_predictor(config)
        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handles = [self.predictor.get_output_handle(name) for name in self.predictor.get_output_names()]
        print("[Paddle Inference Backend] Completed PaddleInferenceEngine init ...")

    def prepare_data(self, input_data):
        """
        Prepare data
        """
        for input_field, input_handle in zip(input_data, self.input_handles):
            input_handle.copy_from_cpu(input_field)

    def run(self):
        """
        Run inference.
        """
        self.predictor.run()
        output = [output_handle.copy_to_cpu() for output_handle in self.output_handles]
        return output
