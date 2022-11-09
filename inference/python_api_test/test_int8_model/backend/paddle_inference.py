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
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        cpu_threads=1,
    ):
        """set AnalysisConfig, generate AnalysisPredictor
        Args:
            model_dir (str): root path of __model__ and __params__
            precision (str): mode of running(fp32/fp16/int8)
            use_trt (bool): whether use TensorRT or not.
            use_mkldnn (bool): whether use MKLDNN or not in CPU.
            batch_size (int): Batch size of infer sample.
            device (str): Choose the device you want to run, it can be: CPU/GPU, default is CPU
            min_subgraph_size (int): min subgraph size in trt.
            use_dynamic_shape (bool): use dynamic shape or not
            trt_min_shape (int): min shape for dynamic shape in trt
            trt_max_shape (int): max shape for dynamic shape in trt
            trt_opt_shape (int): opt shape for dynamic shape in trt
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
                    pred_cfg.enable_mkldnn_int8(
                        {"conv2d", "depthwise_conv2d", "pool2d", "transpose2", "elementwise_mul"}
                    )

        precision_map = {
            "int8": Config.Precision.Int8,
            "fp32": Config.Precision.Float32,
            "fp16": Config.Precision.Half,
        }
        if precision in precision_map.keys() and use_trt:
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=batch_size,
                min_subgraph_size=min_subgraph_size,
                precision_mode=precision_map[precision],
                use_static=True,
                use_calib_mode=False,
            )

            if use_dynamic_shape:
                dynamic_shape_file = os.path.join(model_dir, "dynamic_shape.txt")
                if os.path.exists(dynamic_shape_file):
                    config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
                    print("trt set dynamic shape done!")
                else:
                    config.collect_shape_range_info(dynamic_shape_file)
                    print("Start collect dynamic shape...")
                    self.rerun_flag = True

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
