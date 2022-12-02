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
import onnxruntime as ort


class ONNXRuntimeEngine(object):
    """
    ONNXRuntime instance
    """

    def __init__(
        self,
        model_dir,
        precision="fp32",
        use_trt=False,
        use_mkldnn=False,
        batch_size=1,
        device="CPU",
        min_subgraph_size=3,
        save_optimized_model=False,
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
        if device != "GPU" and use_trt:
            raise ValueError(
                "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".format(precision, device)
            )
        sess_options = ort.SessionOptions()
        if device == "CPU":
            providers = ["CPUExecutionProvider"]
        elif device == "GPU":
            providers = ["CUDAExecutionProvider"]
        elif use_mkldnn:
            providers = ["DnnlExecutionProvider"]
        elif use_trt:
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_max_workspace_size": 1073741824,
                        "trt_min_subgraph_size": min_subgraph_size,
                        "trt_fp16_enable": True if precision == "fp16" else False,
                        "trt_int8_enable": True if precision == "int8" else False,
                    },
                ),
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
            ]

        if save_optimized_model:
            sess_options.optimized_model_filepath = "./optimize_model.onnx"
        self.sess = ort.InferenceSession(model_dir, providers=providers, sess_options=sess_options)

    def prepare_data(self, input_data):
        """
        Prepare data
        """
        self.data_input = {}
        inputs_name = [a.name for a in sess.get_inputs()]
        assert len(input_data) == len(inputs_name)
        for i, k in enumerate(inputs_name):
            self.data_input[k] = np.array(input_data[i])

    def run(self):
        """
        Run inference.
        """
        return sess.run(None, self.data_input)
