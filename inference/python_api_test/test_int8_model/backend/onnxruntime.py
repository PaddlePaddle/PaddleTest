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
import numpy as np
import onnxruntime as ort


class ONNXRuntimeEngine(object):
    """
    ONNXRuntime instance
    """

    def __init__(
        self,
        onnx_model_file,
        use_mkldnn=False,
        device="CPU",
        save_optimized_model=False,
    ):
        """set AnalysisConfig, generate AnalysisPredictor
        Args:
            onnx_model_file (str): root path of ONNX model.
            use_mkldnn (bool): whether use MKLDNN or not in CPU.
            device (str): Choose the device you want to run, it can be: CPU/GPU, default is CPU.
            save_optimized_model (bool): whether save optimized model to debug.
        """
        sess_options = ort.SessionOptions()
        if device == "CPU":
            if use_mkldnn:
                providers = ["DnnlExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        elif device == "GPU":
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ]

        if save_optimized_model:
            sess_options.optimized_model_filepath = "./optimize_model.onnx"
        self.sess = ort.InferenceSession(onnx_model_file, providers=providers, sess_options=sess_options)

    def prepare_data(self, input_data):
        """
        Prepare data
        """
        self.data_input = {}
        inputs_name = [a.name for a in self.sess.get_inputs()]
        assert len(input_data) == len(inputs_name)
        for i, k in enumerate(inputs_name):
            self.data_input[k] = np.array(input_data[i])

    def run(self):
        """
        Run inference.
        """
        return self.sess.run(None, self.data_input)
