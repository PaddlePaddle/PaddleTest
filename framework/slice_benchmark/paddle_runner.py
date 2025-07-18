# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from runner_base import SliceTestCaseRunner

import paddle


class SliceTestCasePaddleRunner(SliceTestCaseRunner):
    def _prepare_data(self, shape, dtype, device, device_id=None, require_grad=False):
        if device == "gpu" or device == "cuda":
            device_id = 0 if device_id is None else device_id
            paddle_place = paddle.CUDAPlace(device_id)
        else:
            raise NotImplementedError(f"Unsupported device: {device}")

        np_data = np.random.randint(0, 100, size=shape)
        np_data = np_data.astype(dtype)
        tensor = paddle.to_tensor(np_data.copy(), place=paddle_place)
        tensor.stop_gradient = not require_grad

        return np_data, tensor

    def _set_device(self):
        if self.device == "gpu" or self.device == "cuda":
            paddle.set_device(f"gpu:{self.device_id}")
        else:
            raise NotImplementedError

    def convert_numpy(self, data):
        if isinstance(data, np.ndarray):
            return paddle.to_tensor(data).cuda(self.device_id)
        elif isinstance(data, list):
            return [self.convert_numpy(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.convert_numpy(item) for item in data)
        else:
            return data


class SliceGradTestCasePaddleRunner(SliceTestCasePaddleRunner):
    def _do_prepare(self):
        super()._do_prepare()

        # prepeare forward
        self.tensor_input.stop_gradient = False
        if self.is_setitem:
            self.z = self.tensor_input * 1
            self.z[self.index] = self.value
        else:
            self.z = self.tensor_input[self.index]
        self.grad_out = paddle.ones_like(self.z).cuda(self.device_id)

    def run(self):
        grad_x = paddle.grad(
            [self.z],
            [self.tensor_input],
            grad_outputs=self.grad_out,
            allow_unused=True,
        )

    def clear(self):
        del self.z
        del self.grad_out

        super().clear()
        # paddle.device.cuda.empty_cache()
