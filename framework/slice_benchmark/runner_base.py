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

from __future__ import annotations


def get_same_shape(shape1, shape2):
    size1 = len(shape1)
    size2 = len(shape2)
    new_shape = []
    i = 1
    while i <= min(size1, size2):
        if shape1[size1 - i] == shape2[size2 - i]:
            new_shape.append(shape1[size1 - i])
            i += 1
        else:
            break
    return new_shape[::-1]


class SliceTestCaseRunner:
    def __init__(self, test_case, device, device_id) -> None:
        self.device = device
        self.device_id = device_id

        self.input_shape = test_case.input_shape
        self.dtype = test_case.dtype
        self.framework = test_case.framework
        self.is_setitem = test_case.is_setitem
        self.index = test_case.index
        self.name = test_case.name
        if self.is_setitem:
            self.value_shape = (
                test_case.value_shape if test_case.value_shape is not None and len(test_case.value_shape) > 0 else None
            )
            self.is_tensor = test_case.is_tensor

        self._do_prepare()

    def _do_prepare(self):
        self.np_input, self.tensor_input = self._prepare_data(
            self.input_shape,
            self.dtype,
            self.device,
            self.device_id,
        )
        self.np_index = self.index
        # self.index = self.convert_numpy(self.index)
        if self.is_setitem:
            np_infer_shape = self.np_input[self.np_index].shape
            if self.value_shape is None or len(self.value_shape) == 0:
                self.value_shape = get_same_shape(np_infer_shape, self.input_shape)
            if self.is_tensor:
                self.np_value, self.value = self._prepare_data(
                    self.value_shape,
                    self.dtype,
                    self.device,
                    self.device_id,
                )
            else:
                self.value = 5.0 if self.dtype.startswith("float") else 5

    def run(self):
        if self.is_setitem:
            self.tensor_input[self.index] = self.value
        else:
            self.output_tensor = self.tensor_input[self.index]

    def name(self):
        return self.name

    def _set_device(self):
        raise NotImplementedError

    def _prepare_data(self, shape, dtype, device, device_id=None, require_grad=False):
        raise NotImplementedError

        # if device == "cpu":
        #     torch_device = torch.device("cpu")
        #     paddle_device = paddle.CPUdevice()
        # elif device == "gpu" or device == "cuda":
        #     device_id = 0 if device_id is None else device_id
        #     torch_device = torch.device(f"cuda:{device_id}")
        #     paddle_device = paddle.CUDAdevice(device_id)
        # else:
        #     raise NotImplementedError(f"Unsupported device: {device}")

        # np_data = np.random.randint(0, 100, size=shape)
        # np_data.astype(dtype)
        # if framework == "paddle":
        #     tensor = paddle.to_tensor(np_data.copy(), device=paddle_device)
        # elif framework == "torch":
        #     tensor = torch.tensor(np_data.copy(), device=torch_device)
        # else:
        #     raise NotImplementedError(f"Unsupported framework: {framework}")

        # return np_data, tensor

    def convert_numpy(self, data):
        raise NotImplementedError
        # if isinstance(data, np.ndarray):
        #     if frame_name == "paddle":
        #         return paddle.to_tensor(data).cuda(cuda_device_num)
        #     elif frame_name == "torch":
        #         return torch.tensor(data).cuda(cuda_device_num)
        #     else:
        #         raise NotImplementedError
        # elif isinstance(data, list):
        #     return [convert_numpy(item) for item in data]
        # elif isinstance(data, tuple):
        #     return tuple(convert_numpy(item) for item in data)
        # else:
        #     return data

    def clear(self):
        to_clear = ["tensor_input", "value", "output_tensor", "index"]
        for attr in to_clear:
            if hasattr(self, attr):
                del self.__dict__[attr]
