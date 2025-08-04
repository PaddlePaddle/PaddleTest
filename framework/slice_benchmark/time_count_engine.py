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

#
import os
import sys

import numpy as np

# Speed up Paddle backward
os.environ["FLAGS_share_tensor_for_grad_tensor_holder"] = "True"

from pprint import pprint


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_case import SliceTestCase, generate_test_cases


def create_case_runner(testcase: SliceTestCase, device, device_id):
    if testcase.framework == "paddle":
        from paddle_runner import (
            SliceGradTestCasePaddleRunner,
            SliceTestCasePaddleRunner,
        )

        if testcase.is_grad:
            return SliceGradTestCasePaddleRunner(testcase, device, device_id)
        else:
            return SliceTestCasePaddleRunner(testcase, device, device_id)
    elif testcase.framework == "torch":
        from torch_runner import (
            SliceGradTestCaseTorchRunner,
            SliceTestCaseTorchRunner,
        )

        if testcase.is_grad:
            return SliceGradTestCaseTorchRunner(testcase, device, device_id)
        else:
            return SliceTestCaseTorchRunner(testcase, device, device_id)


class SliceBenchMark:
    """
    构建Layer评估的性能通用类
    """

    def __init__(self):
        """
        初始化
        """
        self.enable_paddle = False
        self.enable_torch = False
        self._set_env()
        self._set_seed()
        self._load_case()
        self._set_device()

        self.perf_res = {}

    def _set_env(self):

        self.device = os.environ.get("SLICE_BENCHMARK_DEVICE", "gpu").lower()
        self.device_id = int(os.environ.get("SLICE_BENCHMARK_DEVICE_ID", "0"))
        self.n_repeat = int(os.environ.get("SLICE_BENCHMARK_REPEAT", "50"))
        self.n_warmup = int(os.environ.get("SLICE_BENCHMARK_WARMUP", "5"))
        self.seed = int(os.environ.get("SLICE_BM_SEED", "2025"))  # 随机种子
        frameworks = os.environ.get("SLICE_BENCHMARK_FRAMEWORKS", "paddle").lower()

        self.frameworks = []
        if "paddle" in frameworks:
            self.frameworks.append("paddle")
            self.enable_paddle = True
        if "torch" in frameworks:
            self.frameworks.append("torch")
            self.enable_torch = True

    def _set_seed(self):
        np.random.seed(self.seed)

    def _set_device(self):
        if self.device == "gpu" or self.device == "cuda":
            if self.enable_paddle:
                import paddle
                paddle.set_device(f"gpu:{self.device_id}")
            if self.enable_torch:
                import torch                
                torch.cuda.set_device(self.device_id)

    def _load_case(self):
        self.cases = generate_test_cases(frameworks=self.frameworks)
        cases_names = [case.name for case in self.cases]
        assert len(set(cases_names)) == len(cases_names), "Duplicate cases Error"
        pprint(f"Successfully load {len(self.cases)} cases:")
        pprint(cases_names)

    def perf(self):
        for case in self.cases:
            try:
                self.perf_res[case.name] = self.perf_single_case(case)
            except Exception as e:
                print(f"Failed to run case {case.name}: {e}")
                self.perf_res[case.name] = "fail"
            else:
                print(f"{case.name} perf: {self.perf_res[case.name]} ms")

        return self.perf_res

    def perf_single_case(self, case: SliceTestCase):
        """slice perf"""
        runner = create_case_runner(case, self.device, self.device_id)
        if case.framework == "paddle":
            import paddle
            start_event = [paddle.device.Event(enable_timing=True) for _ in range(self.n_repeat)]
            end_event = [paddle.device.Event(enable_timing=True) for _ in range(self.n_repeat)]
            sync_api = paddle.device.synchronize
        elif case.framework == "torch":
            import torch
            start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.n_repeat)]
            end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.n_repeat)]
            sync_api = torch.cuda.synchronize
        sync_api()
        # warmup
        for _ in range(self.n_warmup):
            runner.run()

        sync_api()

        # 开始统计耗时
        for i in range(self.n_repeat):
            start_event[i].record()
            runner.run()
            end_event[i].record()
        sync_api()

        total_time_array = np.array([s.elapsed_time(e) for s, e in zip(start_event, end_event)])

        # 对性能数据进行处理
        # ...
        # print(f"{runner.name}: {total_time_list}")
        runner.clear()
        return float(total_time_array.mean())


if __name__ == "__main__":
    bm = SliceBenchMark()
    res = bm.perf()
    pass
