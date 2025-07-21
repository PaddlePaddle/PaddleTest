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

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass
class SliceTestCase:
    name: str
    input_shape: tuple[int, ...]
    index: Any
    value_shape: tuple[int, ...] = ()
    is_setitem: bool = False
    is_tensor: bool = False
    dtype: str = ""
    framework: str = ""
    notes: str = ""
    skip_grad_test: bool = False
    is_grad: bool = False


base_case = [
    # --- 'scalar' cases ---
    SliceTestCase(
        name="Scalar - Integer",
        input_shape=(108, 64, 12288),
        index=0,
    ),
    SliceTestCase(
        name="Scalar - Tuple of Integers",
        input_shape=(108, 64, 12288),
        index=(2, 2, -1),
    ),
    # --- 'slice' cases ---
    # SliceTestCase(
    #     name="Slice - Full Slice",
    #     input_shape=(108, 64, 12288),
    #     index=slice(None, None, None),
    # ),
    SliceTestCase(
        name="Slice - Slice with Step",
        input_shape=(108, 64, 12288),
        index=slice(0, 100, 2),
    ),
    SliceTestCase(
        name="Slice - Tuple of Slices",
        input_shape=(108, 64, 12288),
        index=(slice(0, 4, 2), slice(None, None, None), slice(1, -1, None)),
    ),
    # --- 'none' cases ---
    # SliceTestCase(
    #     name="None - Single None for Dimension Expansion",
    #     input_shape=(108, 64, 12288),
    #     index=None,
    # ),
    SliceTestCase(
        name="None - Combined with Integers for Expansion",
        input_shape=(108, 64, 12288),
        index=(0, 0, 0, None),
    ),
    # --- 'ellipsis' cases ---
    SliceTestCase(
        name="Ellipsis - Single Ellipsis",
        input_shape=(108, 64, 12288),
        index=Ellipsis,
    ),
    # --- 'tuple' cases ---
    # SliceTestCase(
    #     name="Tuple - Empty Tuple",
    #     input_shape=(108, 64, 12288),
    #     index=(),
    # ),
    SliceTestCase(
        name="Tuple - Combined Int/Slice/None",
        input_shape=(108, 64, 12288),
        index=(0, slice(None, None, None), 0, None),
    ),
    # --- 'bool' cases ---
    SliceTestCase(
        name="Bool - Single True",
        input_shape=(108, 64, 12288),
        index=True,
    ),
    SliceTestCase(
        name="Bool - 1D Boolean Array Mask",
        input_shape=(108, 64, 12288),
        index=np.ones((108), dtype=bool),
    ),
    SliceTestCase(
        name="Bool - 2D Boolean Array Mask",
        input_shape=(108, 64, 12288),
        index=np.ones((108, 64), dtype=bool),
    ),
    SliceTestCase(
        name="Bool - Full Boolean Array Mask",
        input_shape=(108, 64, 12288),
        index=np.ones((108, 64, 12288), dtype=bool),
    ),
    # --- 'list' cases ---
    SliceTestCase(
        name="List - Simple Integer List",
        input_shape=(108, 64, 12288),
        index=[1, 0, 2],
    ),
    SliceTestCase(
        name="List - Tuple of Integer Lists for Advanced Indexing",
        input_shape=(108, 64, 12288),
        index=([0, 1], [3, 2], [0, 2]),
    ),
    # --- 'tensor' cases ---
    SliceTestCase(
        name="Tensor - 3D Integer Tensor",
        input_shape=(108, 64, 12288),
        index=np.ones((2, 4, 6), dtype=np.int64),
    ),
    SliceTestCase(
        name="Tensor - 0D Integer Tensor (Scalar)",
        input_shape=(108, 64, 12288),
        index=np.ones((), dtype=np.int64),
    ),
    SliceTestCase(
        name="Tensor - Tuple of 1D Integer Tensors",
        input_shape=(108, 64, 12288),
        index=(np.ones((2), dtype=np.int64), np.ones((2), dtype=np.int64)),
    ),
    # --- 'combined' cases (the most complex ones with special logic) ---
    SliceTestCase(
        name="Combined - Slice/Int/List",
        input_shape=(108, 64, 12288),
        index=(slice(None, None, None), 3, [0, 2]),
    ),
    SliceTestCase(
        name="Combined - Slice/List/None/Int with Broadcast",
        input_shape=(108, 64, 12288),
        index=(
            slice(None, None, None),
            [
                0,
            ],
            None,
            0,
        ),
        # This corresponds to the `if i == 2`特判 for the first dict.
        # It tests broadcasting a scalar tensor to a slice.
        value_shape=(1,),
        notes="Tests scalar broadcasting for set_item",
    ),
    # SliceTestCase(
    #     name="Combined - Slice/List/None with Broadcast",
    #     input_shape=(108, 64, 12288),
    #     index=(
    #         slice(None, None, None),
    #         [
    #             0,
    #         ],
    #         None,
    #     ),
    #     # This corresponds to the `elif i == 3` 特判.
    #     value_shape=(
    #         108,
    #         1,
    #         1,
    #         1,
    #     ),
    #     skip_grad_test=True,  # As noted in original code, skip grad test if it's known to fail
    # ),
    # SliceTestCase(
    #     name="Combined - Slice/List/Int",
    #     input_shape=(108, 64, 12288),
    #     index=(
    #         slice(None, None, None),
    #         [
    #             0,
    #         ],
    #         0,
    #     ),
    # ),
    SliceTestCase(
        name="Combined - Bool/Slice/Int",
        input_shape=(108, 64, 12288),
        index=(np.ones((108), dtype=bool), slice(None, None, None), -1),
    ),
    SliceTestCase(
        name="Combined - Slice/Int/List (variant)",
        input_shape=(108, 64, 12288),
        index=(slice(0, 4, 2), 3, [0, 2]),
    ),
    # ==========================================================================
    # 来源：second_index_dict，输入张量形状 (108, 64, 12288, 3)
    # ==========================================================================
    SliceTestCase(
        name="4D Combined - Int/List/Slice/Tensor",
        input_shape=(108, 64, 12288, 3),
        index=(
            1,
            [1, 2],
            slice(None, None, None),
            np.ones((2), dtype=np.int64),
        ),
    ),
    # SliceTestCase(
    #     name="4D Combined - Slice/List/Slice/Int",
    #     input_shape=(108, 64, 12288, 3),
    #     index=(slice(None, None, None), [1, 2], slice(None, None, None), 1),
    #     # This corresponds to the `if i == 2` 特判 for the second dict.
    #     # It's a hardcoded shape, likely for robustness as discussed.
    #     value_shape=(
    #         108,
    #         2,
    #         12288,
    #     ),
    #     notes="Special case with hardcoded value_shape for robustness.",
    # ),
    SliceTestCase(
        name="4D Combined - Slice/List/Slice/List",
        input_shape=(108, 64, 12288, 3),
        index=(slice(None, None, None), [1, 2], slice(None, None, None), [1]),
        notes="曾导致set_item_grad OOM",
        skip_grad_test=True,  # As noted, grad test might cause Out-Of-Memory
    ),
]

# base_case = [
#     SliceTestCase(
#         name="4D Combined - Slice/List/Slice/List",
#         input_shape=(108, 64, 12288, 3),
#         index=(slice(None, None, None), [1, 2], slice(None, None, None), [1]),
#         notes="曾导致set_item_grad OOM",
#         skip_grad_test=True,  # As noted, grad test might cause Out-Of-Memory
#     ),
# ]


def generate_test_cases(
    dtypes=["float16"],
    frameworks=["paddle", "torch"],
):
    """
    Generate test cases for different frameworks and dtypes.
    """
    # 生成不同 dytpes 的测试用例
    all_test_cases = []
    for case in base_case:
        for is_setitem in [True, False]:
            for is_grad in [True, False]:
                for is_tensor in [True, False]:
                    if not is_setitem and is_tensor:
                        continue
                    for dtype in dtypes:
                        for framework in frameworks:
                            api_name = "setitem" if is_setitem else "getitem"
                            if is_setitem and is_tensor:
                                api_name = "SetitemTensor"
                            elif is_setitem:
                                api_name = "Setitem"
                            else:
                                api_name = "Getitem"
                            grad_name = "backward" if is_grad else "forward"
                            case_with_dtype = replace(
                                case,
                                dtype=dtype,
                                framework=framework,
                                is_setitem=is_setitem,
                                is_tensor=is_tensor,
                                is_grad=is_grad,
                                name=f"{api_name} - {grad_name} - {case.name} - {dtype} - {framework}",
                            )
                            all_test_cases.append(case_with_dtype)
    return all_test_cases
