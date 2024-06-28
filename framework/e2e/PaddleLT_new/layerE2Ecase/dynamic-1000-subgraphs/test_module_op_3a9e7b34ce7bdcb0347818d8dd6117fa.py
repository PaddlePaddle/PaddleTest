import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
import sys
import unittest
import numpy as np
from dataclasses import dataclass
import typing as t

@dataclass
class Stage:
    name: str
    env_vars: t.Dict[str, str]

cinn_stages = [
    Stage(
        name="dynamic_to_static",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=False,
            FLAGS_prim_enable_dynamic=False,
        ),
    ),
    Stage(
        name="prim",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
        ),
    ),
    Stage(
        name="infer_symbolic",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=False,
            FLAGS_check_infer_symbolic=True,
        ),
    ),
	Stage(
        name="frontend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=True,
        ), 
    ),
    Stage(
        name="backend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=False,
        ), 
    ),
]

def GetCinnStageByName(name):
    for stage in cinn_stages:
        if stage.name == name:
            return stage
    return None

def GetCurrentCinnStage():
    name = os.getenv('PADDLE_DEBUG_CINN_STAGE_NAME')
    if name is None:
        return None
    stage_names = [stage.name for stage in cinn_stages]
    assert name in stage_names, (
        f"PADDLE_DEBUG_CINN_STAGE_NAME should be in {stage_names}"
    )
    return GetCinnStageByName(name)

def GetPrevCinnStage(stage):
    for i in range(1, len(cinn_stages)):
        if stage is cinn_stages[i]:
            return cinn_stages[i - 1]
    return None

def IsCinnStageEnableDiff():
    value = os.getenv('PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF')
    enabled = value in {
        '1',
        'true',
        'True',
    }
    if enabled:
        assert GetCurrentCinnStage() is not None
    return enabled

last_cinn_stage_exit_code = None
def LastCINNStageFailed():
    global last_cinn_stage_exit_code
    if last_cinn_stage_exit_code is not None:
        return last_cinn_stage_exit_code != 0
    last_stage = GetPrevCinnStage(GetCurrentCinnStage())
    if last_stage is None:
        return False
    env_vars = dict(
        PADDLE_DEBUG_CINN_STAGE_NAME=last_stage.name,
        PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
    )
    env_vars_str = " ".join(
        f"{env_var}={value}"
        for env_var, value in env_vars.items()
    )
    last_cinn_stage_exit_code = os.system(
        f"{env_vars_str} {sys.executable} {__file__} > /dev/null 2>&1"
    )
    return last_cinn_stage_exit_code != 0

def SetDefaultEnv(**env_var2value):
    for env_var, value in env_var2value.items():
        if os.getenv(env_var) is None:
            os.environ[env_var] = str(value)

SetDefaultEnv(
    PADDLE_DEBUG_ENABLE_CINN=True,
    FLAGS_enable_pir_api=True,
    FLAGS_prim_all=True,
    FLAGS_prim_enable_dynamic=True,
    FLAGS_use_cinn=False,
    FLAGS_check_infer_symbolic=False,
    FLAGS_enable_fusion_fallback=False,
)

import paddle

def SetEnvVar(env_var2value):
    for env_var, value in env_var2value.items():
        os.environ[env_var] = str(value)
    paddle.set_flags({
        env_var:value
        for env_var, value in env_var2value.items()
        if env_var.startswith('FLAGS_')
    })

if GetCurrentCinnStage() is not None:
    SetEnvVar(GetCurrentCinnStage().env_vars)

def NumOperationsInBlock(block_idx):
    return [160][block_idx] - 1 # number-of-ops-in-block

def GetPaddleDebugNumAllowedOps():
    try:
        return int(os.getenv('PADDLE_DEBUG_NUM_ALLOWED_OPS'))
    except:
        return None

paddle_debug_num_allowed_ops = GetPaddleDebugNumAllowedOps()


if type(paddle_debug_num_allowed_ops) is not int:
    def EarlyReturn(block_idx, op_idx):
        return False      
else:
    def EarlyReturn(block_idx, op_idx):
        return op_idx >= paddle_debug_num_allowed_ops

class BlockEntries:

    def builtin_module_0_0_0(self, data_0, data_1, data_2, data_3, data_4):

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_0 = paddle._C_ops.shape(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_0, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_0, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_0, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_3, paddle.int64)

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full([1], 1, paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_0, cast_0, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_1, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 8, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_3, 0, True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_0, cast_2, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_3 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_3, full_2, 0.5, True)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_3, 0, True)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [scale_3, scale_1]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_0, split_1, = meshgrid_0

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_1 = [split_1, split_0]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_4 = paddle._C_ops.cast(stack_0, paddle.float32)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [1, -1, 2]

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_0, reshape_1 = paddle.reshape(cast_4, full_int_array_5), None

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_0 = slice_2 * slice_3

        # pd_op.assign: (xi32) <- (xi32)
        assign_0 = multiply_0

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full([], 1, paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_5 = paddle._C_ops.cast(multiply_0, paddle.int64)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_2 = [full_4, cast_5, full_4]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(full_3, stack_1, paddle.float32)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_1 = paddle._C_ops.shape(data_1)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_1, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_1, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_1, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.cast: (xi64) <- (xi32)
        cast_6 = paddle._C_ops.cast(slice_7, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_2 = paddle.arange(full_0, cast_6, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_7 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_7, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], 16, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_5, 0, True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_8 = paddle._C_ops.cast(slice_6, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_3 = paddle.arange(full_0, cast_8, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_9 = paddle._C_ops.cast(arange_3, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_9, full_2, 0.5, True)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_5, 0, True)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_3 = [scale_7, scale_5]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_2, split_3, = meshgrid_1

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_4 = [split_3, split_2]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_10 = paddle._C_ops.cast(stack_2, paddle.float32)

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_2, reshape_3 = paddle.reshape(cast_10, full_int_array_5), None

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_1 = slice_6 * slice_7

        # pd_op.assign: (xi32) <- (xi32)
        assign_1 = multiply_1

        # pd_op.cast: (xi64) <- (xi32)
        cast_11 = paddle._C_ops.cast(multiply_1, paddle.int64)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [full_4, cast_11, full_4]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_5, 0)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_5, stack_3, paddle.float32)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_2 = paddle._C_ops.shape(data_2)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_2, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_2, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(shape_2, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_2, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.cast: (xi64) <- (xi32)
        cast_12 = paddle._C_ops.cast(slice_11, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_4 = paddle.arange(full_0, cast_12, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_13 = paddle._C_ops.cast(arange_4, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_13, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], 32, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_6, 0, True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_14 = paddle._C_ops.cast(slice_10, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_5 = paddle.arange(full_0, cast_14, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_15 = paddle._C_ops.cast(arange_5, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(cast_15, full_2, 0.5, True)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(scale_10, full_6, 0, True)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_6 = [scale_11, scale_9]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_4, split_5, = meshgrid_2

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_7 = [split_5, split_4]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_4 = paddle._C_ops.stack(combine_7, -1)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_16 = paddle._C_ops.cast(stack_4, paddle.float32)

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_4, reshape_5 = paddle.reshape(cast_16, full_int_array_5), None

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_2 = slice_10 * slice_11

        # pd_op.assign: (xi32) <- (xi32)
        assign_2 = multiply_2

        # pd_op.cast: (xi64) <- (xi32)
        cast_17 = paddle._C_ops.cast(multiply_2, paddle.int64)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_8 = [full_4, cast_17, full_4]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_8, 0)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(full_6, stack_5, paddle.float32)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_3 = paddle._C_ops.shape(data_3)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_3, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_3, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_3, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_3, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.cast: (xi64) <- (xi32)
        cast_18 = paddle._C_ops.cast(slice_15, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_6 = paddle.arange(full_0, cast_18, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_19 = paddle._C_ops.cast(arange_6, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(cast_19, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], 64, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(scale_12, full_7, 0, True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_20 = paddle._C_ops.cast(slice_14, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_7 = paddle.arange(full_0, cast_20, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_21 = paddle._C_ops.cast(arange_7, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(cast_21, full_2, 0.5, True)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(scale_14, full_7, 0, True)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_9 = [scale_15, scale_13]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_9)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_6, split_7, = meshgrid_3

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_10 = [split_7, split_6]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_6 = paddle._C_ops.stack(combine_10, -1)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_22 = paddle._C_ops.cast(stack_6, paddle.float32)

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_6, reshape_7 = paddle.reshape(cast_22, full_int_array_5), None

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_3 = slice_14 * slice_15

        # pd_op.assign: (xi32) <- (xi32)
        assign_3 = multiply_3

        # pd_op.cast: (xi64) <- (xi32)
        cast_23 = paddle._C_ops.cast(multiply_3, paddle.int64)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [full_4, cast_23, full_4]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_11, 0)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_3 = paddle._C_ops.full_with_tensor(full_7, stack_7, paddle.float32)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_4 = paddle._C_ops.shape(data_4)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_4, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_4, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(shape_4, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(shape_4, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.cast: (xi64) <- (xi32)
        cast_24 = paddle._C_ops.cast(slice_19, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_8 = paddle.arange(full_0, cast_24, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_25 = paddle._C_ops.cast(arange_8, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(cast_25, full_2, 0.5, True)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], 128, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(scale_16, full_8, 0, True)

        # pd_op.cast: (xi64) <- (xi32)
        cast_26 = paddle._C_ops.cast(slice_18, paddle.int64)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_9 = paddle.arange(full_0, cast_26, full_1, dtype=paddle.int64)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_27 = paddle._C_ops.cast(arange_9, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_27, full_2, 0.5, True)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_8, 0, True)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_12 = [scale_19, scale_17]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_12)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_8, split_9, = meshgrid_4

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_13 = [split_9, split_8]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_8 = paddle._C_ops.stack(combine_13, -1)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_28 = paddle._C_ops.cast(stack_8, paddle.float32)

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_8, reshape_9 = paddle.reshape(cast_28, full_int_array_5), None

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_4 = slice_18 * slice_19

        # pd_op.assign: (xi32) <- (xi32)
        assign_4 = multiply_4

        # pd_op.cast: (xi64) <- (xi32)
        cast_29 = paddle._C_ops.cast(multiply_4, paddle.int64)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_14 = [full_4, cast_29, full_4]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_14, 0)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_4 = paddle._C_ops.full_with_tensor(full_8, stack_9, paddle.float32)

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32]) <- (1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32)
        combine_15 = [reshape_0, reshape_2, reshape_4, reshape_6, reshape_8]

        # pd_op.concat: (1x-1x2xf32) <- ([1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_15, full_9)

        # builtin.combine: ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32]) <- (1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32)
        combine_16 = [full_with_tensor_0, full_with_tensor_1, full_with_tensor_2, full_with_tensor_3, full_with_tensor_4]

        # pd_op.concat: (1x-1x1xf32) <- ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_16, full_9)
        return concat_0, concat_1, assign_0, assign_1, assign_2, assign_3, assign_4



def GetEnvVarEnableJit():
    enable_jit = os.getenv('PADDLE_DEBUG_ENABLE_JIT')
    return enable_jit not in {
        "0",
        "False",
        "false",
        "OFF",
    }

def GetEnvVarEnableCinn():
    enable_cinn = os.getenv('PADDLE_DEBUG_ENABLE_CINN')
    return enable_cinn not in {
        "0",
        "False",
        "false",
        "OFF",
    }


def GetTolerance(dtype):
    if dtype == np.float16:
        return GetFloat16Tolerance()
    if dtype == np.float32:
        return GetFloat32Tolerance()
    return 1e-6

def GetFloat16Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT16_TOL'))
    except:
        return 1e-3

def GetFloat32Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT32_TOL'))
    except:
        return 1e-6

def IsInteger(dtype):
    return np.dtype(dtype).char in np.typecodes['AllInteger']


class CinnTestBase:
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def test_train(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            assert x_numpy.dtype == y_numpy.dtype
            if IsInteger(x_numpy.dtype):
                np.testing.assert_equal(x_numpy, y_numpy)
            else:
                tol = GetTolerance(x_numpy.dtype)
                np.testing.assert_allclose(x_numpy, y_numpy, atol=tol, rtol=tol)
        else:
            assert x == y

class Block_builtin_module_0_0_0(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        args = [data_0, data_1, data_2, data_3, data_4]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_shape_0,
            self.op_full_int_array_0,
            self.op_full_int_array_1,
            self.op_slice_0,
            self.op_full_int_array_2,
            self.op_slice_1,
            self.op_full_int_array_3,
            self.op_slice_2,
            self.op_full_int_array_4,
            self.op_slice_3,
            self.op_full_0,
            self.op_cast_0,
            self.op_full_1,
            self.op_arange_0,
            self.op_cast_1,
            self.op_full_2,
            self.op_scale_0,
            self.op_full_3,
            self.op_scale_1,
            self.op_cast_2,
            self.op_arange_1,
            self.op_cast_3,
            self.op_scale_2,
            self.op_scale_3,
            self.op_combine_0,
            self.op_meshgrid_0,
            self.op_split_0,
            self.op_combine_1,
            self.op_stack_0,
            self.op_cast_4,
            self.op_full_int_array_5,
            self.op_reshape_0,
            self.op_multiply_0,
            self.op_assign_0,
            self.op_full_4,
            self.op_cast_5,
            self.op_combine_2,
            self.op_stack_1,
            self.op_full_with_tensor_0,
            self.op_shape_1,
            self.op_slice_4,
            self.op_slice_5,
            self.op_slice_6,
            self.op_slice_7,
            self.op_cast_6,
            self.op_arange_2,
            self.op_cast_7,
            self.op_scale_4,
            self.op_full_5,
            self.op_scale_5,
            self.op_cast_8,
            self.op_arange_3,
            self.op_cast_9,
            self.op_scale_6,
            self.op_scale_7,
            self.op_combine_3,
            self.op_meshgrid_1,
            self.op_split_1,
            self.op_combine_4,
            self.op_stack_2,
            self.op_cast_10,
            self.op_reshape_1,
            self.op_multiply_1,
            self.op_assign_1,
            self.op_cast_11,
            self.op_combine_5,
            self.op_stack_3,
            self.op_full_with_tensor_1,
            self.op_shape_2,
            self.op_slice_8,
            self.op_slice_9,
            self.op_slice_10,
            self.op_slice_11,
            self.op_cast_12,
            self.op_arange_4,
            self.op_cast_13,
            self.op_scale_8,
            self.op_full_6,
            self.op_scale_9,
            self.op_cast_14,
            self.op_arange_5,
            self.op_cast_15,
            self.op_scale_10,
            self.op_scale_11,
            self.op_combine_6,
            self.op_meshgrid_2,
            self.op_split_2,
            self.op_combine_7,
            self.op_stack_4,
            self.op_cast_16,
            self.op_reshape_2,
            self.op_multiply_2,
            self.op_assign_2,
            self.op_cast_17,
            self.op_combine_8,
            self.op_stack_5,
            self.op_full_with_tensor_2,
            self.op_shape_3,
            self.op_slice_12,
            self.op_slice_13,
            self.op_slice_14,
            self.op_slice_15,
            self.op_cast_18,
            self.op_arange_6,
            self.op_cast_19,
            self.op_scale_12,
            self.op_full_7,
            self.op_scale_13,
            self.op_cast_20,
            self.op_arange_7,
            self.op_cast_21,
            self.op_scale_14,
            self.op_scale_15,
            self.op_combine_9,
            self.op_meshgrid_3,
            self.op_split_3,
            self.op_combine_10,
            self.op_stack_6,
            self.op_cast_22,
            self.op_reshape_3,
            self.op_multiply_3,
            self.op_assign_3,
            self.op_cast_23,
            self.op_combine_11,
            self.op_stack_7,
            self.op_full_with_tensor_3,
            self.op_shape_4,
            self.op_slice_16,
            self.op_slice_17,
            self.op_slice_18,
            self.op_slice_19,
            self.op_cast_24,
            self.op_arange_8,
            self.op_cast_25,
            self.op_scale_16,
            self.op_full_8,
            self.op_scale_17,
            self.op_cast_26,
            self.op_arange_9,
            self.op_cast_27,
            self.op_scale_18,
            self.op_scale_19,
            self.op_combine_12,
            self.op_meshgrid_4,
            self.op_split_4,
            self.op_combine_13,
            self.op_stack_8,
            self.op_cast_28,
            self.op_reshape_4,
            self.op_multiply_4,
            self.op_assign_4,
            self.op_cast_29,
            self.op_combine_14,
            self.op_stack_9,
            self.op_full_with_tensor_4,
            self.op_full_9,
            self.op_combine_15,
            self.op_concat_0,
            self.op_combine_16,
            self.op_concat_1,
        ]

    def op_shape_0(self, data_0, data_1, data_2, data_3, data_4):
    
        # EarlyReturn(0, 0)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_0 = paddle._C_ops.shape(data_0)

        return [data_1, data_2, data_3, data_4, shape_0]

    def op_full_int_array_0(self, data_1, data_2, data_3, data_4, shape_0):
    
        # EarlyReturn(0, 1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        return [data_1, data_2, data_3, data_4, shape_0, full_int_array_0]

    def op_full_int_array_1(self, data_1, data_2, data_3, data_4, shape_0, full_int_array_0):
    
        # EarlyReturn(0, 2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        return [data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1]

    def op_slice_0(self, data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1):
    
        # EarlyReturn(0, 3)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1]

    def op_full_int_array_2(self, data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1):
    
        # EarlyReturn(0, 4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        return [data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2]

    def op_slice_1(self, data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2):
    
        # EarlyReturn(0, 5)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_0, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2]

    def op_full_int_array_3(self, data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2):
    
        # EarlyReturn(0, 6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        return [data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3]

    def op_slice_2(self, data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3):
    
        # EarlyReturn(0, 7)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_0, [0], full_int_array_2, full_int_array_3, [1], [0])

        return [data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2]

    def op_full_int_array_4(self, data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2):
    
        # EarlyReturn(0, 8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        return [data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4]

    def op_slice_3(self, data_1, data_2, data_3, data_4, shape_0, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4):
    
        # EarlyReturn(0, 9)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_0, [0], full_int_array_3, full_int_array_4, [1], [0])

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3]

    def op_full_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3):
    
        # EarlyReturn(0, 10)

        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.int64, paddle.core.CPUPlace())

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0]

    def op_cast_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0):
    
        # EarlyReturn(0, 11)

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_3, paddle.int64)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, cast_0]

    def op_full_1(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, cast_0):
    
        # EarlyReturn(0, 12)

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full([1], 1, paddle.int64, paddle.core.CPUPlace())

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, cast_0, full_1]

    def op_arange_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, cast_0, full_1):
    
        # EarlyReturn(0, 13)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_0, cast_0, full_1, dtype=paddle.int64)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, arange_0]

    def op_cast_1(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, arange_0):
    
        # EarlyReturn(0, 14)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(arange_0, paddle.float32)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, cast_1]

    def op_full_2(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, cast_1):
    
        # EarlyReturn(0, 15)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, cast_1, full_2]

    def op_scale_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, cast_1, full_2):
    
        # EarlyReturn(0, 16)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_1, full_2, 0.5, True)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, scale_0]

    def op_full_3(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, scale_0):
    
        # EarlyReturn(0, 17)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 8, paddle.float32, paddle.core.CPUPlace())

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, scale_0, full_3]

    def op_scale_1(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, scale_0, full_3):
    
        # EarlyReturn(0, 18)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_3, 0, True)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1]

    def op_cast_2(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1):
    
        # EarlyReturn(0, 19)

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(slice_2, paddle.int64)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, cast_2]

    def op_arange_1(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, cast_2):
    
        # EarlyReturn(0, 20)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_0, cast_2, full_1, dtype=paddle.int64)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, arange_1]

    def op_cast_3(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, arange_1):
    
        # EarlyReturn(0, 21)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_3 = paddle._C_ops.cast(arange_1, paddle.float32)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, cast_3]

    def op_scale_2(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, cast_3):
    
        # EarlyReturn(0, 22)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_3, full_2, 0.5, True)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, scale_2]

    def op_scale_3(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, scale_2):
    
        # EarlyReturn(0, 23)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_3, 0, True)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, scale_3]

    def op_combine_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, scale_1, scale_3):
    
        # EarlyReturn(0, 24)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [scale_3, scale_1]

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, combine_0]

    def op_meshgrid_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, combine_0):
    
        # EarlyReturn(0, 25)

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, meshgrid_0]

    def op_split_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, meshgrid_0):
    
        # EarlyReturn(0, 26)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_0, split_1, = meshgrid_0

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, split_0, split_1]

    def op_combine_1(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, split_0, split_1):
    
        # EarlyReturn(0, 27)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_1 = [split_1, split_0]

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, combine_1]

    def op_stack_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, combine_1):
    
        # EarlyReturn(0, 28)

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, stack_0]

    def op_cast_4(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, stack_0):
    
        # EarlyReturn(0, 29)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_4 = paddle._C_ops.cast(stack_0, paddle.float32)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, cast_4]

    def op_full_int_array_5(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, cast_4):
    
        # EarlyReturn(0, 30)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [1, -1, 2]

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, cast_4, full_int_array_5]

    def op_reshape_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, cast_4, full_int_array_5):
    
        # EarlyReturn(0, 31)

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_0, reshape_1 = paddle.reshape(cast_4, full_int_array_5), None

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0]

    def op_multiply_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, slice_2, full_int_array_4, slice_3, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0):
    
        # EarlyReturn(0, 32)

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_0 = slice_2 * slice_3

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, multiply_0]

    def op_assign_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, multiply_0):
    
        # EarlyReturn(0, 33)

        # pd_op.assign: (xi32) <- (xi32)
        assign_0 = multiply_0

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, multiply_0, assign_0]

    def op_full_4(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, multiply_0, assign_0):
    
        # EarlyReturn(0, 34)

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full([], 1, paddle.int64, paddle.core.CPUPlace())

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, multiply_0, assign_0, full_4]

    def op_cast_5(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, multiply_0, assign_0, full_4):
    
        # EarlyReturn(0, 35)

        # pd_op.cast: (xi64) <- (xi32)
        cast_5 = paddle._C_ops.cast(multiply_0, paddle.int64)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, assign_0, full_4, cast_5]

    def op_combine_2(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, assign_0, full_4, cast_5):
    
        # EarlyReturn(0, 36)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_2 = [full_4, cast_5, full_4]

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, assign_0, full_4, combine_2]

    def op_stack_1(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, assign_0, full_4, combine_2):
    
        # EarlyReturn(0, 37)

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_2, 0)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, assign_0, full_4, stack_1]

    def op_full_with_tensor_0(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_3, full_int_array_5, reshape_0, assign_0, full_4, stack_1):
    
        # EarlyReturn(0, 38)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(full_3, stack_1, paddle.float32)

        return [data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0]

    def op_shape_1(self, data_1, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0):
    
        # EarlyReturn(0, 39)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_1 = paddle._C_ops.shape(data_1)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, shape_1]

    def op_slice_4(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, shape_1):
    
        # EarlyReturn(0, 40)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_1, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, shape_1]

    def op_slice_5(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, shape_1):
    
        # EarlyReturn(0, 41)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_1, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, shape_1]

    def op_slice_6(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, shape_1):
    
        # EarlyReturn(0, 42)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_1, [0], full_int_array_2, full_int_array_3, [1], [0])

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, shape_1, slice_6]

    def op_slice_7(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, shape_1, slice_6):
    
        # EarlyReturn(0, 43)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7]

    def op_cast_6(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7):
    
        # EarlyReturn(0, 44)

        # pd_op.cast: (xi64) <- (xi32)
        cast_6 = paddle._C_ops.cast(slice_7, paddle.int64)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, cast_6]

    def op_arange_2(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, cast_6):
    
        # EarlyReturn(0, 45)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_2 = paddle.arange(full_0, cast_6, full_1, dtype=paddle.int64)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, arange_2]

    def op_cast_7(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, arange_2):
    
        # EarlyReturn(0, 46)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_7 = paddle._C_ops.cast(arange_2, paddle.float32)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, cast_7]

    def op_scale_4(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, cast_7):
    
        # EarlyReturn(0, 47)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_7, full_2, 0.5, True)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, scale_4]

    def op_full_5(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, scale_4):
    
        # EarlyReturn(0, 48)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], 16, paddle.float32, paddle.core.CPUPlace())

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, scale_4, full_5]

    def op_scale_5(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, scale_4, full_5):
    
        # EarlyReturn(0, 49)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_5, 0, True)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5]

    def op_cast_8(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5):
    
        # EarlyReturn(0, 50)

        # pd_op.cast: (xi64) <- (xi32)
        cast_8 = paddle._C_ops.cast(slice_6, paddle.int64)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, cast_8]

    def op_arange_3(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, cast_8):
    
        # EarlyReturn(0, 51)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_3 = paddle.arange(full_0, cast_8, full_1, dtype=paddle.int64)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, arange_3]

    def op_cast_9(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, arange_3):
    
        # EarlyReturn(0, 52)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_9 = paddle._C_ops.cast(arange_3, paddle.float32)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, cast_9]

    def op_scale_6(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, cast_9):
    
        # EarlyReturn(0, 53)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_9, full_2, 0.5, True)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, scale_6]

    def op_scale_7(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, scale_6):
    
        # EarlyReturn(0, 54)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_5, 0, True)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, scale_7]

    def op_combine_3(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, scale_5, scale_7):
    
        # EarlyReturn(0, 55)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_3 = [scale_7, scale_5]

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, combine_3]

    def op_meshgrid_1(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, combine_3):
    
        # EarlyReturn(0, 56)

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, meshgrid_1]

    def op_split_1(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, meshgrid_1):
    
        # EarlyReturn(0, 57)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_2, split_3, = meshgrid_1

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, split_2, split_3]

    def op_combine_4(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, split_2, split_3):
    
        # EarlyReturn(0, 58)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_4 = [split_3, split_2]

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, combine_4]

    def op_stack_2(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, combine_4):
    
        # EarlyReturn(0, 59)

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, stack_2]

    def op_cast_10(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, stack_2):
    
        # EarlyReturn(0, 60)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_10 = paddle._C_ops.cast(stack_2, paddle.float32)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, cast_10]

    def op_reshape_1(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, cast_10):
    
        # EarlyReturn(0, 61)

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_2, reshape_3 = paddle.reshape(cast_10, full_int_array_5), None

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, reshape_2]

    def op_multiply_1(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, slice_6, slice_7, full_5, reshape_2):
    
        # EarlyReturn(0, 62)

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_1 = slice_6 * slice_7

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, multiply_1]

    def op_assign_1(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, multiply_1):
    
        # EarlyReturn(0, 63)

        # pd_op.assign: (xi32) <- (xi32)
        assign_1 = multiply_1

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, multiply_1, assign_1]

    def op_cast_11(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, multiply_1, assign_1):
    
        # EarlyReturn(0, 64)

        # pd_op.cast: (xi64) <- (xi32)
        cast_11 = paddle._C_ops.cast(multiply_1, paddle.int64)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, assign_1, cast_11]

    def op_combine_5(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, assign_1, cast_11):
    
        # EarlyReturn(0, 65)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_5 = [full_4, cast_11, full_4]

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, assign_1, combine_5]

    def op_stack_3(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, assign_1, combine_5):
    
        # EarlyReturn(0, 66)

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_5, 0)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, assign_1, stack_3]

    def op_full_with_tensor_1(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, full_5, reshape_2, assign_1, stack_3):
    
        # EarlyReturn(0, 67)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_1 = paddle._C_ops.full_with_tensor(full_5, stack_3, paddle.float32)

        return [data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1]

    def op_shape_2(self, data_2, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1):
    
        # EarlyReturn(0, 68)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_2 = paddle._C_ops.shape(data_2)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, shape_2]

    def op_slice_8(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, shape_2):
    
        # EarlyReturn(0, 69)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_2, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, shape_2]

    def op_slice_9(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, shape_2):
    
        # EarlyReturn(0, 70)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_2, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, shape_2]

    def op_slice_10(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, shape_2):
    
        # EarlyReturn(0, 71)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(shape_2, [0], full_int_array_2, full_int_array_3, [1], [0])

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, shape_2, slice_10]

    def op_slice_11(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, shape_2, slice_10):
    
        # EarlyReturn(0, 72)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_2, [0], full_int_array_3, full_int_array_4, [1], [0])

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11]

    def op_cast_12(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11):
    
        # EarlyReturn(0, 73)

        # pd_op.cast: (xi64) <- (xi32)
        cast_12 = paddle._C_ops.cast(slice_11, paddle.int64)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, cast_12]

    def op_arange_4(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, cast_12):
    
        # EarlyReturn(0, 74)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_4 = paddle.arange(full_0, cast_12, full_1, dtype=paddle.int64)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, arange_4]

    def op_cast_13(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, arange_4):
    
        # EarlyReturn(0, 75)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_13 = paddle._C_ops.cast(arange_4, paddle.float32)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, cast_13]

    def op_scale_8(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, cast_13):
    
        # EarlyReturn(0, 76)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_13, full_2, 0.5, True)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, scale_8]

    def op_full_6(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, scale_8):
    
        # EarlyReturn(0, 77)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], 32, paddle.float32, paddle.core.CPUPlace())

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, scale_8, full_6]

    def op_scale_9(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, scale_8, full_6):
    
        # EarlyReturn(0, 78)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_6, 0, True)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9]

    def op_cast_14(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9):
    
        # EarlyReturn(0, 79)

        # pd_op.cast: (xi64) <- (xi32)
        cast_14 = paddle._C_ops.cast(slice_10, paddle.int64)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, cast_14]

    def op_arange_5(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, cast_14):
    
        # EarlyReturn(0, 80)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_5 = paddle.arange(full_0, cast_14, full_1, dtype=paddle.int64)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, arange_5]

    def op_cast_15(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, arange_5):
    
        # EarlyReturn(0, 81)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_15 = paddle._C_ops.cast(arange_5, paddle.float32)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, cast_15]

    def op_scale_10(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, cast_15):
    
        # EarlyReturn(0, 82)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(cast_15, full_2, 0.5, True)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, scale_10]

    def op_scale_11(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, scale_10):
    
        # EarlyReturn(0, 83)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(scale_10, full_6, 0, True)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, scale_11]

    def op_combine_6(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, scale_9, scale_11):
    
        # EarlyReturn(0, 84)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_6 = [scale_11, scale_9]

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, combine_6]

    def op_meshgrid_2(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, combine_6):
    
        # EarlyReturn(0, 85)

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, meshgrid_2]

    def op_split_2(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, meshgrid_2):
    
        # EarlyReturn(0, 86)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_4, split_5, = meshgrid_2

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, split_4, split_5]

    def op_combine_7(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, split_4, split_5):
    
        # EarlyReturn(0, 87)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_7 = [split_5, split_4]

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, combine_7]

    def op_stack_4(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, combine_7):
    
        # EarlyReturn(0, 88)

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_4 = paddle._C_ops.stack(combine_7, -1)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, stack_4]

    def op_cast_16(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, stack_4):
    
        # EarlyReturn(0, 89)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_16 = paddle._C_ops.cast(stack_4, paddle.float32)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, cast_16]

    def op_reshape_2(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, cast_16):
    
        # EarlyReturn(0, 90)

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_4, reshape_5 = paddle.reshape(cast_16, full_int_array_5), None

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, reshape_4]

    def op_multiply_2(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, slice_10, slice_11, full_6, reshape_4):
    
        # EarlyReturn(0, 91)

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_2 = slice_10 * slice_11

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, multiply_2]

    def op_assign_2(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, multiply_2):
    
        # EarlyReturn(0, 92)

        # pd_op.assign: (xi32) <- (xi32)
        assign_2 = multiply_2

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, multiply_2, assign_2]

    def op_cast_17(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, multiply_2, assign_2):
    
        # EarlyReturn(0, 93)

        # pd_op.cast: (xi64) <- (xi32)
        cast_17 = paddle._C_ops.cast(multiply_2, paddle.int64)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, assign_2, cast_17]

    def op_combine_8(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, assign_2, cast_17):
    
        # EarlyReturn(0, 94)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_8 = [full_4, cast_17, full_4]

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, assign_2, combine_8]

    def op_stack_5(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, assign_2, combine_8):
    
        # EarlyReturn(0, 95)

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_8, 0)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, assign_2, stack_5]

    def op_full_with_tensor_2(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, full_6, reshape_4, assign_2, stack_5):
    
        # EarlyReturn(0, 96)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_2 = paddle._C_ops.full_with_tensor(full_6, stack_5, paddle.float32)

        return [data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2]

    def op_shape_3(self, data_3, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2):
    
        # EarlyReturn(0, 97)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_3 = paddle._C_ops.shape(data_3)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, shape_3]

    def op_slice_12(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, shape_3):
    
        # EarlyReturn(0, 98)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_3, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, shape_3]

    def op_slice_13(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, shape_3):
    
        # EarlyReturn(0, 99)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_3, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, shape_3]

    def op_slice_14(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, shape_3):
    
        # EarlyReturn(0, 100)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_3, [0], full_int_array_2, full_int_array_3, [1], [0])

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, shape_3, slice_14]

    def op_slice_15(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, shape_3, slice_14):
    
        # EarlyReturn(0, 101)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_3, [0], full_int_array_3, full_int_array_4, [1], [0])

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15]

    def op_cast_18(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15):
    
        # EarlyReturn(0, 102)

        # pd_op.cast: (xi64) <- (xi32)
        cast_18 = paddle._C_ops.cast(slice_15, paddle.int64)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, cast_18]

    def op_arange_6(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, cast_18):
    
        # EarlyReturn(0, 103)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_6 = paddle.arange(full_0, cast_18, full_1, dtype=paddle.int64)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, arange_6]

    def op_cast_19(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, arange_6):
    
        # EarlyReturn(0, 104)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_19 = paddle._C_ops.cast(arange_6, paddle.float32)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, cast_19]

    def op_scale_12(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, cast_19):
    
        # EarlyReturn(0, 105)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(cast_19, full_2, 0.5, True)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, scale_12]

    def op_full_7(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, scale_12):
    
        # EarlyReturn(0, 106)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], 64, paddle.float32, paddle.core.CPUPlace())

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, scale_12, full_7]

    def op_scale_13(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, scale_12, full_7):
    
        # EarlyReturn(0, 107)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(scale_12, full_7, 0, True)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13]

    def op_cast_20(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13):
    
        # EarlyReturn(0, 108)

        # pd_op.cast: (xi64) <- (xi32)
        cast_20 = paddle._C_ops.cast(slice_14, paddle.int64)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, cast_20]

    def op_arange_7(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, cast_20):
    
        # EarlyReturn(0, 109)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_7 = paddle.arange(full_0, cast_20, full_1, dtype=paddle.int64)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, arange_7]

    def op_cast_21(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, arange_7):
    
        # EarlyReturn(0, 110)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_21 = paddle._C_ops.cast(arange_7, paddle.float32)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, cast_21]

    def op_scale_14(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, cast_21):
    
        # EarlyReturn(0, 111)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(cast_21, full_2, 0.5, True)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, scale_14]

    def op_scale_15(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, scale_14):
    
        # EarlyReturn(0, 112)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(scale_14, full_7, 0, True)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, scale_15]

    def op_combine_9(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, scale_13, scale_15):
    
        # EarlyReturn(0, 113)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_9 = [scale_15, scale_13]

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, combine_9]

    def op_meshgrid_3(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, combine_9):
    
        # EarlyReturn(0, 114)

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_9)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, meshgrid_3]

    def op_split_3(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, meshgrid_3):
    
        # EarlyReturn(0, 115)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_6, split_7, = meshgrid_3

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, split_6, split_7]

    def op_combine_10(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, split_6, split_7):
    
        # EarlyReturn(0, 116)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_10 = [split_7, split_6]

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, combine_10]

    def op_stack_6(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, combine_10):
    
        # EarlyReturn(0, 117)

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_6 = paddle._C_ops.stack(combine_10, -1)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, stack_6]

    def op_cast_22(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, stack_6):
    
        # EarlyReturn(0, 118)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_22 = paddle._C_ops.cast(stack_6, paddle.float32)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, cast_22]

    def op_reshape_3(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, cast_22):
    
        # EarlyReturn(0, 119)

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_6, reshape_7 = paddle.reshape(cast_22, full_int_array_5), None

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, reshape_6]

    def op_multiply_3(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, slice_14, slice_15, full_7, reshape_6):
    
        # EarlyReturn(0, 120)

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_3 = slice_14 * slice_15

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, multiply_3]

    def op_assign_3(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, multiply_3):
    
        # EarlyReturn(0, 121)

        # pd_op.assign: (xi32) <- (xi32)
        assign_3 = multiply_3

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, multiply_3, assign_3]

    def op_cast_23(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, multiply_3, assign_3):
    
        # EarlyReturn(0, 122)

        # pd_op.cast: (xi64) <- (xi32)
        cast_23 = paddle._C_ops.cast(multiply_3, paddle.int64)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, assign_3, cast_23]

    def op_combine_11(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, assign_3, cast_23):
    
        # EarlyReturn(0, 123)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_11 = [full_4, cast_23, full_4]

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, assign_3, combine_11]

    def op_stack_7(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, assign_3, combine_11):
    
        # EarlyReturn(0, 124)

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_11, 0)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, assign_3, stack_7]

    def op_full_with_tensor_3(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, full_7, reshape_6, assign_3, stack_7):
    
        # EarlyReturn(0, 125)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_3 = paddle._C_ops.full_with_tensor(full_7, stack_7, paddle.float32)

        return [data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3]

    def op_shape_4(self, data_4, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3):
    
        # EarlyReturn(0, 126)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_4 = paddle._C_ops.shape(data_4)

        return [full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, shape_4]

    def op_slice_16(self, full_int_array_0, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, shape_4):
    
        # EarlyReturn(0, 127)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_4, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, shape_4]

    def op_slice_17(self, full_int_array_1, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, shape_4):
    
        # EarlyReturn(0, 128)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_4, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, shape_4]

    def op_slice_18(self, full_int_array_2, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, shape_4):
    
        # EarlyReturn(0, 129)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(shape_4, [0], full_int_array_2, full_int_array_3, [1], [0])

        return [full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, shape_4, slice_18]

    def op_slice_19(self, full_int_array_3, full_int_array_4, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, shape_4, slice_18):
    
        # EarlyReturn(0, 130)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(shape_4, [0], full_int_array_3, full_int_array_4, [1], [0])

        return [full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19]

    def op_cast_24(self, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19):
    
        # EarlyReturn(0, 131)

        # pd_op.cast: (xi64) <- (xi32)
        cast_24 = paddle._C_ops.cast(slice_19, paddle.int64)

        return [full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, cast_24]

    def op_arange_8(self, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, cast_24):
    
        # EarlyReturn(0, 132)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_8 = paddle.arange(full_0, cast_24, full_1, dtype=paddle.int64)

        return [full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, arange_8]

    def op_cast_25(self, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, arange_8):
    
        # EarlyReturn(0, 133)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_25 = paddle._C_ops.cast(arange_8, paddle.float32)

        return [full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, cast_25]

    def op_scale_16(self, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, cast_25):
    
        # EarlyReturn(0, 134)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(cast_25, full_2, 0.5, True)

        return [full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, scale_16]

    def op_full_8(self, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, scale_16):
    
        # EarlyReturn(0, 135)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], 128, paddle.float32, paddle.core.CPUPlace())

        return [full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, scale_16, full_8]

    def op_scale_17(self, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, scale_16, full_8):
    
        # EarlyReturn(0, 136)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(scale_16, full_8, 0, True)

        return [full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17]

    def op_cast_26(self, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17):
    
        # EarlyReturn(0, 137)

        # pd_op.cast: (xi64) <- (xi32)
        cast_26 = paddle._C_ops.cast(slice_18, paddle.int64)

        return [full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, cast_26]

    def op_arange_9(self, full_0, full_1, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, cast_26):
    
        # EarlyReturn(0, 138)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_9 = paddle.arange(full_0, cast_26, full_1, dtype=paddle.int64)

        return [full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, arange_9]

    def op_cast_27(self, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, arange_9):
    
        # EarlyReturn(0, 139)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_27 = paddle._C_ops.cast(arange_9, paddle.float32)

        return [full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, cast_27]

    def op_scale_18(self, full_2, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, cast_27):
    
        # EarlyReturn(0, 140)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_27, full_2, 0.5, True)

        return [full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, scale_18]

    def op_scale_19(self, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, scale_18):
    
        # EarlyReturn(0, 141)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_8, 0, True)

        return [full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, scale_19]

    def op_combine_12(self, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, scale_17, scale_19):
    
        # EarlyReturn(0, 142)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_12 = [scale_19, scale_17]

        return [full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, combine_12]

    def op_meshgrid_4(self, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, combine_12):
    
        # EarlyReturn(0, 143)

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_12)

        return [full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, meshgrid_4]

    def op_split_4(self, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, meshgrid_4):
    
        # EarlyReturn(0, 144)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        split_8, split_9, = meshgrid_4

        return [full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, split_8, split_9]

    def op_combine_13(self, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, split_8, split_9):
    
        # EarlyReturn(0, 145)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_13 = [split_9, split_8]

        return [full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, combine_13]

    def op_stack_8(self, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, combine_13):
    
        # EarlyReturn(0, 146)

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_8 = paddle._C_ops.stack(combine_13, -1)

        return [full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, stack_8]

    def op_cast_28(self, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, stack_8):
    
        # EarlyReturn(0, 147)

        # pd_op.cast: (-1x-1x2xf32) <- (-1x-1x2xf32)
        cast_28 = paddle._C_ops.cast(stack_8, paddle.float32)

        return [full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, cast_28]

    def op_reshape_4(self, full_int_array_5, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, cast_28):
    
        # EarlyReturn(0, 148)

        # pd_op.reshape: (1x-1x2xf32, 0x-1x-1x2xi64) <- (-1x-1x2xf32, 3xi64)
        reshape_8, reshape_9 = paddle.reshape(cast_28, full_int_array_5), None

        return [reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, reshape_8]

    def op_multiply_4(self, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, slice_18, slice_19, full_8, reshape_8):
    
        # EarlyReturn(0, 149)

        # pd_op.multiply: (xi32) <- (xi32, xi32)
        multiply_4 = slice_18 * slice_19

        return [reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, multiply_4]

    def op_assign_4(self, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, multiply_4):
    
        # EarlyReturn(0, 150)

        # pd_op.assign: (xi32) <- (xi32)
        assign_4 = multiply_4

        return [reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, multiply_4, assign_4]

    def op_cast_29(self, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, multiply_4, assign_4):
    
        # EarlyReturn(0, 151)

        # pd_op.cast: (xi64) <- (xi32)
        cast_29 = paddle._C_ops.cast(multiply_4, paddle.int64)

        return [reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, assign_4, cast_29]

    def op_combine_14(self, reshape_0, assign_0, full_4, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, assign_4, cast_29):
    
        # EarlyReturn(0, 152)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_14 = [full_4, cast_29, full_4]

        return [reshape_0, assign_0, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, assign_4, combine_14]

    def op_stack_9(self, reshape_0, assign_0, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, assign_4, combine_14):
    
        # EarlyReturn(0, 153)

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_14, 0)

        return [reshape_0, assign_0, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, assign_4, stack_9]

    def op_full_with_tensor_4(self, reshape_0, assign_0, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, full_8, reshape_8, assign_4, stack_9):
    
        # EarlyReturn(0, 154)

        # pd_op.full_with_tensor: (1x-1x1xf32) <- (1xf32, 3xi64)
        full_with_tensor_4 = paddle._C_ops.full_with_tensor(full_8, stack_9, paddle.float32)

        return [reshape_0, assign_0, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, reshape_8, assign_4, full_with_tensor_4]

    def op_full_9(self, reshape_0, assign_0, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, reshape_8, assign_4, full_with_tensor_4):
    
        # EarlyReturn(0, 155)

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        return [reshape_0, assign_0, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, reshape_8, assign_4, full_with_tensor_4, full_9]

    def op_combine_15(self, reshape_0, assign_0, full_with_tensor_0, reshape_2, assign_1, full_with_tensor_1, reshape_4, assign_2, full_with_tensor_2, reshape_6, assign_3, full_with_tensor_3, reshape_8, assign_4, full_with_tensor_4, full_9):
    
        # EarlyReturn(0, 156)

        # builtin.combine: ([1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32]) <- (1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32)
        combine_15 = [reshape_0, reshape_2, reshape_4, reshape_6, reshape_8]

        return [assign_0, full_with_tensor_0, assign_1, full_with_tensor_1, assign_2, full_with_tensor_2, assign_3, full_with_tensor_3, assign_4, full_with_tensor_4, full_9, combine_15]

    def op_concat_0(self, assign_0, full_with_tensor_0, assign_1, full_with_tensor_1, assign_2, full_with_tensor_2, assign_3, full_with_tensor_3, assign_4, full_with_tensor_4, full_9, combine_15):
    
        # EarlyReturn(0, 157)

        # pd_op.concat: (1x-1x2xf32) <- ([1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_15, full_9)

        return [assign_0, full_with_tensor_0, assign_1, full_with_tensor_1, assign_2, full_with_tensor_2, assign_3, full_with_tensor_3, assign_4, full_with_tensor_4, full_9, concat_0]

    def op_combine_16(self, assign_0, full_with_tensor_0, assign_1, full_with_tensor_1, assign_2, full_with_tensor_2, assign_3, full_with_tensor_3, assign_4, full_with_tensor_4, full_9, concat_0):
    
        # EarlyReturn(0, 158)

        # builtin.combine: ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32]) <- (1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32)
        combine_16 = [full_with_tensor_0, full_with_tensor_1, full_with_tensor_2, full_with_tensor_3, full_with_tensor_4]

        return [assign_0, assign_1, assign_2, assign_3, assign_4, full_9, concat_0, combine_16]

    def op_concat_1(self, assign_0, assign_1, assign_2, assign_3, assign_4, full_9, concat_0, combine_16):
    
        # EarlyReturn(0, 159)

        # pd_op.concat: (1x-1x1xf32) <- ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_16, full_9)

        return [concat_0, concat_1, assign_0, assign_1, assign_2, assign_3, assign_4]

if True and not (IsCinnStageEnableDiff() and LastCINNStageFailed()):

    class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
        def prepare_data(self):
            self.inputs = [
                # data_0
                paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
                # data_1
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
                # data_2
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
                # data_3
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
                # data_4
                paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            ]
            for input in self.inputs:
                input.stop_gradient = True

        def apply_to_static(self, net, use_cinn):
            build_strategy = paddle.static.BuildStrategy()
            input_spec = [
                # data_0
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_1
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_2
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_3
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_4
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            build_strategy.build_cinn_pass = use_cinn
            return paddle.jit.to_static(
                net,
                input_spec=input_spec,
                build_strategy=build_strategy,
                full_graph=True,
            )

        def train(self, use_cinn):
            net = Block_builtin_module_0_0_0()
            if GetEnvVarEnableJit():
                net = self.apply_to_static(net, use_cinn)
            paddle.seed(2024)
            out = net(*self.inputs)
            return out

if __name__ == '__main__':
    unittest.main()