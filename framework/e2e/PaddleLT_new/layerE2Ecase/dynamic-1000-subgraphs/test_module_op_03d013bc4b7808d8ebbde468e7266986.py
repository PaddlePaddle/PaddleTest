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
    return [98][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8):

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32)
        combine_0 = [data_0, data_1, data_2]

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32)
        combine_1 = [data_3, data_4, data_5]

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, assign_1)

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32)
        combine_2 = [data_6, data_7, data_8]

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, assign_0)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 80, paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (80xi64) <- (1xf32, 1xf32, 1xf32)
        arange_0 = paddle.arange(full_1, full_2, full_3, dtype=paddle.int64)

        # pd_op.cast: (80xf32) <- (80xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_3, 0, True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 8, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, 0, True)

        # builtin.combine: ([80xf32, 80xf32]) <- (80xf32, 80xf32)
        combine_3 = [scale_1, scale_1]

        # pd_op.meshgrid: ([80x80xf32, 80x80xf32]) <- ([80xf32, 80xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (80x80xf32, 80x80xf32) <- ([80x80xf32, 80x80xf32])
        split_0, split_1, = meshgrid_0

        # builtin.combine: ([80x80xf32, 80x80xf32]) <- (80x80xf32, 80x80xf32)
        combine_4 = [split_1, split_0]

        # pd_op.stack: (80x80x2xf32) <- ([80x80xf32, 80x80xf32])
        stack_0 = paddle._C_ops.stack(combine_4, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 2]

        # pd_op.reshape: (6400x2xf32, 0x80x80x2xi64) <- (80x80x2xf32, 2xi64)
        reshape_0, reshape_1 = paddle.reshape(stack_0, full_int_array_0), None

        # pd_op.full: (6400x1xf32) <- ()
        full_5 = paddle._C_ops.full([6400, 1], 8, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], 40, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (40xi64) <- (1xf32, 1xf32, 1xf32)
        arange_1 = paddle.arange(full_1, full_6, full_3, dtype=paddle.int64)

        # pd_op.cast: (40xf32) <- (40xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_1, full_3, 0, True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], 16, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_7, 0, True)

        # builtin.combine: ([40xf32, 40xf32]) <- (40xf32, 40xf32)
        combine_5 = [scale_3, scale_3]

        # pd_op.meshgrid: ([40x40xf32, 40x40xf32]) <- ([40xf32, 40xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_5)

        # builtin.split: (40x40xf32, 40x40xf32) <- ([40x40xf32, 40x40xf32])
        split_2, split_3, = meshgrid_1

        # builtin.combine: ([40x40xf32, 40x40xf32]) <- (40x40xf32, 40x40xf32)
        combine_6 = [split_3, split_2]

        # pd_op.stack: (40x40x2xf32) <- ([40x40xf32, 40x40xf32])
        stack_1 = paddle._C_ops.stack(combine_6, -1)

        # pd_op.reshape: (1600x2xf32, 0x40x40x2xi64) <- (40x40x2xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(stack_1, full_int_array_0), None

        # pd_op.full: (1600x1xf32) <- ()
        full_8 = paddle._C_ops.full([1600, 1], 16, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], 20, paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (20xi64) <- (1xf32, 1xf32, 1xf32)
        arange_2 = paddle.arange(full_1, full_9, full_3, dtype=paddle.int64)

        # pd_op.cast: (20xf32) <- (20xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_2, full_3, 0, True)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], 32, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_10, 0, True)

        # builtin.combine: ([20xf32, 20xf32]) <- (20xf32, 20xf32)
        combine_7 = [scale_5, scale_5]

        # pd_op.meshgrid: ([20x20xf32, 20x20xf32]) <- ([20xf32, 20xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_7)

        # builtin.split: (20x20xf32, 20x20xf32) <- ([20x20xf32, 20x20xf32])
        split_4, split_5, = meshgrid_2

        # builtin.combine: ([20x20xf32, 20x20xf32]) <- (20x20xf32, 20x20xf32)
        combine_8 = [split_5, split_4]

        # pd_op.stack: (20x20x2xf32) <- ([20x20xf32, 20x20xf32])
        stack_2 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.reshape: (400x2xf32, 0x20x20x2xi64) <- (20x20x2xf32, 2xi64)
        reshape_4, reshape_5 = paddle.reshape(stack_2, full_int_array_0), None

        # pd_op.full: (400x1xf32) <- ()
        full_11 = paddle._C_ops.full([400, 1], 32, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([6400x2xf32, 1600x2xf32, 400x2xf32]) <- (6400x2xf32, 1600x2xf32, 400x2xf32)
        combine_9 = [reshape_0, reshape_2, reshape_4]

        # pd_op.concat: (8400x2xf32) <- ([6400x2xf32, 1600x2xf32, 400x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_9, full_12)

        # pd_op.cast: (8400x2xf32) <- (8400x2xf32)
        cast_3 = paddle._C_ops.cast(concat_3, paddle.float32)

        # builtin.combine: ([6400x1xf32, 1600x1xf32, 400x1xf32]) <- (6400x1xf32, 1600x1xf32, 400x1xf32)
        combine_10 = [full_5, full_8, full_11]

        # pd_op.concat: (8400x1xf32) <- ([6400x1xf32, 1600x1xf32, 400x1xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_10, full_12)

        # pd_op.assign: (8400x1xf32) <- (8400x1xf32)
        assign_2 = concat_4

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], 2, paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_1, 2, full_13)

        # builtin.split: (-1x-1x-1xf32, -1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32])
        split_6, split_7, = split_with_num_0

        # pd_op.divide: (8400x2xf32) <- (8400x2xf32, 8400x1xf32)
        divide_0 = cast_3 / concat_4

        # pd_op.add: (-1x8400x2xf32) <- (-1x-1x-1xf32, 8400x2xf32)
        add_0 = split_6 + divide_0

        # pd_op.exp: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        exp_0 = paddle._C_ops.exp(split_7)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], 0.5, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1x-1xf32) <- (-1x-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(exp_0, full_14, 0, True)

        # pd_op.subtract: (-1x8400x2xf32) <- (-1x8400x2xf32, -1x-1x-1xf32)
        subtract_0 = add_0 - scale_6

        # pd_op.add: (-1x8400x2xf32) <- (-1x8400x2xf32, -1x-1x-1xf32)
        add_1 = add_0 + scale_6

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], -1, paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x8400x2xf32, -1x8400x2xf32]) <- (-1x8400x2xf32, -1x8400x2xf32)
        combine_11 = [subtract_0, add_1]

        # pd_op.concat: (-1x8400x4xf32) <- ([-1x8400x2xf32, -1x8400x2xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_11, full_15)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(cast_0, full_3, 0.5, True)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(scale_7, full_4, 0, True)

        # builtin.combine: ([80xf32, 80xf32]) <- (80xf32, 80xf32)
        combine_12 = [scale_8, scale_8]

        # pd_op.meshgrid: ([80x80xf32, 80x80xf32]) <- ([80xf32, 80xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_12)

        # builtin.split: (80x80xf32, 80x80xf32) <- ([80x80xf32, 80x80xf32])
        split_8, split_9, = meshgrid_3

        # builtin.combine: ([80x80xf32, 80x80xf32]) <- (80x80xf32, 80x80xf32)
        combine_13 = [split_9, split_8]

        # pd_op.stack: (80x80x2xf32) <- ([80x80xf32, 80x80xf32])
        stack_3 = paddle._C_ops.stack(combine_13, -1)

        # pd_op.reshape: (6400x2xf32, 0x80x80x2xi64) <- (80x80x2xf32, 2xi64)
        reshape_6, reshape_7 = paddle.reshape(stack_3, full_int_array_0), None

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(cast_1, full_3, 0.5, True)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(scale_9, full_7, 0, True)

        # builtin.combine: ([40xf32, 40xf32]) <- (40xf32, 40xf32)
        combine_14 = [scale_10, scale_10]

        # pd_op.meshgrid: ([40x40xf32, 40x40xf32]) <- ([40xf32, 40xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_14)

        # builtin.split: (40x40xf32, 40x40xf32) <- ([40x40xf32, 40x40xf32])
        split_10, split_11, = meshgrid_4

        # builtin.combine: ([40x40xf32, 40x40xf32]) <- (40x40xf32, 40x40xf32)
        combine_15 = [split_11, split_10]

        # pd_op.stack: (40x40x2xf32) <- ([40x40xf32, 40x40xf32])
        stack_4 = paddle._C_ops.stack(combine_15, -1)

        # pd_op.reshape: (1600x2xf32, 0x40x40x2xi64) <- (40x40x2xf32, 2xi64)
        reshape_8, reshape_9 = paddle.reshape(stack_4, full_int_array_0), None

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(cast_2, full_3, 0.5, True)

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(scale_11, full_10, 0, True)

        # builtin.combine: ([20xf32, 20xf32]) <- (20xf32, 20xf32)
        combine_16 = [scale_12, scale_12]

        # pd_op.meshgrid: ([20x20xf32, 20x20xf32]) <- ([20xf32, 20xf32])
        meshgrid_5 = paddle._C_ops.meshgrid(combine_16)

        # builtin.split: (20x20xf32, 20x20xf32) <- ([20x20xf32, 20x20xf32])
        split_12, split_13, = meshgrid_5

        # builtin.combine: ([20x20xf32, 20x20xf32]) <- (20x20xf32, 20x20xf32)
        combine_17 = [split_13, split_12]

        # pd_op.stack: (20x20x2xf32) <- ([20x20xf32, 20x20xf32])
        stack_5 = paddle._C_ops.stack(combine_17, -1)

        # pd_op.reshape: (400x2xf32, 0x20x20x2xi64) <- (20x20x2xf32, 2xi64)
        reshape_10, reshape_11 = paddle.reshape(stack_5, full_int_array_0), None

        # builtin.combine: ([6400x2xf32, 1600x2xf32, 400x2xf32]) <- (6400x2xf32, 1600x2xf32, 400x2xf32)
        combine_18 = [reshape_6, reshape_8, reshape_10]

        # pd_op.concat: (8400x2xf32) <- ([6400x2xf32, 1600x2xf32, 400x2xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_18, full_12)

        # pd_op.cast: (8400x2xf32) <- (8400x2xf32)
        cast_4 = paddle._C_ops.cast(concat_6, paddle.float32)
        return full_0, assign_1, assign_0, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_0, concat_5, concat_2, cast_4, assign_2, reshape_6, reshape_8, reshape_10



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

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8):
        args = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_full_0,
            self.op_assign_0,
            self.op_assign_1,
            self.op_combine_0,
            self.op_concat_0,
            self.op_combine_1,
            self.op_concat_1,
            self.op_combine_2,
            self.op_concat_2,
            self.op_full_1,
            self.op_full_2,
            self.op_full_3,
            self.op_arange_0,
            self.op_cast_0,
            self.op_scale_0,
            self.op_full_4,
            self.op_scale_1,
            self.op_combine_3,
            self.op_meshgrid_0,
            self.op_split_0,
            self.op_combine_4,
            self.op_stack_0,
            self.op_full_int_array_0,
            self.op_reshape_0,
            self.op_full_5,
            self.op_full_6,
            self.op_arange_1,
            self.op_cast_1,
            self.op_scale_2,
            self.op_full_7,
            self.op_scale_3,
            self.op_combine_5,
            self.op_meshgrid_1,
            self.op_split_1,
            self.op_combine_6,
            self.op_stack_1,
            self.op_reshape_1,
            self.op_full_8,
            self.op_full_9,
            self.op_arange_2,
            self.op_cast_2,
            self.op_scale_4,
            self.op_full_10,
            self.op_scale_5,
            self.op_combine_7,
            self.op_meshgrid_2,
            self.op_split_2,
            self.op_combine_8,
            self.op_stack_2,
            self.op_reshape_2,
            self.op_full_11,
            self.op_full_12,
            self.op_combine_9,
            self.op_concat_3,
            self.op_cast_3,
            self.op_combine_10,
            self.op_concat_4,
            self.op_assign_2,
            self.op_full_13,
            self.op_split_with_num_0,
            self.op_split_3,
            self.op_divide_0,
            self.op_add_0,
            self.op_exp_0,
            self.op_full_14,
            self.op_scale_6,
            self.op_subtract_0,
            self.op_add_1,
            self.op_full_15,
            self.op_combine_11,
            self.op_concat_5,
            self.op_scale_7,
            self.op_scale_8,
            self.op_combine_12,
            self.op_meshgrid_3,
            self.op_split_4,
            self.op_combine_13,
            self.op_stack_3,
            self.op_reshape_3,
            self.op_scale_9,
            self.op_scale_10,
            self.op_combine_14,
            self.op_meshgrid_4,
            self.op_split_5,
            self.op_combine_15,
            self.op_stack_4,
            self.op_reshape_4,
            self.op_scale_11,
            self.op_scale_12,
            self.op_combine_16,
            self.op_meshgrid_5,
            self.op_split_6,
            self.op_combine_17,
            self.op_stack_5,
            self.op_reshape_5,
            self.op_combine_18,
            self.op_concat_6,
            self.op_cast_4,
        ]

    def op_full_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8):
    
        # EarlyReturn(0, 0)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        return [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, full_0]

    def op_assign_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, full_0):
    
        # EarlyReturn(0, 1)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        return [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, full_0, assign_0]

    def op_assign_1(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, full_0, assign_0):
    
        # EarlyReturn(0, 2)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        return [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, full_0, assign_0, assign_1]

    def op_combine_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, full_0, assign_0, assign_1):
    
        # EarlyReturn(0, 3)

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32)
        combine_0 = [data_0, data_1, data_2]

        return [data_3, data_4, data_5, data_6, data_7, data_8, full_0, assign_0, assign_1, combine_0]

    def op_concat_0(self, data_3, data_4, data_5, data_6, data_7, data_8, full_0, assign_0, assign_1, combine_0):
    
        # EarlyReturn(0, 4)

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        return [data_3, data_4, data_5, data_6, data_7, data_8, full_0, assign_0, assign_1, concat_0]

    def op_combine_1(self, data_3, data_4, data_5, data_6, data_7, data_8, full_0, assign_0, assign_1, concat_0):
    
        # EarlyReturn(0, 5)

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32)
        combine_1 = [data_3, data_4, data_5]

        return [data_6, data_7, data_8, full_0, assign_0, assign_1, concat_0, combine_1]

    def op_concat_1(self, data_6, data_7, data_8, full_0, assign_0, assign_1, concat_0, combine_1):
    
        # EarlyReturn(0, 6)

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, assign_1)

        return [data_6, data_7, data_8, full_0, assign_0, assign_1, concat_0, concat_1]

    def op_combine_2(self, data_6, data_7, data_8, full_0, assign_0, assign_1, concat_0, concat_1):
    
        # EarlyReturn(0, 7)

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32)
        combine_2 = [data_6, data_7, data_8]

        return [full_0, assign_0, assign_1, concat_0, concat_1, combine_2]

    def op_concat_2(self, full_0, assign_0, assign_1, concat_0, concat_1, combine_2):
    
        # EarlyReturn(0, 8)

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, assign_0)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2]

    def op_full_1(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2):
    
        # EarlyReturn(0, 9)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1]

    def op_full_2(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1):
    
        # EarlyReturn(0, 10)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 80, paddle.float32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_2]

    def op_full_3(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_2):
    
        # EarlyReturn(0, 11)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_2, full_3]

    def op_arange_0(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_2, full_3):
    
        # EarlyReturn(0, 12)

        # pd_op.arange: (80xi64) <- (1xf32, 1xf32, 1xf32)
        arange_0 = paddle.arange(full_1, full_2, full_3, dtype=paddle.int64)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, arange_0]

    def op_cast_0(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, arange_0):
    
        # EarlyReturn(0, 13)

        # pd_op.cast: (80xf32) <- (80xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0]

    def op_scale_0(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0):
    
        # EarlyReturn(0, 14)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_3, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, scale_0]

    def op_full_4(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, scale_0):
    
        # EarlyReturn(0, 15)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 8, paddle.float32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, scale_0, full_4]

    def op_scale_1(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, scale_0, full_4):
    
        # EarlyReturn(0, 16)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_4, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, scale_1]

    def op_combine_3(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, scale_1):
    
        # EarlyReturn(0, 17)

        # builtin.combine: ([80xf32, 80xf32]) <- (80xf32, 80xf32)
        combine_3 = [scale_1, scale_1]

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, combine_3]

    def op_meshgrid_0(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, combine_3):
    
        # EarlyReturn(0, 18)

        # pd_op.meshgrid: ([80x80xf32, 80x80xf32]) <- ([80xf32, 80xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_3)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, meshgrid_0]

    def op_split_0(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, meshgrid_0):
    
        # EarlyReturn(0, 19)

        # builtin.split: (80x80xf32, 80x80xf32) <- ([80x80xf32, 80x80xf32])
        split_0, split_1, = meshgrid_0

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, split_0, split_1]

    def op_combine_4(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, split_0, split_1):
    
        # EarlyReturn(0, 20)

        # builtin.combine: ([80x80xf32, 80x80xf32]) <- (80x80xf32, 80x80xf32)
        combine_4 = [split_1, split_0]

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, combine_4]

    def op_stack_0(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, combine_4):
    
        # EarlyReturn(0, 21)

        # pd_op.stack: (80x80x2xf32) <- ([80x80xf32, 80x80xf32])
        stack_0 = paddle._C_ops.stack(combine_4, -1)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, stack_0]

    def op_full_int_array_0(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, stack_0):
    
        # EarlyReturn(0, 22)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 2]

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, stack_0, full_int_array_0]

    def op_reshape_0(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, stack_0, full_int_array_0):
    
        # EarlyReturn(0, 23)

        # pd_op.reshape: (6400x2xf32, 0x80x80x2xi64) <- (80x80x2xf32, 2xi64)
        reshape_0, reshape_1 = paddle.reshape(stack_0, full_int_array_0), None

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0]

    def op_full_5(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0):
    
        # EarlyReturn(0, 24)

        # pd_op.full: (6400x1xf32) <- ()
        full_5 = paddle._C_ops.full([6400, 1], 8, paddle.float32, paddle.framework._current_expected_place())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5]

    def op_full_6(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5):
    
        # EarlyReturn(0, 25)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], 40, paddle.float32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, full_6]

    def op_arange_1(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, full_6):
    
        # EarlyReturn(0, 26)

        # pd_op.arange: (40xi64) <- (1xf32, 1xf32, 1xf32)
        arange_1 = paddle.arange(full_1, full_6, full_3, dtype=paddle.int64)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, arange_1]

    def op_cast_1(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, arange_1):
    
        # EarlyReturn(0, 27)

        # pd_op.cast: (40xf32) <- (40xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1]

    def op_scale_2(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1):
    
        # EarlyReturn(0, 28)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_1, full_3, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, scale_2]

    def op_full_7(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, scale_2):
    
        # EarlyReturn(0, 29)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], 16, paddle.float32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, scale_2, full_7]

    def op_scale_3(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, scale_2, full_7):
    
        # EarlyReturn(0, 30)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_7, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, scale_3]

    def op_combine_5(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, scale_3):
    
        # EarlyReturn(0, 31)

        # builtin.combine: ([40xf32, 40xf32]) <- (40xf32, 40xf32)
        combine_5 = [scale_3, scale_3]

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, combine_5]

    def op_meshgrid_1(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, combine_5):
    
        # EarlyReturn(0, 32)

        # pd_op.meshgrid: ([40x40xf32, 40x40xf32]) <- ([40xf32, 40xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_5)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, meshgrid_1]

    def op_split_1(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, meshgrid_1):
    
        # EarlyReturn(0, 33)

        # builtin.split: (40x40xf32, 40x40xf32) <- ([40x40xf32, 40x40xf32])
        split_2, split_3, = meshgrid_1

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, split_2, split_3]

    def op_combine_6(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, split_2, split_3):
    
        # EarlyReturn(0, 34)

        # builtin.combine: ([40x40xf32, 40x40xf32]) <- (40x40xf32, 40x40xf32)
        combine_6 = [split_3, split_2]

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, combine_6]

    def op_stack_1(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, combine_6):
    
        # EarlyReturn(0, 35)

        # pd_op.stack: (40x40x2xf32) <- ([40x40xf32, 40x40xf32])
        stack_1 = paddle._C_ops.stack(combine_6, -1)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, stack_1]

    def op_reshape_1(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, stack_1):
    
        # EarlyReturn(0, 36)

        # pd_op.reshape: (1600x2xf32, 0x40x40x2xi64) <- (40x40x2xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(stack_1, full_int_array_0), None

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2]

    def op_full_8(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2):
    
        # EarlyReturn(0, 37)

        # pd_op.full: (1600x1xf32) <- ()
        full_8 = paddle._C_ops.full([1600, 1], 16, paddle.float32, paddle.framework._current_expected_place())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8]

    def op_full_9(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8):
    
        # EarlyReturn(0, 38)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], 20, paddle.float32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, full_9]

    def op_arange_2(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_1, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, full_9):
    
        # EarlyReturn(0, 39)

        # pd_op.arange: (20xi64) <- (1xf32, 1xf32, 1xf32)
        arange_2 = paddle.arange(full_1, full_9, full_3, dtype=paddle.int64)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, arange_2]

    def op_cast_2(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, arange_2):
    
        # EarlyReturn(0, 40)

        # pd_op.cast: (20xf32) <- (20xi64)
        cast_2 = paddle._C_ops.cast(arange_2, paddle.float32)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2]

    def op_scale_4(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2):
    
        # EarlyReturn(0, 41)

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_2, full_3, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, scale_4]

    def op_full_10(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, scale_4):
    
        # EarlyReturn(0, 42)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], 32, paddle.float32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, scale_4, full_10]

    def op_scale_5(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, scale_4, full_10):
    
        # EarlyReturn(0, 43)

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_10, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, scale_5]

    def op_combine_7(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, scale_5):
    
        # EarlyReturn(0, 44)

        # builtin.combine: ([20xf32, 20xf32]) <- (20xf32, 20xf32)
        combine_7 = [scale_5, scale_5]

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, combine_7]

    def op_meshgrid_2(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, combine_7):
    
        # EarlyReturn(0, 45)

        # pd_op.meshgrid: ([20x20xf32, 20x20xf32]) <- ([20xf32, 20xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_7)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, meshgrid_2]

    def op_split_2(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, meshgrid_2):
    
        # EarlyReturn(0, 46)

        # builtin.split: (20x20xf32, 20x20xf32) <- ([20x20xf32, 20x20xf32])
        split_4, split_5, = meshgrid_2

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, split_4, split_5]

    def op_combine_8(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, split_4, split_5):
    
        # EarlyReturn(0, 47)

        # builtin.combine: ([20x20xf32, 20x20xf32]) <- (20x20xf32, 20x20xf32)
        combine_8 = [split_5, split_4]

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, combine_8]

    def op_stack_2(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, combine_8):
    
        # EarlyReturn(0, 48)

        # pd_op.stack: (20x20x2xf32) <- ([20x20xf32, 20x20xf32])
        stack_2 = paddle._C_ops.stack(combine_8, -1)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, stack_2]

    def op_reshape_2(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, stack_2):
    
        # EarlyReturn(0, 49)

        # pd_op.reshape: (400x2xf32, 0x20x20x2xi64) <- (20x20x2xf32, 2xi64)
        reshape_4, reshape_5 = paddle.reshape(stack_2, full_int_array_0), None

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, reshape_4]

    def op_full_11(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, reshape_4):
    
        # EarlyReturn(0, 50)

        # pd_op.full: (400x1xf32) <- ()
        full_11 = paddle._C_ops.full([400, 1], 32, paddle.float32, paddle.framework._current_expected_place())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, reshape_4, full_11]

    def op_full_12(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, reshape_4, full_11):
    
        # EarlyReturn(0, 51)

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, reshape_4, full_11, full_12]

    def op_combine_9(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, reshape_0, full_5, cast_1, full_7, reshape_2, full_8, cast_2, full_10, reshape_4, full_11, full_12):
    
        # EarlyReturn(0, 52)

        # builtin.combine: ([6400x2xf32, 1600x2xf32, 400x2xf32]) <- (6400x2xf32, 1600x2xf32, 400x2xf32)
        combine_9 = [reshape_0, reshape_2, reshape_4]

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, full_5, cast_1, full_7, full_8, cast_2, full_10, full_11, full_12, combine_9]

    def op_concat_3(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, full_5, cast_1, full_7, full_8, cast_2, full_10, full_11, full_12, combine_9):
    
        # EarlyReturn(0, 53)

        # pd_op.concat: (8400x2xf32) <- ([6400x2xf32, 1600x2xf32, 400x2xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_9, full_12)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, full_5, cast_1, full_7, full_8, cast_2, full_10, full_11, full_12, concat_3]

    def op_cast_3(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, full_5, cast_1, full_7, full_8, cast_2, full_10, full_11, full_12, concat_3):
    
        # EarlyReturn(0, 54)

        # pd_op.cast: (8400x2xf32) <- (8400x2xf32)
        cast_3 = paddle._C_ops.cast(concat_3, paddle.float32)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, full_5, cast_1, full_7, full_8, cast_2, full_10, full_11, full_12, cast_3]

    def op_combine_10(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, full_5, cast_1, full_7, full_8, cast_2, full_10, full_11, full_12, cast_3):
    
        # EarlyReturn(0, 55)

        # builtin.combine: ([6400x1xf32, 1600x1xf32, 400x1xf32]) <- (6400x1xf32, 1600x1xf32, 400x1xf32)
        combine_10 = [full_5, full_8, full_11]

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, combine_10]

    def op_concat_4(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, combine_10):
    
        # EarlyReturn(0, 56)

        # pd_op.concat: (8400x1xf32) <- ([6400x1xf32, 1600x1xf32, 400x1xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_10, full_12)

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4]

    def op_assign_2(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4):
    
        # EarlyReturn(0, 57)

        # pd_op.assign: (8400x1xf32) <- (8400x1xf32)
        assign_2 = concat_4

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4, assign_2]

    def op_full_13(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4, assign_2):
    
        # EarlyReturn(0, 58)

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], 2, paddle.int32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4, assign_2, full_13]

    def op_split_with_num_0(self, full_0, assign_0, assign_1, concat_0, concat_1, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4, assign_2, full_13):
    
        # EarlyReturn(0, 59)

        # pd_op.split_with_num: ([-1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_1, 2, full_13)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4, assign_2, full_13, split_with_num_0]

    def op_split_3(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4, assign_2, full_13, split_with_num_0):
    
        # EarlyReturn(0, 60)

        # builtin.split: (-1x-1x-1xf32, -1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32])
        split_6, split_7, = split_with_num_0

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4, assign_2, full_13, split_6, split_7]

    def op_divide_0(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, cast_3, concat_4, assign_2, full_13, split_6, split_7):
    
        # EarlyReturn(0, 61)

        # pd_op.divide: (8400x2xf32) <- (8400x2xf32, 8400x1xf32)
        divide_0 = cast_3 / concat_4

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, split_7, divide_0]

    def op_add_0(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, split_7, divide_0):
    
        # EarlyReturn(0, 62)

        # pd_op.add: (-1x8400x2xf32) <- (-1x-1x-1xf32, 8400x2xf32)
        add_0 = split_6 + divide_0

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, split_7, divide_0, add_0]

    def op_exp_0(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, split_7, divide_0, add_0):
    
        # EarlyReturn(0, 63)

        # pd_op.exp: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        exp_0 = paddle._C_ops.exp(split_7)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0]

    def op_full_14(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0):
    
        # EarlyReturn(0, 64)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], 0.5, paddle.float32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14]

    def op_scale_6(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14):
    
        # EarlyReturn(0, 65)

        # pd_op.scale: (-1x-1x-1xf32) <- (-1x-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(exp_0, full_14, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6]

    def op_subtract_0(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6):
    
        # EarlyReturn(0, 66)

        # pd_op.subtract: (-1x8400x2xf32) <- (-1x8400x2xf32, -1x-1x-1xf32)
        subtract_0 = add_0 - scale_6

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0]

    def op_add_1(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0):
    
        # EarlyReturn(0, 67)

        # pd_op.add: (-1x8400x2xf32) <- (-1x8400x2xf32, -1x-1x-1xf32)
        add_1 = add_0 + scale_6

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1]

    def op_full_15(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1):
    
        # EarlyReturn(0, 68)

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], -1, paddle.int32, paddle.core.CPUPlace())

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15]

    def op_combine_11(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15):
    
        # EarlyReturn(0, 69)

        # builtin.combine: ([-1x8400x2xf32, -1x8400x2xf32]) <- (-1x8400x2xf32, -1x8400x2xf32)
        combine_11 = [subtract_0, add_1]

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, combine_11]

    def op_concat_5(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, combine_11):
    
        # EarlyReturn(0, 70)

        # pd_op.concat: (-1x8400x4xf32) <- ([-1x8400x2xf32, -1x8400x2xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_11, full_15)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5]

    def op_scale_7(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, cast_0, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5):
    
        # EarlyReturn(0, 71)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(cast_0, full_3, 0.5, True)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, scale_7]

    def op_scale_8(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_4, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, scale_7):
    
        # EarlyReturn(0, 72)

        # pd_op.scale: (80xf32) <- (80xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(scale_7, full_4, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, scale_8]

    def op_combine_12(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, scale_8):
    
        # EarlyReturn(0, 73)

        # builtin.combine: ([80xf32, 80xf32]) <- (80xf32, 80xf32)
        combine_12 = [scale_8, scale_8]

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, combine_12]

    def op_meshgrid_3(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, combine_12):
    
        # EarlyReturn(0, 74)

        # pd_op.meshgrid: ([80x80xf32, 80x80xf32]) <- ([80xf32, 80xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_12)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, meshgrid_3]

    def op_split_4(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, meshgrid_3):
    
        # EarlyReturn(0, 75)

        # builtin.split: (80x80xf32, 80x80xf32) <- ([80x80xf32, 80x80xf32])
        split_8, split_9, = meshgrid_3

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, split_8, split_9]

    def op_combine_13(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, split_8, split_9):
    
        # EarlyReturn(0, 76)

        # builtin.combine: ([80x80xf32, 80x80xf32]) <- (80x80xf32, 80x80xf32)
        combine_13 = [split_9, split_8]

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, combine_13]

    def op_stack_3(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, combine_13):
    
        # EarlyReturn(0, 77)

        # pd_op.stack: (80x80x2xf32) <- ([80x80xf32, 80x80xf32])
        stack_3 = paddle._C_ops.stack(combine_13, -1)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, stack_3]

    def op_reshape_3(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, stack_3):
    
        # EarlyReturn(0, 78)

        # pd_op.reshape: (6400x2xf32, 0x80x80x2xi64) <- (80x80x2xf32, 2xi64)
        reshape_6, reshape_7 = paddle.reshape(stack_3, full_int_array_0), None

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6]

    def op_scale_9(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_1, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6):
    
        # EarlyReturn(0, 79)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(cast_1, full_3, 0.5, True)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, scale_9]

    def op_scale_10(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, full_7, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, scale_9):
    
        # EarlyReturn(0, 80)

        # pd_op.scale: (40xf32) <- (40xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(scale_9, full_7, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, scale_10]

    def op_combine_14(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, scale_10):
    
        # EarlyReturn(0, 81)

        # builtin.combine: ([40xf32, 40xf32]) <- (40xf32, 40xf32)
        combine_14 = [scale_10, scale_10]

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, combine_14]

    def op_meshgrid_4(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, combine_14):
    
        # EarlyReturn(0, 82)

        # pd_op.meshgrid: ([40x40xf32, 40x40xf32]) <- ([40xf32, 40xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_14)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, meshgrid_4]

    def op_split_5(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, meshgrid_4):
    
        # EarlyReturn(0, 83)

        # builtin.split: (40x40xf32, 40x40xf32) <- ([40x40xf32, 40x40xf32])
        split_10, split_11, = meshgrid_4

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, split_10, split_11]

    def op_combine_15(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, split_10, split_11):
    
        # EarlyReturn(0, 84)

        # builtin.combine: ([40x40xf32, 40x40xf32]) <- (40x40xf32, 40x40xf32)
        combine_15 = [split_11, split_10]

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, combine_15]

    def op_stack_4(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, combine_15):
    
        # EarlyReturn(0, 85)

        # pd_op.stack: (40x40x2xf32) <- ([40x40xf32, 40x40xf32])
        stack_4 = paddle._C_ops.stack(combine_15, -1)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, stack_4]

    def op_reshape_4(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, stack_4):
    
        # EarlyReturn(0, 86)

        # pd_op.reshape: (1600x2xf32, 0x40x40x2xi64) <- (40x40x2xf32, 2xi64)
        reshape_8, reshape_9 = paddle.reshape(stack_4, full_int_array_0), None

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8]

    def op_scale_11(self, full_0, assign_0, assign_1, concat_0, concat_2, full_3, full_int_array_0, cast_2, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8):
    
        # EarlyReturn(0, 87)

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(cast_2, full_3, 0.5, True)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, scale_11]

    def op_scale_12(self, full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_10, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, scale_11):
    
        # EarlyReturn(0, 88)

        # pd_op.scale: (20xf32) <- (20xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(scale_11, full_10, 0, True)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, scale_12]

    def op_combine_16(self, full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, scale_12):
    
        # EarlyReturn(0, 89)

        # builtin.combine: ([20xf32, 20xf32]) <- (20xf32, 20xf32)
        combine_16 = [scale_12, scale_12]

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, combine_16]

    def op_meshgrid_5(self, full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, combine_16):
    
        # EarlyReturn(0, 90)

        # pd_op.meshgrid: ([20x20xf32, 20x20xf32]) <- ([20xf32, 20xf32])
        meshgrid_5 = paddle._C_ops.meshgrid(combine_16)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, meshgrid_5]

    def op_split_6(self, full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, meshgrid_5):
    
        # EarlyReturn(0, 91)

        # builtin.split: (20x20xf32, 20x20xf32) <- ([20x20xf32, 20x20xf32])
        split_12, split_13, = meshgrid_5

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, split_12, split_13]

    def op_combine_17(self, full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, split_12, split_13):
    
        # EarlyReturn(0, 92)

        # builtin.combine: ([20x20xf32, 20x20xf32]) <- (20x20xf32, 20x20xf32)
        combine_17 = [split_13, split_12]

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, combine_17]

    def op_stack_5(self, full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, combine_17):
    
        # EarlyReturn(0, 93)

        # pd_op.stack: (20x20x2xf32) <- ([20x20xf32, 20x20xf32])
        stack_5 = paddle._C_ops.stack(combine_17, -1)

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, stack_5]

    def op_reshape_5(self, full_0, assign_0, assign_1, concat_0, concat_2, full_int_array_0, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, stack_5):
    
        # EarlyReturn(0, 94)

        # pd_op.reshape: (400x2xf32, 0x20x20x2xi64) <- (20x20x2xf32, 2xi64)
        reshape_10, reshape_11 = paddle.reshape(stack_5, full_int_array_0), None

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, reshape_10]

    def op_combine_18(self, full_0, assign_0, assign_1, concat_0, concat_2, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, reshape_10):
    
        # EarlyReturn(0, 95)

        # builtin.combine: ([6400x2xf32, 1600x2xf32, 400x2xf32]) <- (6400x2xf32, 1600x2xf32, 400x2xf32)
        combine_18 = [reshape_6, reshape_8, reshape_10]

        return [full_0, assign_0, assign_1, concat_0, concat_2, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, reshape_10, combine_18]

    def op_concat_6(self, full_0, assign_0, assign_1, concat_0, concat_2, full_12, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, reshape_10, combine_18):
    
        # EarlyReturn(0, 96)

        # pd_op.concat: (8400x2xf32) <- ([6400x2xf32, 1600x2xf32, 400x2xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_18, full_12)

        return [full_0, assign_0, assign_1, concat_0, concat_2, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, reshape_10, concat_6]

    def op_cast_4(self, full_0, assign_0, assign_1, concat_0, concat_2, assign_2, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_5, reshape_6, reshape_8, reshape_10, concat_6):
    
        # EarlyReturn(0, 97)

        # pd_op.cast: (8400x2xf32) <- (8400x2xf32)
        cast_4 = paddle._C_ops.cast(concat_6, paddle.float32)

        return [full_0, assign_1, assign_0, full_13, split_6, divide_0, add_0, exp_0, full_14, scale_6, subtract_0, add_1, full_15, concat_0, concat_5, concat_2, cast_4, assign_2, reshape_6, reshape_8, reshape_10]

if True and not (IsCinnStageEnableDiff() and LastCINNStageFailed()):

    class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
        def prepare_data(self):
            self.inputs = [
                # data_0
                paddle.uniform([1, 6400, 80], dtype='float32', min=0, max=0.5),
                # data_1
                paddle.uniform([1, 1600, 80], dtype='float32', min=0, max=0.5),
                # data_2
                paddle.uniform([1, 400, 80], dtype='float32', min=0, max=0.5),
                # data_3
                paddle.uniform([1, 6400, 4], dtype='float32', min=0, max=0.5),
                # data_4
                paddle.uniform([1, 1600, 4], dtype='float32', min=0, max=0.5),
                # data_5
                paddle.uniform([1, 400, 4], dtype='float32', min=0, max=0.5),
                # data_6
                paddle.uniform([1, 6400, 1], dtype='float32', min=0, max=0.5),
                # data_7
                paddle.uniform([1, 1600, 1], dtype='float32', min=0, max=0.5),
                # data_8
                paddle.uniform([1, 400, 1], dtype='float32', min=0, max=0.5),
            ]
            for input in self.inputs:
                input.stop_gradient = True

        def apply_to_static(self, net, use_cinn):
            build_strategy = paddle.static.BuildStrategy()
            input_spec = [
                # data_0
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                # data_1
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                # data_2
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                # data_3
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                # data_4
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                # data_5
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                # data_6
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                # data_7
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                # data_8
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
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