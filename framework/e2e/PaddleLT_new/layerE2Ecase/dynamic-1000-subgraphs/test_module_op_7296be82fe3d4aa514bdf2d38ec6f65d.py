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
            PADDLE_DEBUG_ENABLE_CINN=False,
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
    PADDLE_DEBUG_CINN_STAGE_NAME="backend",
    PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF=False,
    PADDLE_DEBUG_ENABLE_CINN=True,
    FLAGS_enable_pir_api=True,
    FLAGS_prim_all=True,
    FLAGS_prim_enable_dynamic=True,
    FLAGS_use_cinn=False,
    FLAGS_check_infer_symbolic=False,
    FLAGS_enable_fusion_fallback=False,
)

last_stage_failed = (IsCinnStageEnableDiff() and LastCINNStageFailed())

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
    return [78][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2):

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_0 = data_0 + data_1

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 1.11111, paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_0

        # pd_op.scale: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_0, 0, True)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_0 = paddle._C_ops.shape(add_0)

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

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(slice_1, paddle.int64)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full([], 1, paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [cast_0, cast_1, full_1, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(stack_0, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 0.1, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(uniform_0, full_4)

        # pd_op.cast: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        cast_2 = paddle._C_ops.cast(scale_0, paddle.float32)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_3 = paddle._C_ops.cast(greater_equal_0, paddle.float32)

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x-1x1x1xf32)
        multiply_0 = cast_2 * cast_3

        # pd_op.conv2d: (-1x21x-1x-1xf32) <- (-1x512x-1x-1xf32, 21x512x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(multiply_0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, -1, 1, 1]

        # pd_op.reshape: (1x21x1x1xf32, 0x21xi64) <- (21xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_3), None

        # pd_op.add: (-1x21x-1x-1xf32) <- (-1x21x-1x-1xf32, 1x21x1x1xf32)
        add_1 = conv2d_0 + reshape_0

        # pd_op.scale: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(data_1, assign_2, 0, True)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_1 = paddle._C_ops.shape(data_1)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_1, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_1, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.cast: (xi64) <- (xi32)
        cast_4 = paddle._C_ops.cast(slice_2, paddle.int64)

        # pd_op.cast: (xi64) <- (xi32)
        cast_5 = paddle._C_ops.cast(slice_3, paddle.int64)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [cast_4, cast_5, full_1, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(stack_1, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_1 = paddle._C_ops.greater_equal(uniform_1, full_4)

        # pd_op.cast: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        cast_6 = paddle._C_ops.cast(scale_1, paddle.float32)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_7 = paddle._C_ops.cast(greater_equal_1, paddle.float32)

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x-1x1x1xf32)
        multiply_1 = cast_6 * cast_7

        # pd_op.conv2d: (-1x21x-1x-1xf32) <- (-1x512x-1x-1xf32, 21x512x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(multiply_1, parameter_2, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x21x1x1xf32, 0x21xi64) <- (21xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(parameter_3, full_int_array_3), None

        # pd_op.assign: (0x21xi64) <- (0x21xi64)
        assign_3 = reshape_3

        # pd_op.assign: (1x21x1x1xf32) <- (1x21x1x1xf32)
        assign_4 = reshape_2

        # pd_op.add: (-1x21x-1x-1xf32) <- (-1x21x-1x-1xf32, 1x21x1x1xf32)
        add_2 = conv2d_1 + reshape_2

        # pd_op.scale: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(data_0, assign_1, 0, True)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_2 = paddle._C_ops.shape(data_0)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_2, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_2, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.cast: (xi64) <- (xi32)
        cast_8 = paddle._C_ops.cast(slice_4, paddle.int64)

        # pd_op.cast: (xi64) <- (xi32)
        cast_9 = paddle._C_ops.cast(slice_5, paddle.int64)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [cast_8, cast_9, full_1, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(stack_2, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_2 = paddle._C_ops.greater_equal(uniform_2, full_4)

        # pd_op.cast: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        cast_10 = paddle._C_ops.cast(scale_2, paddle.float32)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_11 = paddle._C_ops.cast(greater_equal_2, paddle.float32)

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x-1x1x1xf32)
        multiply_2 = cast_10 * cast_11

        # pd_op.conv2d: (-1x21x-1x-1xf32) <- (-1x512x-1x-1xf32, 21x512x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(multiply_2, parameter_2, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x21x-1x-1xf32) <- (-1x21x-1x-1xf32, 1x21x1x1xf32)
        add_3 = conv2d_2 + assign_4

        # pd_op.scale: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(data_2, assign_0, 0, True)

        # pd_op.shape: (4xi32) <- (-1x2048x-1x-1xf32)
        shape_3 = paddle._C_ops.shape(data_2)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_3, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_3, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.cast: (xi64) <- (xi32)
        cast_12 = paddle._C_ops.cast(slice_6, paddle.int64)

        # pd_op.cast: (xi64) <- (xi32)
        cast_13 = paddle._C_ops.cast(slice_7, paddle.int64)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_3 = [cast_12, cast_13, full_1, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(stack_3, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_3 = paddle._C_ops.greater_equal(uniform_3, full_4)

        # pd_op.cast: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32)
        cast_14 = paddle._C_ops.cast(scale_3, paddle.float32)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_15 = paddle._C_ops.cast(greater_equal_3, paddle.float32)

        # pd_op.multiply: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, -1x-1x1x1xf32)
        multiply_3 = cast_14 * cast_15

        # pd_op.conv2d: (-1x21x-1x-1xf32) <- (-1x2048x-1x-1xf32, 21x2048x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(multiply_3, parameter_4, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x21x1x1xf32, 0x21xi64) <- (21xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_5, full_int_array_3), None

        # pd_op.add: (-1x21x-1x-1xf32) <- (-1x21x-1x-1xf32, 1x21x1x1xf32)
        add_4 = conv2d_3 + reshape_4
        return full_0, cast_2, cast_3, multiply_0, conv2d_0, reshape_0, reshape_1, assign_2, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_1, cast_10, cast_11, multiply_2, conv2d_2, assign_4, assign_3, assign_0, cast_14, cast_15, multiply_3, conv2d_3, reshape_4, reshape_5, add_1, add_2, add_3, add_4



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

    def forward(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2):
        args = [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_add_0,
            self.op_full_0,
            self.op_assign_0,
            self.op_assign_1,
            self.op_assign_2,
            self.op_scale_0,
            self.op_shape_0,
            self.op_full_int_array_0,
            self.op_full_int_array_1,
            self.op_slice_0,
            self.op_full_int_array_2,
            self.op_slice_1,
            self.op_cast_0,
            self.op_cast_1,
            self.op_full_1,
            self.op_combine_0,
            self.op_stack_0,
            self.op_full_2,
            self.op_full_3,
            self.op_uniform_0,
            self.op_full_4,
            self.op_greater_equal_0,
            self.op_cast_2,
            self.op_cast_3,
            self.op_multiply_0,
            self.op_conv2d_0,
            self.op_full_int_array_3,
            self.op_reshape_0,
            self.op_add_1,
            self.op_scale_1,
            self.op_shape_1,
            self.op_slice_2,
            self.op_slice_3,
            self.op_cast_4,
            self.op_cast_5,
            self.op_combine_1,
            self.op_stack_1,
            self.op_uniform_1,
            self.op_greater_equal_1,
            self.op_cast_6,
            self.op_cast_7,
            self.op_multiply_1,
            self.op_conv2d_1,
            self.op_reshape_1,
            self.op_assign_3,
            self.op_assign_4,
            self.op_add_2,
            self.op_scale_2,
            self.op_shape_2,
            self.op_slice_4,
            self.op_slice_5,
            self.op_cast_8,
            self.op_cast_9,
            self.op_combine_2,
            self.op_stack_2,
            self.op_uniform_2,
            self.op_greater_equal_2,
            self.op_cast_10,
            self.op_cast_11,
            self.op_multiply_2,
            self.op_conv2d_2,
            self.op_add_3,
            self.op_scale_3,
            self.op_shape_3,
            self.op_slice_6,
            self.op_slice_7,
            self.op_cast_12,
            self.op_cast_13,
            self.op_combine_3,
            self.op_stack_3,
            self.op_uniform_3,
            self.op_greater_equal_3,
            self.op_cast_14,
            self.op_cast_15,
            self.op_multiply_3,
            self.op_conv2d_3,
            self.op_reshape_2,
            self.op_add_4,
        ]

    def op_add_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2):
    
        # EarlyReturn(0, 0)

        # pd_op.add: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x512x-1x-1xf32)
        add_0 = data_0 + data_1

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0]

    def op_full_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0):
    
        # EarlyReturn(0, 1)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 1.11111, paddle.float32, paddle.core.CPUPlace())

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0]

    def op_assign_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0):
    
        # EarlyReturn(0, 2)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0, assign_0]

    def op_assign_1(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0, assign_0):
    
        # EarlyReturn(0, 3)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_0

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0, assign_0, assign_1]

    def op_assign_2(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0, assign_0, assign_1):
    
        # EarlyReturn(0, 4)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_0

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0, assign_0, assign_1, assign_2]

    def op_scale_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0, assign_0, assign_1, assign_2):
    
        # EarlyReturn(0, 5)

        # pd_op.scale: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_0, full_0, 0, True)

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0, assign_0, assign_1, assign_2, scale_0]

    def op_shape_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, add_0, full_0, assign_0, assign_1, assign_2, scale_0):
    
        # EarlyReturn(0, 6)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_0 = paddle._C_ops.shape(add_0)

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0]

    def op_full_int_array_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0):
    
        # EarlyReturn(0, 7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0, full_int_array_0]

    def op_full_int_array_1(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0, full_int_array_0):
    
        # EarlyReturn(0, 8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0, full_int_array_0, full_int_array_1]

    def op_slice_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0, full_int_array_0, full_int_array_1):
    
        # EarlyReturn(0, 9)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0, full_int_array_0, full_int_array_1, slice_0]

    def op_full_int_array_2(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0, full_int_array_0, full_int_array_1, slice_0):
    
        # EarlyReturn(0, 10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0, full_int_array_0, full_int_array_1, slice_0, full_int_array_2]

    def op_slice_1(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, shape_0, full_int_array_0, full_int_array_1, slice_0, full_int_array_2):
    
        # EarlyReturn(0, 11)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_0, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, slice_0, full_int_array_2, slice_1]

    def op_cast_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, slice_0, full_int_array_2, slice_1):
    
        # EarlyReturn(0, 12)

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, slice_1, cast_0]

    def op_cast_1(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, slice_1, cast_0):
    
        # EarlyReturn(0, 13)

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(slice_1, paddle.int64)

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, cast_0, cast_1]

    def op_full_1(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, cast_0, cast_1):
    
        # EarlyReturn(0, 14)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full([], 1, paddle.int64, paddle.core.CPUPlace())

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, cast_0, cast_1, full_1]

    def op_combine_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, cast_0, cast_1, full_1):
    
        # EarlyReturn(0, 15)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [cast_0, cast_1, full_1, full_1]

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, combine_0]

    def op_stack_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, combine_0):
    
        # EarlyReturn(0, 16)

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, stack_0]

    def op_full_2(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, stack_0):
    
        # EarlyReturn(0, 17)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, stack_0, full_2]

    def op_full_3(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, stack_0, full_2):
    
        # EarlyReturn(0, 18)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, stack_0, full_2, full_3]

    def op_uniform_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, stack_0, full_2, full_3):
    
        # EarlyReturn(0, 19)

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(stack_0, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, uniform_0]

    def op_full_4(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, uniform_0):
    
        # EarlyReturn(0, 20)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 0.1, paddle.float32, paddle.framework._current_expected_place())

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, uniform_0, full_4]

    def op_greater_equal_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, uniform_0, full_4):
    
        # EarlyReturn(0, 21)

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(uniform_0, full_4)

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, greater_equal_0]

    def op_cast_2(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, scale_0, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, greater_equal_0):
    
        # EarlyReturn(0, 22)

        # pd_op.cast: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        cast_2 = paddle._C_ops.cast(scale_0, paddle.float32)

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, greater_equal_0, cast_2]

    def op_cast_3(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, greater_equal_0, cast_2):
    
        # EarlyReturn(0, 23)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_3 = paddle._C_ops.cast(greater_equal_0, paddle.float32)

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3]

    def op_multiply_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3):
    
        # EarlyReturn(0, 24)

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x-1x1x1xf32)
        multiply_0 = cast_2 * cast_3

        return [parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0]

    def op_conv2d_0(self, parameter_0, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0):
    
        # EarlyReturn(0, 25)

        # pd_op.conv2d: (-1x21x-1x-1xf32) <- (-1x512x-1x-1xf32, 21x512x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(multiply_0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0]

    def op_full_int_array_3(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0):
    
        # EarlyReturn(0, 26)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, -1, 1, 1]

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3]

    def op_reshape_0(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3):
    
        # EarlyReturn(0, 27)

        # pd_op.reshape: (1x21x1x1xf32, 0x21xi64) <- (21xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_3), None

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1]

    def op_add_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1):
    
        # EarlyReturn(0, 28)

        # pd_op.add: (-1x21x-1x-1xf32) <- (-1x21x-1x-1xf32, 1x21x1x1xf32)
        add_1 = conv2d_0 + reshape_0

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1]

    def op_scale_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1):
    
        # EarlyReturn(0, 29)

        # pd_op.scale: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(data_1, assign_2, 0, True)

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1]

    def op_shape_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_1, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1):
    
        # EarlyReturn(0, 30)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_1 = paddle._C_ops.shape(data_1)

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, shape_1]

    def op_slice_2(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, shape_1):
    
        # EarlyReturn(0, 31)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_1, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, shape_1, slice_2]

    def op_slice_3(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, shape_1, slice_2):
    
        # EarlyReturn(0, 32)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_1, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, slice_2, slice_3]

    def op_cast_4(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, slice_2, slice_3):
    
        # EarlyReturn(0, 33)

        # pd_op.cast: (xi64) <- (xi32)
        cast_4 = paddle._C_ops.cast(slice_2, paddle.int64)

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, slice_3, cast_4]

    def op_cast_5(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, slice_3, cast_4):
    
        # EarlyReturn(0, 34)

        # pd_op.cast: (xi64) <- (xi32)
        cast_5 = paddle._C_ops.cast(slice_3, paddle.int64)

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, cast_4, cast_5]

    def op_combine_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, cast_4, cast_5):
    
        # EarlyReturn(0, 35)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [cast_4, cast_5, full_1, full_1]

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, combine_1]

    def op_stack_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, combine_1):
    
        # EarlyReturn(0, 36)

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, stack_1]

    def op_uniform_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, stack_1):
    
        # EarlyReturn(0, 37)

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_1 = paddle._C_ops.uniform(stack_1, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, uniform_1]

    def op_greater_equal_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, uniform_1):
    
        # EarlyReturn(0, 38)

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_1 = paddle._C_ops.greater_equal(uniform_1, full_4)

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, greater_equal_1]

    def op_cast_6(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, scale_1, greater_equal_1):
    
        # EarlyReturn(0, 39)

        # pd_op.cast: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        cast_6 = paddle._C_ops.cast(scale_1, paddle.float32)

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, greater_equal_1, cast_6]

    def op_cast_7(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, greater_equal_1, cast_6):
    
        # EarlyReturn(0, 40)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_7 = paddle._C_ops.cast(greater_equal_1, paddle.float32)

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7]

    def op_multiply_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7):
    
        # EarlyReturn(0, 41)

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x-1x1x1xf32)
        multiply_1 = cast_6 * cast_7

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1]

    def op_conv2d_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1):
    
        # EarlyReturn(0, 42)

        # pd_op.conv2d: (-1x21x-1x-1xf32) <- (-1x512x-1x-1xf32, 21x512x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(multiply_1, parameter_2, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1]

    def op_reshape_1(self, parameter_2, parameter_3, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1):
    
        # EarlyReturn(0, 43)

        # pd_op.reshape: (1x21x1x1xf32, 0x21xi64) <- (21xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(parameter_3, full_int_array_3), None

        return [parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3]

    def op_assign_3(self, parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3):
    
        # EarlyReturn(0, 44)

        # pd_op.assign: (0x21xi64) <- (0x21xi64)
        assign_3 = reshape_3

        return [parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3]

    def op_assign_4(self, parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3):
    
        # EarlyReturn(0, 45)

        # pd_op.assign: (1x21x1x1xf32) <- (1x21x1x1xf32)
        assign_4 = reshape_2

        return [parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4]

    def op_add_2(self, parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4):
    
        # EarlyReturn(0, 46)

        # pd_op.add: (-1x21x-1x-1xf32) <- (-1x21x-1x-1xf32, 1x21x1x1xf32)
        add_2 = conv2d_1 + reshape_2

        return [parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2]

    def op_scale_2(self, parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2):
    
        # EarlyReturn(0, 47)

        # pd_op.scale: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(data_0, assign_1, 0, True)

        return [parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2]

    def op_shape_2(self, parameter_2, parameter_5, parameter_4, data_0, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2):
    
        # EarlyReturn(0, 48)

        # pd_op.shape: (4xi32) <- (-1x512x-1x-1xf32)
        shape_2 = paddle._C_ops.shape(data_0)

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, shape_2]

    def op_slice_4(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, shape_2):
    
        # EarlyReturn(0, 49)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_2, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, shape_2, slice_4]

    def op_slice_5(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, shape_2, slice_4):
    
        # EarlyReturn(0, 50)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_2, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, slice_4, slice_5]

    def op_cast_8(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, slice_4, slice_5):
    
        # EarlyReturn(0, 51)

        # pd_op.cast: (xi64) <- (xi32)
        cast_8 = paddle._C_ops.cast(slice_4, paddle.int64)

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, slice_5, cast_8]

    def op_cast_9(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, slice_5, cast_8):
    
        # EarlyReturn(0, 52)

        # pd_op.cast: (xi64) <- (xi32)
        cast_9 = paddle._C_ops.cast(slice_5, paddle.int64)

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, cast_8, cast_9]

    def op_combine_2(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, cast_8, cast_9):
    
        # EarlyReturn(0, 53)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [cast_8, cast_9, full_1, full_1]

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, combine_2]

    def op_stack_2(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, combine_2):
    
        # EarlyReturn(0, 54)

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, stack_2]

    def op_uniform_2(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, stack_2):
    
        # EarlyReturn(0, 55)

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_2 = paddle._C_ops.uniform(stack_2, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, uniform_2]

    def op_greater_equal_2(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, uniform_2):
    
        # EarlyReturn(0, 56)

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_2 = paddle._C_ops.greater_equal(uniform_2, full_4)

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, greater_equal_2]

    def op_cast_10(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, scale_2, greater_equal_2):
    
        # EarlyReturn(0, 57)

        # pd_op.cast: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32)
        cast_10 = paddle._C_ops.cast(scale_2, paddle.float32)

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, greater_equal_2, cast_10]

    def op_cast_11(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, greater_equal_2, cast_10):
    
        # EarlyReturn(0, 58)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_11 = paddle._C_ops.cast(greater_equal_2, paddle.float32)

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11]

    def op_multiply_2(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11):
    
        # EarlyReturn(0, 59)

        # pd_op.multiply: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, -1x-1x1x1xf32)
        multiply_2 = cast_10 * cast_11

        return [parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2]

    def op_conv2d_2(self, parameter_2, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2):
    
        # EarlyReturn(0, 60)

        # pd_op.conv2d: (-1x21x-1x-1xf32) <- (-1x512x-1x-1xf32, 21x512x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(multiply_2, parameter_2, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2]

    def op_add_3(self, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2):
    
        # EarlyReturn(0, 61)

        # pd_op.add: (-1x21x-1x-1xf32) <- (-1x21x-1x-1xf32, 1x21x1x1xf32)
        add_3 = conv2d_2 + assign_4

        return [parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3]

    def op_scale_3(self, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3):
    
        # EarlyReturn(0, 62)

        # pd_op.scale: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(data_2, assign_0, 0, True)

        return [parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3]

    def op_shape_3(self, parameter_5, parameter_4, data_2, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3):
    
        # EarlyReturn(0, 63)

        # pd_op.shape: (4xi32) <- (-1x2048x-1x-1xf32)
        shape_3 = paddle._C_ops.shape(data_2)

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, shape_3]

    def op_slice_6(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_int_array_0, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, shape_3):
    
        # EarlyReturn(0, 64)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_3, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, shape_3, slice_6]

    def op_slice_7(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_int_array_1, full_int_array_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, shape_3, slice_6):
    
        # EarlyReturn(0, 65)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_3, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, slice_6, slice_7]

    def op_cast_12(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, slice_6, slice_7):
    
        # EarlyReturn(0, 66)

        # pd_op.cast: (xi64) <- (xi32)
        cast_12 = paddle._C_ops.cast(slice_6, paddle.int64)

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, slice_7, cast_12]

    def op_cast_13(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, slice_7, cast_12):
    
        # EarlyReturn(0, 67)

        # pd_op.cast: (xi64) <- (xi32)
        cast_13 = paddle._C_ops.cast(slice_7, paddle.int64)

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, cast_12, cast_13]

    def op_combine_3(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_1, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, cast_12, cast_13):
    
        # EarlyReturn(0, 68)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_3 = [cast_12, cast_13, full_1, full_1]

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, combine_3]

    def op_stack_3(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, combine_3):
    
        # EarlyReturn(0, 69)

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, stack_3]

    def op_uniform_3(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_2, full_3, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, stack_3):
    
        # EarlyReturn(0, 70)

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_3 = paddle._C_ops.uniform(stack_3, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, uniform_3]

    def op_greater_equal_3(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, full_4, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, uniform_3):
    
        # EarlyReturn(0, 71)

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_3 = paddle._C_ops.greater_equal(uniform_3, full_4)

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, greater_equal_3]

    def op_cast_14(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, scale_3, greater_equal_3):
    
        # EarlyReturn(0, 72)

        # pd_op.cast: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32)
        cast_14 = paddle._C_ops.cast(scale_3, paddle.float32)

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, greater_equal_3, cast_14]

    def op_cast_15(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, greater_equal_3, cast_14):
    
        # EarlyReturn(0, 73)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_15 = paddle._C_ops.cast(greater_equal_3, paddle.float32)

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, cast_14, cast_15]

    def op_multiply_3(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, cast_14, cast_15):
    
        # EarlyReturn(0, 74)

        # pd_op.multiply: (-1x2048x-1x-1xf32) <- (-1x2048x-1x-1xf32, -1x-1x1x1xf32)
        multiply_3 = cast_14 * cast_15

        return [parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, cast_14, cast_15, multiply_3]

    def op_conv2d_3(self, parameter_5, parameter_4, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, cast_14, cast_15, multiply_3):
    
        # EarlyReturn(0, 75)

        # pd_op.conv2d: (-1x21x-1x-1xf32) <- (-1x2048x-1x-1xf32, 21x2048x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(multiply_3, parameter_4, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_5, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, cast_14, cast_15, multiply_3, conv2d_3]

    def op_reshape_2(self, parameter_5, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, full_int_array_3, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, cast_14, cast_15, multiply_3, conv2d_3):
    
        # EarlyReturn(0, 76)

        # pd_op.reshape: (1x21x1x1xf32, 0x21xi64) <- (21xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_5, full_int_array_3), None

        return [full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, cast_14, cast_15, multiply_3, conv2d_3, reshape_4, reshape_5]

    def op_add_4(self, full_0, assign_0, assign_1, assign_2, cast_2, cast_3, multiply_0, conv2d_0, reshape_0, reshape_1, add_1, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_3, assign_4, add_2, cast_10, cast_11, multiply_2, conv2d_2, add_3, cast_14, cast_15, multiply_3, conv2d_3, reshape_4, reshape_5):
    
        # EarlyReturn(0, 77)

        # pd_op.add: (-1x21x-1x-1xf32) <- (-1x21x-1x-1xf32, 1x21x1x1xf32)
        add_4 = conv2d_3 + reshape_4

        return [full_0, cast_2, cast_3, multiply_0, conv2d_0, reshape_0, reshape_1, assign_2, cast_6, cast_7, multiply_1, conv2d_1, reshape_2, reshape_3, assign_1, cast_10, cast_11, multiply_2, conv2d_2, assign_4, assign_3, assign_0, cast_14, cast_15, multiply_3, conv2d_3, reshape_4, reshape_5, add_1, add_2, add_3, add_4]

is_module_block_and_last_stage_passed = (
    True and not last_stage_failed
)
@unittest.skipIf(not is_module_block_and_last_stage_passed, "last stage failed")
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([21, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([21], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([21], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([21, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([21], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[21, 512, 1, 1], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[21, 512, 1, 1], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[21], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[21], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[21, 2048, 1, 1], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[21], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
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