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

def GetExitCodeAndStdErr(cmd, env):
    import subprocess
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return result.returncode, result.stderr

def GetStageExitCodeAndStdErr(stage):
    return GetExitCodeAndStdErr(
        [sys.executable, __file__],
        env=dict(
            PADDLE_DEBUG_CINN_STAGE_NAME=stage.name,
            PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
            PYTHONPATH=os.getenv('PYTHONPATH'),
            ATHENA_ENABLE_TRY_RUN="False",
        ),
    )

def AthenaTryRunEnabled():
    return os.getenv('ATHENA_ENABLE_TRY_RUN') not in {
        "0",
        "False",
        "false",
        "OFF"
    }

def GetNeedSkipAndSkipMessage():
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    if not IsCinnStageEnableDiff():
        return False, ""
    last_stage = GetPrevCinnStage(current_stage)
    if last_stage is None:
        return False, ""
    exitcode, stderr = GetStageExitCodeAndStdErr(last_stage)
    if exitcode != 0:
        return True, f"last stage failed. stderr: {stderr}"
    return False, ""

def GetCurrentStageTryRunExitCodeAndStdErr():
    if not AthenaTryRunEnabled():
        return False, ""
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    return GetStageExitCodeAndStdErr(current_stage)

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

need_skip, skip_message = GetNeedSkipAndSkipMessage()
try_run_exit_code, try_run_stderr = GetCurrentStageTryRunExitCodeAndStdErr()

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
    return [77][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        # pd_op.pool2d: (1x20x1x1xf32) <- (1x20x128x256xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(data_0, full_int_array_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (1x5x1x1xf32) <- (1x20x1x1xf32, 5x20x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(pool2d_0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x5x1x1xf32, 0x5xi64) <- (5xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_1), None

        # pd_op.add: (1x5x1x1xf32) <- (1x5x1x1xf32, 1x5x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        # pd_op.relu: (1x5x1x1xf32) <- (1x5x1x1xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (1x20x1x1xf32) <- (1x5x1x1xf32, 20x5x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_2, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x20x1x1xf32, 0x20xi64) <- (20xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(parameter_3, full_int_array_1), None

        # pd_op.add: (1x20x1x1xf32) <- (1x20x1x1xf32, 1x20x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        # pd_op.sigmoid: (1x20x1x1xf32) <- (1x20x1x1xf32)
        sigmoid_0 = paddle.nn.functional.sigmoid(add_1)

        # pd_op.multiply: (1x20x128x256xf32) <- (1x20x128x256xf32, 1x20x1x1xf32)
        multiply_0 = data_0 * sigmoid_0

        # pd_op.pool2d: (1x40x1x1xf32) <- (1x40x64x128xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(data_1, assign_2, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (1x10x1x1xf32) <- (1x40x1x1xf32, 10x40x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_1, parameter_4, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x10x1x1xf32, 0x10xi64) <- (10xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_5, full_int_array_1), None

        # pd_op.add: (1x10x1x1xf32) <- (1x10x1x1xf32, 1x10x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        # pd_op.relu: (1x10x1x1xf32) <- (1x10x1x1xf32)
        relu_1 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (1x40x1x1xf32) <- (1x10x1x1xf32, 40x10x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu_1, parameter_6, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x40x1x1xf32, 0x40xi64) <- (40xf32, 4xi64)
        reshape_6, reshape_7 = paddle.reshape(parameter_7, full_int_array_1), None

        # pd_op.add: (1x40x1x1xf32) <- (1x40x1x1xf32, 1x40x1x1xf32)
        add_3 = conv2d_3 + reshape_6

        # pd_op.sigmoid: (1x40x1x1xf32) <- (1x40x1x1xf32)
        sigmoid_1 = paddle.nn.functional.sigmoid(add_3)

        # pd_op.multiply: (1x40x64x128xf32) <- (1x40x64x128xf32, 1x40x1x1xf32)
        multiply_1 = data_1 * sigmoid_1

        # pd_op.pool2d: (1x80x1x1xf32) <- (1x80x32x64xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(data_2, assign_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (1x20x1x1xf32) <- (1x80x1x1xf32, 20x80x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_2, parameter_8, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x20x1x1xf32, 0x20xi64) <- (20xf32, 4xi64)
        reshape_8, reshape_9 = paddle.reshape(parameter_9, full_int_array_1), None

        # pd_op.add: (1x20x1x1xf32) <- (1x20x1x1xf32, 1x20x1x1xf32)
        add_4 = conv2d_4 + reshape_8

        # pd_op.relu: (1x20x1x1xf32) <- (1x20x1x1xf32)
        relu_2 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (1x80x1x1xf32) <- (1x20x1x1xf32, 80x20x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu_2, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x80x1x1xf32, 0x80xi64) <- (80xf32, 4xi64)
        reshape_10, reshape_11 = paddle.reshape(parameter_11, full_int_array_1), None

        # pd_op.add: (1x80x1x1xf32) <- (1x80x1x1xf32, 1x80x1x1xf32)
        add_5 = conv2d_5 + reshape_10

        # pd_op.sigmoid: (1x80x1x1xf32) <- (1x80x1x1xf32)
        sigmoid_2 = paddle.nn.functional.sigmoid(add_5)

        # pd_op.multiply: (1x80x32x64xf32) <- (1x80x32x64xf32, 1x80x1x1xf32)
        multiply_2 = data_2 * sigmoid_2

        # pd_op.pool2d: (1x160x1x1xf32) <- (1x160x16x32xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(data_3, assign_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (1x40x1x1xf32) <- (1x160x1x1xf32, 40x160x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(pool2d_3, parameter_12, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x40x1x1xf32, 0x40xi64) <- (40xf32, 4xi64)
        reshape_12, reshape_13 = paddle.reshape(parameter_13, full_int_array_1), None

        # pd_op.add: (1x40x1x1xf32) <- (1x40x1x1xf32, 1x40x1x1xf32)
        add_6 = conv2d_6 + reshape_12

        # pd_op.relu: (1x40x1x1xf32) <- (1x40x1x1xf32)
        relu_3 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (1x160x1x1xf32) <- (1x40x1x1xf32, 160x40x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu_3, parameter_14, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x160x1x1xf32, 0x160xi64) <- (160xf32, 4xi64)
        reshape_14, reshape_15 = paddle.reshape(parameter_15, full_int_array_1), None

        # pd_op.add: (1x160x1x1xf32) <- (1x160x1x1xf32, 1x160x1x1xf32)
        add_7 = conv2d_7 + reshape_14

        # pd_op.sigmoid: (1x160x1x1xf32) <- (1x160x1x1xf32)
        sigmoid_3 = paddle.nn.functional.sigmoid(add_7)

        # pd_op.multiply: (1x160x16x32xf32) <- (1x160x16x32xf32, 1x160x1x1xf32)
        multiply_3 = data_3 * sigmoid_3

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

        # builtin.combine: ([1x20x128x256xf32, 1x20x128x256xf32]) <- (1x20x128x256xf32, 1x20x128x256xf32)
        combine_0 = [data_4, multiply_0]

        # pd_op.concat: (1x40x128x256xf32) <- ([1x20x128x256xf32, 1x20x128x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([1x40x64x128xf32, 1x40x64x128xf32]) <- (1x40x64x128xf32, 1x40x64x128xf32)
        combine_1 = [data_5, multiply_1]

        # pd_op.concat: (1x80x64x128xf32) <- ([1x40x64x128xf32, 1x40x64x128xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, assign_5)

        # builtin.combine: ([1x80x32x64xf32, 1x80x32x64xf32]) <- (1x80x32x64xf32, 1x80x32x64xf32)
        combine_2 = [data_6, multiply_2]

        # pd_op.concat: (1x160x32x64xf32) <- ([1x80x32x64xf32, 1x80x32x64xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, assign_4)

        # builtin.combine: ([1x160x16x32xf32, 1x160x16x32xf32]) <- (1x160x16x32xf32, 1x160x16x32xf32)
        combine_3 = [data_7, multiply_3]

        # pd_op.concat: (1x320x16x32xf32) <- ([1x160x16x32xf32, 1x160x16x32xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, assign_3)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_2 = [1, 2, 20, 128, 256]

        # pd_op.reshape: (1x2x20x128x256xf32, 0x1x40x128x256xi64) <- (1x40x128x256xf32, 5xi64)
        reshape_16, reshape_17 = paddle.reshape(concat_0, full_int_array_2), None

        # pd_op.transpose: (1x20x2x128x256xf32) <- (1x2x20x128x256xf32)
        transpose_0 = paddle.transpose(reshape_16, perm=[0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 40, 128, 256]

        # pd_op.reshape: (1x40x128x256xf32, 0x1x20x2x128x256xi64) <- (1x20x2x128x256xf32, 4xi64)
        reshape_18, reshape_19 = paddle.reshape(transpose_0, full_int_array_3), None

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [1, 2, 40, 64, 128]

        # pd_op.reshape: (1x2x40x64x128xf32, 0x1x80x64x128xi64) <- (1x80x64x128xf32, 5xi64)
        reshape_20, reshape_21 = paddle.reshape(concat_1, full_int_array_4), None

        # pd_op.transpose: (1x40x2x64x128xf32) <- (1x2x40x64x128xf32)
        transpose_1 = paddle.transpose(reshape_20, perm=[0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 80, 64, 128]

        # pd_op.reshape: (1x80x64x128xf32, 0x1x40x2x64x128xi64) <- (1x40x2x64x128xf32, 4xi64)
        reshape_22, reshape_23 = paddle.reshape(transpose_1, full_int_array_5), None

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_6 = [1, 2, 80, 32, 64]

        # pd_op.reshape: (1x2x80x32x64xf32, 0x1x160x32x64xi64) <- (1x160x32x64xf32, 5xi64)
        reshape_24, reshape_25 = paddle.reshape(concat_2, full_int_array_6), None

        # pd_op.transpose: (1x80x2x32x64xf32) <- (1x2x80x32x64xf32)
        transpose_2 = paddle.transpose(reshape_24, perm=[0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 160, 32, 64]

        # pd_op.reshape: (1x160x32x64xf32, 0x1x80x2x32x64xi64) <- (1x80x2x32x64xf32, 4xi64)
        reshape_26, reshape_27 = paddle.reshape(transpose_2, full_int_array_7), None

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_8 = [1, 2, 160, 16, 32]

        # pd_op.reshape: (1x2x160x16x32xf32, 0x1x320x16x32xi64) <- (1x320x16x32xf32, 5xi64)
        reshape_28, reshape_29 = paddle.reshape(concat_3, full_int_array_8), None

        # pd_op.transpose: (1x160x2x16x32xf32) <- (1x2x160x16x32xf32)
        transpose_3 = paddle.transpose(reshape_28, perm=[0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 320, 16, 32]

        # pd_op.reshape: (1x320x16x32xf32, 0x1x160x2x16x32xi64) <- (1x160x2x16x32xf32, 4xi64)
        reshape_30, reshape_31 = paddle.reshape(transpose_3, full_int_array_9), None
        return full_int_array_0, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, assign_2, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, assign_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, assign_0, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_5, assign_4, assign_3, reshape_17, reshape_19, reshape_21, reshape_23, reshape_25, reshape_27, reshape_29, reshape_31, reshape_18, reshape_22, reshape_26, reshape_30



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

    def _test_entry(self):
        dy_outs = self.entry(use_cinn=False)
        cinn_outs = self.entry(use_cinn=GetEnvVarEnableCinn())

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

    def forward(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        args = [parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_full_int_array_0,
            self.op_assign_0,
            self.op_assign_1,
            self.op_assign_2,
            self.op_pool2d_0,
            self.op_conv2d_0,
            self.op_full_int_array_1,
            self.op_reshape_0,
            self.op_add_0,
            self.op_relu_0,
            self.op_conv2d_1,
            self.op_reshape_1,
            self.op_add_1,
            self.op_sigmoid_0,
            self.op_multiply_0,
            self.op_pool2d_1,
            self.op_conv2d_2,
            self.op_reshape_2,
            self.op_add_2,
            self.op_relu_1,
            self.op_conv2d_3,
            self.op_reshape_3,
            self.op_add_3,
            self.op_sigmoid_1,
            self.op_multiply_1,
            self.op_pool2d_2,
            self.op_conv2d_4,
            self.op_reshape_4,
            self.op_add_4,
            self.op_relu_2,
            self.op_conv2d_5,
            self.op_reshape_5,
            self.op_add_5,
            self.op_sigmoid_2,
            self.op_multiply_2,
            self.op_pool2d_3,
            self.op_conv2d_6,
            self.op_reshape_6,
            self.op_add_6,
            self.op_relu_3,
            self.op_conv2d_7,
            self.op_reshape_7,
            self.op_add_7,
            self.op_sigmoid_3,
            self.op_multiply_3,
            self.op_full_0,
            self.op_assign_3,
            self.op_assign_4,
            self.op_assign_5,
            self.op_combine_0,
            self.op_concat_0,
            self.op_combine_1,
            self.op_concat_1,
            self.op_combine_2,
            self.op_concat_2,
            self.op_combine_3,
            self.op_concat_3,
            self.op_full_int_array_2,
            self.op_reshape_8,
            self.op_transpose_0,
            self.op_full_int_array_3,
            self.op_reshape_9,
            self.op_full_int_array_4,
            self.op_reshape_10,
            self.op_transpose_1,
            self.op_full_int_array_5,
            self.op_reshape_11,
            self.op_full_int_array_6,
            self.op_reshape_12,
            self.op_transpose_2,
            self.op_full_int_array_7,
            self.op_reshape_13,
            self.op_full_int_array_8,
            self.op_reshape_14,
            self.op_transpose_3,
            self.op_full_int_array_9,
            self.op_reshape_15,
        ]

    def op_full_int_array_0(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
    
        # EarlyReturn(0, 0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        return [parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0]

    def op_assign_0(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0):
    
        # EarlyReturn(0, 1)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        return [parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0]

    def op_assign_1(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0):
    
        # EarlyReturn(0, 2)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        return [parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1]

    def op_assign_2(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1):
    
        # EarlyReturn(0, 3)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        return [parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2]

    def op_pool2d_0(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2):
    
        # EarlyReturn(0, 4)

        # pd_op.pool2d: (1x20x1x1xf32) <- (1x20x128x256xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(data_0, full_int_array_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0]

    def op_conv2d_0(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0):
    
        # EarlyReturn(0, 5)

        # pd_op.conv2d: (1x5x1x1xf32) <- (1x20x1x1xf32, 5x20x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(pool2d_0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_1, parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0]

    def op_full_int_array_1(self, parameter_1, parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0):
    
        # EarlyReturn(0, 6)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        return [parameter_1, parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1]

    def op_reshape_0(self, parameter_1, parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1):
    
        # EarlyReturn(0, 7)

        # pd_op.reshape: (1x5x1x1xf32, 0x5xi64) <- (5xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_1), None

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1]

    def op_add_0(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1):
    
        # EarlyReturn(0, 8)

        # pd_op.add: (1x5x1x1xf32) <- (1x5x1x1xf32, 1x5x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, add_0]

    def op_relu_0(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, add_0):
    
        # EarlyReturn(0, 9)

        # pd_op.relu: (1x5x1x1xf32) <- (1x5x1x1xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0]

    def op_conv2d_1(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0):
    
        # EarlyReturn(0, 10)

        # pd_op.conv2d: (1x20x1x1xf32) <- (1x5x1x1xf32, 20x5x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_2, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1]

    def op_reshape_1(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1):
    
        # EarlyReturn(0, 11)

        # pd_op.reshape: (1x20x1x1xf32, 0x20xi64) <- (20xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(parameter_3, full_int_array_1), None

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3]

    def op_add_1(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3):
    
        # EarlyReturn(0, 12)

        # pd_op.add: (1x20x1x1xf32) <- (1x20x1x1xf32, 1x20x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, add_1]

    def op_sigmoid_0(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, add_1):
    
        # EarlyReturn(0, 13)

        # pd_op.sigmoid: (1x20x1x1xf32) <- (1x20x1x1xf32)
        sigmoid_0 = paddle.nn.functional.sigmoid(add_1)

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0]

    def op_multiply_0(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0):
    
        # EarlyReturn(0, 14)

        # pd_op.multiply: (1x20x128x256xf32) <- (1x20x128x256xf32, 1x20x1x1xf32)
        multiply_0 = data_0 * sigmoid_0

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0]

    def op_pool2d_1(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0):
    
        # EarlyReturn(0, 15)

        # pd_op.pool2d: (1x40x1x1xf32) <- (1x40x64x128xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(data_1, assign_2, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1]

    def op_conv2d_2(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1):
    
        # EarlyReturn(0, 16)

        # pd_op.conv2d: (1x10x1x1xf32) <- (1x40x1x1xf32, 10x40x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_1, parameter_4, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2]

    def op_reshape_2(self, parameter_11, parameter_5, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2):
    
        # EarlyReturn(0, 17)

        # pd_op.reshape: (1x10x1x1xf32, 0x10xi64) <- (10xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_5, full_int_array_1), None

        return [parameter_11, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5]

    def op_add_2(self, parameter_11, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5):
    
        # EarlyReturn(0, 18)

        # pd_op.add: (1x10x1x1xf32) <- (1x10x1x1xf32, 1x10x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        return [parameter_11, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, add_2]

    def op_relu_1(self, parameter_11, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, add_2):
    
        # EarlyReturn(0, 19)

        # pd_op.relu: (1x10x1x1xf32) <- (1x10x1x1xf32)
        relu_1 = paddle._C_ops.relu(add_2)

        return [parameter_11, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1]

    def op_conv2d_3(self, parameter_11, parameter_14, parameter_13, parameter_7, parameter_6, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1):
    
        # EarlyReturn(0, 20)

        # pd_op.conv2d: (1x40x1x1xf32) <- (1x10x1x1xf32, 40x10x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu_1, parameter_6, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_11, parameter_14, parameter_13, parameter_7, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3]

    def op_reshape_3(self, parameter_11, parameter_14, parameter_13, parameter_7, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3):
    
        # EarlyReturn(0, 21)

        # pd_op.reshape: (1x40x1x1xf32, 0x40xi64) <- (40xf32, 4xi64)
        reshape_6, reshape_7 = paddle.reshape(parameter_7, full_int_array_1), None

        return [parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7]

    def op_add_3(self, parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7):
    
        # EarlyReturn(0, 22)

        # pd_op.add: (1x40x1x1xf32) <- (1x40x1x1xf32, 1x40x1x1xf32)
        add_3 = conv2d_3 + reshape_6

        return [parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, add_3]

    def op_sigmoid_1(self, parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, add_3):
    
        # EarlyReturn(0, 23)

        # pd_op.sigmoid: (1x40x1x1xf32) <- (1x40x1x1xf32)
        sigmoid_1 = paddle.nn.functional.sigmoid(add_3)

        return [parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1]

    def op_multiply_1(self, parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_1, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1):
    
        # EarlyReturn(0, 24)

        # pd_op.multiply: (1x40x64x128xf32) <- (1x40x64x128xf32, 1x40x1x1xf32)
        multiply_1 = data_1 * sigmoid_1

        return [parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1]

    def op_pool2d_2(self, parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1):
    
        # EarlyReturn(0, 25)

        # pd_op.pool2d: (1x80x1x1xf32) <- (1x80x32x64xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(data_2, assign_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2]

    def op_conv2d_4(self, parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_8, parameter_12, parameter_9, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2):
    
        # EarlyReturn(0, 26)

        # pd_op.conv2d: (1x20x1x1xf32) <- (1x80x1x1xf32, 20x80x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_2, parameter_8, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_12, parameter_9, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4]

    def op_reshape_4(self, parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_12, parameter_9, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4):
    
        # EarlyReturn(0, 27)

        # pd_op.reshape: (1x20x1x1xf32, 0x20xi64) <- (20xf32, 4xi64)
        reshape_8, reshape_9 = paddle.reshape(parameter_9, full_int_array_1), None

        return [parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9]

    def op_add_4(self, parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9):
    
        # EarlyReturn(0, 28)

        # pd_op.add: (1x20x1x1xf32) <- (1x20x1x1xf32, 1x20x1x1xf32)
        add_4 = conv2d_4 + reshape_8

        return [parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, add_4]

    def op_relu_2(self, parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, add_4):
    
        # EarlyReturn(0, 29)

        # pd_op.relu: (1x20x1x1xf32) <- (1x20x1x1xf32)
        relu_2 = paddle._C_ops.relu(add_4)

        return [parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2]

    def op_conv2d_5(self, parameter_11, parameter_14, parameter_13, parameter_10, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2):
    
        # EarlyReturn(0, 30)

        # pd_op.conv2d: (1x80x1x1xf32) <- (1x20x1x1xf32, 80x20x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu_2, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_11, parameter_14, parameter_13, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5]

    def op_reshape_5(self, parameter_11, parameter_14, parameter_13, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5):
    
        # EarlyReturn(0, 31)

        # pd_op.reshape: (1x80x1x1xf32, 0x80xi64) <- (80xf32, 4xi64)
        reshape_10, reshape_11 = paddle.reshape(parameter_11, full_int_array_1), None

        return [parameter_14, parameter_13, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11]

    def op_add_5(self, parameter_14, parameter_13, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11):
    
        # EarlyReturn(0, 32)

        # pd_op.add: (1x80x1x1xf32) <- (1x80x1x1xf32, 1x80x1x1xf32)
        add_5 = conv2d_5 + reshape_10

        return [parameter_14, parameter_13, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, add_5]

    def op_sigmoid_2(self, parameter_14, parameter_13, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, add_5):
    
        # EarlyReturn(0, 33)

        # pd_op.sigmoid: (1x80x1x1xf32) <- (1x80x1x1xf32)
        sigmoid_2 = paddle.nn.functional.sigmoid(add_5)

        return [parameter_14, parameter_13, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2]

    def op_multiply_2(self, parameter_14, parameter_13, parameter_15, parameter_12, data_2, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2):
    
        # EarlyReturn(0, 34)

        # pd_op.multiply: (1x80x32x64xf32) <- (1x80x32x64xf32, 1x80x1x1xf32)
        multiply_2 = data_2 * sigmoid_2

        return [parameter_14, parameter_13, parameter_15, parameter_12, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2]

    def op_pool2d_3(self, parameter_14, parameter_13, parameter_15, parameter_12, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2):
    
        # EarlyReturn(0, 35)

        # pd_op.pool2d: (1x160x1x1xf32) <- (1x160x16x32xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(data_3, assign_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [parameter_14, parameter_13, parameter_15, parameter_12, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3]

    def op_conv2d_6(self, parameter_14, parameter_13, parameter_15, parameter_12, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3):
    
        # EarlyReturn(0, 36)

        # pd_op.conv2d: (1x40x1x1xf32) <- (1x160x1x1xf32, 40x160x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(pool2d_3, parameter_12, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_14, parameter_13, parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6]

    def op_reshape_6(self, parameter_14, parameter_13, parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6):
    
        # EarlyReturn(0, 37)

        # pd_op.reshape: (1x40x1x1xf32, 0x40xi64) <- (40xf32, 4xi64)
        reshape_12, reshape_13 = paddle.reshape(parameter_13, full_int_array_1), None

        return [parameter_14, parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13]

    def op_add_6(self, parameter_14, parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13):
    
        # EarlyReturn(0, 38)

        # pd_op.add: (1x40x1x1xf32) <- (1x40x1x1xf32, 1x40x1x1xf32)
        add_6 = conv2d_6 + reshape_12

        return [parameter_14, parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, add_6]

    def op_relu_3(self, parameter_14, parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, add_6):
    
        # EarlyReturn(0, 39)

        # pd_op.relu: (1x40x1x1xf32) <- (1x40x1x1xf32)
        relu_3 = paddle._C_ops.relu(add_6)

        return [parameter_14, parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3]

    def op_conv2d_7(self, parameter_14, parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3):
    
        # EarlyReturn(0, 40)

        # pd_op.conv2d: (1x160x1x1xf32) <- (1x40x1x1xf32, 160x40x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu_3, parameter_14, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7]

    def op_reshape_7(self, parameter_15, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, full_int_array_1, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7):
    
        # EarlyReturn(0, 41)

        # pd_op.reshape: (1x160x1x1xf32, 0x160xi64) <- (160xf32, 4xi64)
        reshape_14, reshape_15 = paddle.reshape(parameter_15, full_int_array_1), None

        return [data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15]

    def op_add_7(self, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15):
    
        # EarlyReturn(0, 42)

        # pd_op.add: (1x160x1x1xf32) <- (1x160x1x1xf32, 1x160x1x1xf32)
        add_7 = conv2d_7 + reshape_14

        return [data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, add_7]

    def op_sigmoid_3(self, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, add_7):
    
        # EarlyReturn(0, 43)

        # pd_op.sigmoid: (1x160x1x1xf32) <- (1x160x1x1xf32)
        sigmoid_3 = paddle.nn.functional.sigmoid(add_7)

        return [data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3]

    def op_multiply_3(self, data_3, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3):
    
        # EarlyReturn(0, 44)

        # pd_op.multiply: (1x160x16x32xf32) <- (1x160x16x32xf32, 1x160x1x1xf32)
        multiply_3 = data_3 * sigmoid_3

        return [data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3]

    def op_full_0(self, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3):
    
        # EarlyReturn(0, 45)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        return [data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0]

    def op_assign_3(self, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0):
    
        # EarlyReturn(0, 46)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        return [data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3]

    def op_assign_4(self, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3):
    
        # EarlyReturn(0, 47)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        return [data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4]

    def op_assign_5(self, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4):
    
        # EarlyReturn(0, 48)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

        return [data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5]

    def op_combine_0(self, data_4, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5):
    
        # EarlyReturn(0, 49)

        # builtin.combine: ([1x20x128x256xf32, 1x20x128x256xf32]) <- (1x20x128x256xf32, 1x20x128x256xf32)
        combine_0 = [data_4, multiply_0]

        return [data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, combine_0]

    def op_concat_0(self, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, combine_0):
    
        # EarlyReturn(0, 50)

        # pd_op.concat: (1x40x128x256xf32) <- ([1x20x128x256xf32, 1x20x128x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        return [data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0]

    def op_combine_1(self, data_5, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0):
    
        # EarlyReturn(0, 51)

        # builtin.combine: ([1x40x64x128xf32, 1x40x64x128xf32]) <- (1x40x64x128xf32, 1x40x64x128xf32)
        combine_1 = [data_5, multiply_1]

        return [data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, combine_1]

    def op_concat_1(self, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, combine_1):
    
        # EarlyReturn(0, 52)

        # pd_op.concat: (1x80x64x128xf32) <- ([1x40x64x128xf32, 1x40x64x128xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, assign_5)

        return [data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1]

    def op_combine_2(self, data_6, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1):
    
        # EarlyReturn(0, 53)

        # builtin.combine: ([1x80x32x64xf32, 1x80x32x64xf32]) <- (1x80x32x64xf32, 1x80x32x64xf32)
        combine_2 = [data_6, multiply_2]

        return [data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, combine_2]

    def op_concat_2(self, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, combine_2):
    
        # EarlyReturn(0, 54)

        # pd_op.concat: (1x160x32x64xf32) <- ([1x80x32x64xf32, 1x80x32x64xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, assign_4)

        return [data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, concat_2]

    def op_combine_3(self, data_7, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, concat_2):
    
        # EarlyReturn(0, 55)

        # builtin.combine: ([1x160x16x32xf32, 1x160x16x32xf32]) <- (1x160x16x32xf32, 1x160x16x32xf32)
        combine_3 = [data_7, multiply_3]

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, concat_2, combine_3]

    def op_concat_3(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, concat_2, combine_3):
    
        # EarlyReturn(0, 56)

        # pd_op.concat: (1x320x16x32xf32) <- ([1x160x16x32xf32, 1x160x16x32xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, assign_3)

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, concat_2, concat_3]

    def op_full_int_array_2(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, concat_2, concat_3):
    
        # EarlyReturn(0, 57)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_2 = [1, 2, 20, 128, 256]

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, concat_2, concat_3, full_int_array_2]

    def op_reshape_8(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_0, concat_1, concat_2, concat_3, full_int_array_2):
    
        # EarlyReturn(0, 58)

        # pd_op.reshape: (1x2x20x128x256xf32, 0x1x40x128x256xi64) <- (1x40x128x256xf32, 5xi64)
        reshape_16, reshape_17 = paddle.reshape(concat_0, full_int_array_2), None

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_16, reshape_17]

    def op_transpose_0(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_16, reshape_17):
    
        # EarlyReturn(0, 59)

        # pd_op.transpose: (1x20x2x128x256xf32) <- (1x2x20x128x256xf32)
        transpose_0 = paddle.transpose(reshape_16, perm=[0, 2, 1, 3, 4])

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_17, transpose_0]

    def op_full_int_array_3(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_17, transpose_0):
    
        # EarlyReturn(0, 60)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 40, 128, 256]

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_17, transpose_0, full_int_array_3]

    def op_reshape_9(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_17, transpose_0, full_int_array_3):
    
        # EarlyReturn(0, 61)

        # pd_op.reshape: (1x40x128x256xf32, 0x1x20x2x128x256xi64) <- (1x20x2x128x256xf32, 4xi64)
        reshape_18, reshape_19 = paddle.reshape(transpose_0, full_int_array_3), None

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_17, reshape_18, reshape_19]

    def op_full_int_array_4(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_17, reshape_18, reshape_19):
    
        # EarlyReturn(0, 62)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [1, 2, 40, 64, 128]

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_17, reshape_18, reshape_19, full_int_array_4]

    def op_reshape_10(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_1, concat_2, concat_3, reshape_17, reshape_18, reshape_19, full_int_array_4):
    
        # EarlyReturn(0, 63)

        # pd_op.reshape: (1x2x40x64x128xf32, 0x1x80x64x128xi64) <- (1x80x64x128xf32, 5xi64)
        reshape_20, reshape_21 = paddle.reshape(concat_1, full_int_array_4), None

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_20, reshape_21]

    def op_transpose_1(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_20, reshape_21):
    
        # EarlyReturn(0, 64)

        # pd_op.transpose: (1x40x2x64x128xf32) <- (1x2x40x64x128xf32)
        transpose_1 = paddle.transpose(reshape_20, perm=[0, 2, 1, 3, 4])

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, transpose_1]

    def op_full_int_array_5(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, transpose_1):
    
        # EarlyReturn(0, 65)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 80, 64, 128]

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, transpose_1, full_int_array_5]

    def op_reshape_11(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, transpose_1, full_int_array_5):
    
        # EarlyReturn(0, 66)

        # pd_op.reshape: (1x80x64x128xf32, 0x1x40x2x64x128xi64) <- (1x40x2x64x128xf32, 4xi64)
        reshape_22, reshape_23 = paddle.reshape(transpose_1, full_int_array_5), None

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23]

    def op_full_int_array_6(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23):
    
        # EarlyReturn(0, 67)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_6 = [1, 2, 80, 32, 64]

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, full_int_array_6]

    def op_reshape_12(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_2, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, full_int_array_6):
    
        # EarlyReturn(0, 68)

        # pd_op.reshape: (1x2x80x32x64xf32, 0x1x160x32x64xi64) <- (1x160x32x64xf32, 5xi64)
        reshape_24, reshape_25 = paddle.reshape(concat_2, full_int_array_6), None

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25]

    def op_transpose_2(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25):
    
        # EarlyReturn(0, 69)

        # pd_op.transpose: (1x80x2x32x64xf32) <- (1x2x80x32x64xf32)
        transpose_2 = paddle.transpose(reshape_24, perm=[0, 2, 1, 3, 4])

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, transpose_2]

    def op_full_int_array_7(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, transpose_2):
    
        # EarlyReturn(0, 70)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 160, 32, 64]

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, transpose_2, full_int_array_7]

    def op_reshape_13(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, transpose_2, full_int_array_7):
    
        # EarlyReturn(0, 71)

        # pd_op.reshape: (1x160x32x64xf32, 0x1x80x2x32x64xi64) <- (1x80x2x32x64xf32, 4xi64)
        reshape_26, reshape_27 = paddle.reshape(transpose_2, full_int_array_7), None

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27]

    def op_full_int_array_8(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27):
    
        # EarlyReturn(0, 72)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_8 = [1, 2, 160, 16, 32]

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27, full_int_array_8]

    def op_reshape_14(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, concat_3, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27, full_int_array_8):
    
        # EarlyReturn(0, 73)

        # pd_op.reshape: (1x2x160x16x32xf32, 0x1x320x16x32xi64) <- (1x320x16x32xf32, 5xi64)
        reshape_28, reshape_29 = paddle.reshape(concat_3, full_int_array_8), None

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27, reshape_28, reshape_29]

    def op_transpose_3(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27, reshape_28, reshape_29):
    
        # EarlyReturn(0, 74)

        # pd_op.transpose: (1x160x2x16x32xf32) <- (1x2x160x16x32xf32)
        transpose_3 = paddle.transpose(reshape_28, perm=[0, 2, 1, 3, 4])

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27, reshape_29, transpose_3]

    def op_full_int_array_9(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27, reshape_29, transpose_3):
    
        # EarlyReturn(0, 75)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 320, 16, 32]

        return [full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27, reshape_29, transpose_3, full_int_array_9]

    def op_reshape_15(self, full_int_array_0, assign_0, assign_1, assign_2, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_3, assign_4, assign_5, reshape_17, reshape_18, reshape_19, reshape_21, reshape_22, reshape_23, reshape_25, reshape_26, reshape_27, reshape_29, transpose_3, full_int_array_9):
    
        # EarlyReturn(0, 76)

        # pd_op.reshape: (1x320x16x32xf32, 0x1x160x2x16x32xi64) <- (1x160x2x16x32xf32, 4xi64)
        reshape_30, reshape_31 = paddle.reshape(transpose_3, full_int_array_9), None

        return [full_int_array_0, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, assign_2, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, assign_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, assign_0, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_5, assign_4, assign_3, reshape_17, reshape_19, reshape_21, reshape_23, reshape_25, reshape_27, reshape_29, reshape_31, reshape_18, reshape_22, reshape_26, reshape_30]

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_1
            paddle.uniform([5], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            # data_5
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            # data_6
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            # data_7
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_1
            paddle.static.InputSpec(shape=[5], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[5, 20, 1, 1], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[20, 5, 1, 1], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[40, 10, 1, 1], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[80, 20, 1, 1], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[10, 40, 1, 1], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[20, 80, 1, 1], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
            # data_5
            paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
            # data_6
            paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
            # data_7
            paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = Block_builtin_module_0_0_0()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        paddle.seed(2024)
        out = net(*self.inputs)
        return out

    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()