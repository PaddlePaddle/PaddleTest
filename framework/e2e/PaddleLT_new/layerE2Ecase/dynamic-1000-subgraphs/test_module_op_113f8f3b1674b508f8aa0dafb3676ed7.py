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
    return [63][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, parameter_1, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_0, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5):

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x512x-1x-1xf32, 16x512x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xi64) <- (16xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_0), None

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        # pd_op.transpose: (-1x-1x-1x16xf32) <- (-1x16x-1x-1xf32)
        transpose_0 = paddle.transpose(add_0, perm=[0, 2, 3, 1])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [0, -1, 4]

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x16xi64) <- (-1x-1x-1x16xf32, 3xi64)
        reshape_2, reshape_3 = paddle.reshape(transpose_0, full_int_array_1), None

        # pd_op.conv2d: (-1x84x-1x-1xf32) <- (-1x512x-1x-1xf32, 84x512x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(data_0, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x84x1x1xf32, 0x84xi64) <- (84xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_3, full_int_array_0), None

        # pd_op.add: (-1x84x-1x-1xf32) <- (-1x84x-1x-1xf32, 1x84x1x1xf32)
        add_1 = conv2d_1 + reshape_4

        # pd_op.transpose: (-1x-1x-1x84xf32) <- (-1x84x-1x-1xf32)
        transpose_1 = paddle.transpose(add_1, perm=[0, 2, 3, 1])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, -1, 21]

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x84xi64) <- (-1x-1x-1x84xf32, 3xi64)
        reshape_6, reshape_7 = paddle.reshape(transpose_1, full_int_array_2), None

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x1024x-1x-1xf32, 24x1024x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(data_1, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_8, reshape_9 = paddle.reshape(parameter_5, full_int_array_0), None

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, 1x24x1x1xf32)
        add_2 = conv2d_2 + reshape_8

        # pd_op.transpose: (-1x-1x-1x24xf32) <- (-1x24x-1x-1xf32)
        transpose_2 = paddle.transpose(add_2, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x24xi64) <- (-1x-1x-1x24xf32, 3xi64)
        reshape_10, reshape_11 = paddle.reshape(transpose_2, full_int_array_1), None

        # pd_op.conv2d: (-1x126x-1x-1xf32) <- (-1x1024x-1x-1xf32, 126x1024x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(data_1, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x126x1x1xf32, 0x126xi64) <- (126xf32, 4xi64)
        reshape_12, reshape_13 = paddle.reshape(parameter_7, full_int_array_0), None

        # pd_op.add: (-1x126x-1x-1xf32) <- (-1x126x-1x-1xf32, 1x126x1x1xf32)
        add_3 = conv2d_3 + reshape_12

        # pd_op.transpose: (-1x-1x-1x126xf32) <- (-1x126x-1x-1xf32)
        transpose_3 = paddle.transpose(add_3, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x126xi64) <- (-1x-1x-1x126xf32, 3xi64)
        reshape_14, reshape_15 = paddle.reshape(transpose_3, full_int_array_2), None

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x512x-1x-1xf32, 24x512x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(data_2, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_16, reshape_17 = paddle.reshape(parameter_9, full_int_array_0), None

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, 1x24x1x1xf32)
        add_4 = conv2d_4 + reshape_16

        # pd_op.transpose: (-1x-1x-1x24xf32) <- (-1x24x-1x-1xf32)
        transpose_4 = paddle.transpose(add_4, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x24xi64) <- (-1x-1x-1x24xf32, 3xi64)
        reshape_18, reshape_19 = paddle.reshape(transpose_4, full_int_array_1), None

        # pd_op.conv2d: (-1x126x-1x-1xf32) <- (-1x512x-1x-1xf32, 126x512x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(data_2, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x126x1x1xf32, 0x126xi64) <- (126xf32, 4xi64)
        reshape_20, reshape_21 = paddle.reshape(parameter_11, full_int_array_0), None

        # pd_op.add: (-1x126x-1x-1xf32) <- (-1x126x-1x-1xf32, 1x126x1x1xf32)
        add_5 = conv2d_5 + reshape_20

        # pd_op.transpose: (-1x-1x-1x126xf32) <- (-1x126x-1x-1xf32)
        transpose_5 = paddle.transpose(add_5, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x126xi64) <- (-1x-1x-1x126xf32, 3xi64)
        reshape_22, reshape_23 = paddle.reshape(transpose_5, full_int_array_2), None

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x256x-1x-1xf32, 24x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(data_3, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_24, reshape_25 = paddle.reshape(parameter_13, full_int_array_0), None

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, 1x24x1x1xf32)
        add_6 = conv2d_6 + reshape_24

        # pd_op.transpose: (-1x-1x-1x24xf32) <- (-1x24x-1x-1xf32)
        transpose_6 = paddle.transpose(add_6, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x24xi64) <- (-1x-1x-1x24xf32, 3xi64)
        reshape_26, reshape_27 = paddle.reshape(transpose_6, full_int_array_1), None

        # pd_op.conv2d: (-1x126x-1x-1xf32) <- (-1x256x-1x-1xf32, 126x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(data_3, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x126x1x1xf32, 0x126xi64) <- (126xf32, 4xi64)
        reshape_28, reshape_29 = paddle.reshape(parameter_15, full_int_array_0), None

        # pd_op.add: (-1x126x-1x-1xf32) <- (-1x126x-1x-1xf32, 1x126x1x1xf32)
        add_7 = conv2d_7 + reshape_28

        # pd_op.transpose: (-1x-1x-1x126xf32) <- (-1x126x-1x-1xf32)
        transpose_7 = paddle.transpose(add_7, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x126xi64) <- (-1x-1x-1x126xf32, 3xi64)
        reshape_30, reshape_31 = paddle.reshape(transpose_7, full_int_array_2), None

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x256x-1x-1xf32, 16x256x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(data_4, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x16x1x1xf32, 0x16xi64) <- (16xf32, 4xi64)
        reshape_32, reshape_33 = paddle.reshape(parameter_17, full_int_array_0), None

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_8 = conv2d_8 + reshape_32

        # pd_op.transpose: (-1x-1x-1x16xf32) <- (-1x16x-1x-1xf32)
        transpose_8 = paddle.transpose(add_8, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x16xi64) <- (-1x-1x-1x16xf32, 3xi64)
        reshape_34, reshape_35 = paddle.reshape(transpose_8, full_int_array_1), None

        # pd_op.conv2d: (-1x84x-1x-1xf32) <- (-1x256x-1x-1xf32, 84x256x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(data_4, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x84x1x1xf32, 0x84xi64) <- (84xf32, 4xi64)
        reshape_36, reshape_37 = paddle.reshape(parameter_19, full_int_array_0), None

        # pd_op.add: (-1x84x-1x-1xf32) <- (-1x84x-1x-1xf32, 1x84x1x1xf32)
        add_9 = conv2d_9 + reshape_36

        # pd_op.transpose: (-1x-1x-1x84xf32) <- (-1x84x-1x-1xf32)
        transpose_9 = paddle.transpose(add_9, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x84xi64) <- (-1x-1x-1x84xf32, 3xi64)
        reshape_38, reshape_39 = paddle.reshape(transpose_9, full_int_array_2), None

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x256x-1x-1xf32, 16x256x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(data_5, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x16x1x1xf32, 0x16xi64) <- (16xf32, 4xi64)
        reshape_40, reshape_41 = paddle.reshape(parameter_21, full_int_array_0), None

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_10 = conv2d_10 + reshape_40

        # pd_op.transpose: (-1x-1x-1x16xf32) <- (-1x16x-1x-1xf32)
        transpose_10 = paddle.transpose(add_10, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x16xi64) <- (-1x-1x-1x16xf32, 3xi64)
        reshape_42, reshape_43 = paddle.reshape(transpose_10, full_int_array_1), None

        # pd_op.conv2d: (-1x84x-1x-1xf32) <- (-1x256x-1x-1xf32, 84x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(data_5, parameter_22, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x84x1x1xf32, 0x84xi64) <- (84xf32, 4xi64)
        reshape_44, reshape_45 = paddle.reshape(parameter_23, full_int_array_0), None

        # pd_op.add: (-1x84x-1x-1xf32) <- (-1x84x-1x-1xf32, 1x84x1x1xf32)
        add_11 = conv2d_11 + reshape_44

        # pd_op.transpose: (-1x-1x-1x84xf32) <- (-1x84x-1x-1xf32)
        transpose_11 = paddle.transpose(add_11, perm=[0, 2, 3, 1])

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x84xi64) <- (-1x-1x-1x84xf32, 3xi64)
        reshape_46, reshape_47 = paddle.reshape(transpose_11, full_int_array_2), None
        return conv2d_0, reshape_0, reshape_1, reshape_3, conv2d_1, reshape_4, reshape_5, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_43, conv2d_11, reshape_44, reshape_45, reshape_47, reshape_2, reshape_10, reshape_18, reshape_26, reshape_34, reshape_42, reshape_6, reshape_14, reshape_22, reshape_30, reshape_38, reshape_46



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

    def forward(self, parameter_1, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_0, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5):
        args = [parameter_1, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_0, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_conv2d_0,
            self.op_full_int_array_0,
            self.op_reshape_0,
            self.op_add_0,
            self.op_transpose_0,
            self.op_full_int_array_1,
            self.op_reshape_1,
            self.op_conv2d_1,
            self.op_reshape_2,
            self.op_add_1,
            self.op_transpose_1,
            self.op_full_int_array_2,
            self.op_reshape_3,
            self.op_conv2d_2,
            self.op_reshape_4,
            self.op_add_2,
            self.op_transpose_2,
            self.op_reshape_5,
            self.op_conv2d_3,
            self.op_reshape_6,
            self.op_add_3,
            self.op_transpose_3,
            self.op_reshape_7,
            self.op_conv2d_4,
            self.op_reshape_8,
            self.op_add_4,
            self.op_transpose_4,
            self.op_reshape_9,
            self.op_conv2d_5,
            self.op_reshape_10,
            self.op_add_5,
            self.op_transpose_5,
            self.op_reshape_11,
            self.op_conv2d_6,
            self.op_reshape_12,
            self.op_add_6,
            self.op_transpose_6,
            self.op_reshape_13,
            self.op_conv2d_7,
            self.op_reshape_14,
            self.op_add_7,
            self.op_transpose_7,
            self.op_reshape_15,
            self.op_conv2d_8,
            self.op_reshape_16,
            self.op_add_8,
            self.op_transpose_8,
            self.op_reshape_17,
            self.op_conv2d_9,
            self.op_reshape_18,
            self.op_add_9,
            self.op_transpose_9,
            self.op_reshape_19,
            self.op_conv2d_10,
            self.op_reshape_20,
            self.op_add_10,
            self.op_transpose_10,
            self.op_reshape_21,
            self.op_conv2d_11,
            self.op_reshape_22,
            self.op_add_11,
            self.op_transpose_11,
            self.op_reshape_23,
        ]

    def op_conv2d_0(self, parameter_1, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_0, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5):
    
        # EarlyReturn(0, 0)

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x512x-1x-1xf32, 16x512x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_1, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0]

    def op_full_int_array_0(self, parameter_1, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0):
    
        # EarlyReturn(0, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        return [parameter_1, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0]

    def op_reshape_0(self, parameter_1, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0):
    
        # EarlyReturn(0, 2)

        # pd_op.reshape: (1x16x1x1xf32, 0x16xi64) <- (16xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_0), None

        return [parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1]

    def op_add_0(self, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1):
    
        # EarlyReturn(0, 3)

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        return [parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0]

    def op_transpose_0(self, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0):
    
        # EarlyReturn(0, 4)

        # pd_op.transpose: (-1x-1x-1x16xf32) <- (-1x16x-1x-1xf32)
        transpose_0 = paddle.transpose(add_0, perm=[0, 2, 3, 1])

        return [parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, transpose_0]

    def op_full_int_array_1(self, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, transpose_0):
    
        # EarlyReturn(0, 5)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [0, -1, 4]

        return [parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, transpose_0, full_int_array_1]

    def op_reshape_1(self, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, transpose_0, full_int_array_1):
    
        # EarlyReturn(0, 6)

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x16xi64) <- (-1x-1x-1x16xf32, 3xi64)
        reshape_2, reshape_3 = paddle.reshape(transpose_0, full_int_array_1), None

        return [parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3]

    def op_conv2d_1(self, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_2, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_0, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3):
    
        # EarlyReturn(0, 7)

        # pd_op.conv2d: (-1x84x-1x-1xf32) <- (-1x512x-1x-1xf32, 84x512x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(data_0, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1]

    def op_reshape_2(self, parameter_17, parameter_3, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1):
    
        # EarlyReturn(0, 8)

        # pd_op.reshape: (1x84x1x1xf32, 0x84xi64) <- (84xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_3, full_int_array_0), None

        return [parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5]

    def op_add_1(self, parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5):
    
        # EarlyReturn(0, 9)

        # pd_op.add: (-1x84x-1x-1xf32) <- (-1x84x-1x-1xf32, 1x84x1x1xf32)
        add_1 = conv2d_1 + reshape_4

        return [parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, add_1]

    def op_transpose_1(self, parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, add_1):
    
        # EarlyReturn(0, 10)

        # pd_op.transpose: (-1x-1x-1x84xf32) <- (-1x84x-1x-1xf32)
        transpose_1 = paddle.transpose(add_1, perm=[0, 2, 3, 1])

        return [parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, transpose_1]

    def op_full_int_array_2(self, parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, transpose_1):
    
        # EarlyReturn(0, 11)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [0, -1, 21]

        return [parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, transpose_1, full_int_array_2]

    def op_reshape_3(self, parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, transpose_1, full_int_array_2):
    
        # EarlyReturn(0, 12)

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x84xi64) <- (-1x-1x-1x84xf32, 3xi64)
        reshape_6, reshape_7 = paddle.reshape(transpose_1, full_int_array_2), None

        return [parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7]

    def op_conv2d_2(self, parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_4, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7):
    
        # EarlyReturn(0, 13)

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x1024x-1x-1xf32, 24x1024x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(data_1, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2]

    def op_reshape_4(self, parameter_17, parameter_5, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2):
    
        # EarlyReturn(0, 14)

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_8, reshape_9 = paddle.reshape(parameter_5, full_int_array_0), None

        return [parameter_17, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9]

    def op_add_2(self, parameter_17, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9):
    
        # EarlyReturn(0, 15)

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, 1x24x1x1xf32)
        add_2 = conv2d_2 + reshape_8

        return [parameter_17, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, add_2]

    def op_transpose_2(self, parameter_17, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, add_2):
    
        # EarlyReturn(0, 16)

        # pd_op.transpose: (-1x-1x-1x24xf32) <- (-1x24x-1x-1xf32)
        transpose_2 = paddle.transpose(add_2, perm=[0, 2, 3, 1])

        return [parameter_17, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, transpose_2]

    def op_reshape_5(self, parameter_17, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, transpose_2):
    
        # EarlyReturn(0, 17)

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x24xi64) <- (-1x-1x-1x24xf32, 3xi64)
        reshape_10, reshape_11 = paddle.reshape(transpose_2, full_int_array_1), None

        return [parameter_17, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11]

    def op_conv2d_3(self, parameter_17, parameter_6, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_1, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11):
    
        # EarlyReturn(0, 18)

        # pd_op.conv2d: (-1x126x-1x-1xf32) <- (-1x1024x-1x-1xf32, 126x1024x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(data_1, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3]

    def op_reshape_6(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_7, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3):
    
        # EarlyReturn(0, 19)

        # pd_op.reshape: (1x126x1x1xf32, 0x126xi64) <- (126xf32, 4xi64)
        reshape_12, reshape_13 = paddle.reshape(parameter_7, full_int_array_0), None

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13]

    def op_add_3(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13):
    
        # EarlyReturn(0, 20)

        # pd_op.add: (-1x126x-1x-1xf32) <- (-1x126x-1x-1xf32, 1x126x1x1xf32)
        add_3 = conv2d_3 + reshape_12

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, add_3]

    def op_transpose_3(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, add_3):
    
        # EarlyReturn(0, 21)

        # pd_op.transpose: (-1x-1x-1x126xf32) <- (-1x126x-1x-1xf32)
        transpose_3 = paddle.transpose(add_3, perm=[0, 2, 3, 1])

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, transpose_3]

    def op_reshape_7(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, transpose_3):
    
        # EarlyReturn(0, 22)

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x126xi64) <- (-1x-1x-1x126xf32, 3xi64)
        reshape_14, reshape_15 = paddle.reshape(transpose_3, full_int_array_2), None

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15]

    def op_conv2d_4(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_8, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15):
    
        # EarlyReturn(0, 23)

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x512x-1x-1xf32, 24x512x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(data_2, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4]

    def op_reshape_8(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_9, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4):
    
        # EarlyReturn(0, 24)

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_16, reshape_17 = paddle.reshape(parameter_9, full_int_array_0), None

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17]

    def op_add_4(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17):
    
        # EarlyReturn(0, 25)

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, 1x24x1x1xf32)
        add_4 = conv2d_4 + reshape_16

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, add_4]

    def op_transpose_4(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, add_4):
    
        # EarlyReturn(0, 26)

        # pd_op.transpose: (-1x-1x-1x24xf32) <- (-1x24x-1x-1xf32)
        transpose_4 = paddle.transpose(add_4, perm=[0, 2, 3, 1])

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, transpose_4]

    def op_reshape_9(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, transpose_4):
    
        # EarlyReturn(0, 27)

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x24xi64) <- (-1x-1x-1x24xf32, 3xi64)
        reshape_18, reshape_19 = paddle.reshape(transpose_4, full_int_array_1), None

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19]

    def op_conv2d_5(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, parameter_10, data_2, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19):
    
        # EarlyReturn(0, 28)

        # pd_op.conv2d: (-1x126x-1x-1xf32) <- (-1x512x-1x-1xf32, 126x512x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(data_2, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5]

    def op_reshape_10(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, parameter_11, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5):
    
        # EarlyReturn(0, 29)

        # pd_op.reshape: (1x126x1x1xf32, 0x126xi64) <- (126xf32, 4xi64)
        reshape_20, reshape_21 = paddle.reshape(parameter_11, full_int_array_0), None

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21]

    def op_add_5(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21):
    
        # EarlyReturn(0, 30)

        # pd_op.add: (-1x126x-1x-1xf32) <- (-1x126x-1x-1xf32, 1x126x1x1xf32)
        add_5 = conv2d_5 + reshape_20

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, add_5]

    def op_transpose_5(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, add_5):
    
        # EarlyReturn(0, 31)

        # pd_op.transpose: (-1x-1x-1x126xf32) <- (-1x126x-1x-1xf32)
        transpose_5 = paddle.transpose(add_5, perm=[0, 2, 3, 1])

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, transpose_5]

    def op_reshape_11(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, transpose_5):
    
        # EarlyReturn(0, 32)

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x126xi64) <- (-1x-1x-1x126xf32, 3xi64)
        reshape_22, reshape_23 = paddle.reshape(transpose_5, full_int_array_2), None

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23]

    def op_conv2d_6(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_12, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23):
    
        # EarlyReturn(0, 33)

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x256x-1x-1xf32, 24x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(data_3, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6]

    def op_reshape_12(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_13, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6):
    
        # EarlyReturn(0, 34)

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_24, reshape_25 = paddle.reshape(parameter_13, full_int_array_0), None

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25]

    def op_add_6(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25):
    
        # EarlyReturn(0, 35)

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, 1x24x1x1xf32)
        add_6 = conv2d_6 + reshape_24

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, add_6]

    def op_transpose_6(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, add_6):
    
        # EarlyReturn(0, 36)

        # pd_op.transpose: (-1x-1x-1x24xf32) <- (-1x24x-1x-1xf32)
        transpose_6 = paddle.transpose(add_6, perm=[0, 2, 3, 1])

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, transpose_6]

    def op_reshape_13(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, transpose_6):
    
        # EarlyReturn(0, 37)

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x24xi64) <- (-1x-1x-1x24xf32, 3xi64)
        reshape_26, reshape_27 = paddle.reshape(transpose_6, full_int_array_1), None

        return [parameter_17, parameter_21, parameter_14, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27]

    def op_conv2d_7(self, parameter_17, parameter_21, parameter_14, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_3, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27):
    
        # EarlyReturn(0, 38)

        # pd_op.conv2d: (-1x126x-1x-1xf32) <- (-1x256x-1x-1xf32, 126x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(data_3, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_17, parameter_21, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7]

    def op_reshape_14(self, parameter_17, parameter_21, parameter_15, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7):
    
        # EarlyReturn(0, 39)

        # pd_op.reshape: (1x126x1x1xf32, 0x126xi64) <- (126xf32, 4xi64)
        reshape_28, reshape_29 = paddle.reshape(parameter_15, full_int_array_0), None

        return [parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29]

    def op_add_7(self, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29):
    
        # EarlyReturn(0, 40)

        # pd_op.add: (-1x126x-1x-1xf32) <- (-1x126x-1x-1xf32, 1x126x1x1xf32)
        add_7 = conv2d_7 + reshape_28

        return [parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, add_7]

    def op_transpose_7(self, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, add_7):
    
        # EarlyReturn(0, 41)

        # pd_op.transpose: (-1x-1x-1x126xf32) <- (-1x126x-1x-1xf32)
        transpose_7 = paddle.transpose(add_7, perm=[0, 2, 3, 1])

        return [parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, transpose_7]

    def op_reshape_15(self, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, transpose_7):
    
        # EarlyReturn(0, 42)

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x126xi64) <- (-1x-1x-1x126xf32, 3xi64)
        reshape_30, reshape_31 = paddle.reshape(transpose_7, full_int_array_2), None

        return [parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31]

    def op_conv2d_8(self, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_16, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31):
    
        # EarlyReturn(0, 43)

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x256x-1x-1xf32, 16x256x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(data_4, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8]

    def op_reshape_16(self, parameter_17, parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8):
    
        # EarlyReturn(0, 44)

        # pd_op.reshape: (1x16x1x1xf32, 0x16xi64) <- (16xf32, 4xi64)
        reshape_32, reshape_33 = paddle.reshape(parameter_17, full_int_array_0), None

        return [parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33]

    def op_add_8(self, parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33):
    
        # EarlyReturn(0, 45)

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_8 = conv2d_8 + reshape_32

        return [parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, add_8]

    def op_transpose_8(self, parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, add_8):
    
        # EarlyReturn(0, 46)

        # pd_op.transpose: (-1x-1x-1x16xf32) <- (-1x16x-1x-1xf32)
        transpose_8 = paddle.transpose(add_8, perm=[0, 2, 3, 1])

        return [parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, transpose_8]

    def op_reshape_17(self, parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, transpose_8):
    
        # EarlyReturn(0, 47)

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x16xi64) <- (-1x-1x-1x16xf32, 3xi64)
        reshape_34, reshape_35 = paddle.reshape(transpose_8, full_int_array_1), None

        return [parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35]

    def op_conv2d_9(self, parameter_21, parameter_18, parameter_20, parameter_19, parameter_23, parameter_22, data_4, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35):
    
        # EarlyReturn(0, 48)

        # pd_op.conv2d: (-1x84x-1x-1xf32) <- (-1x256x-1x-1xf32, 84x256x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(data_4, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_21, parameter_20, parameter_19, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9]

    def op_reshape_18(self, parameter_21, parameter_20, parameter_19, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9):
    
        # EarlyReturn(0, 49)

        # pd_op.reshape: (1x84x1x1xf32, 0x84xi64) <- (84xf32, 4xi64)
        reshape_36, reshape_37 = paddle.reshape(parameter_19, full_int_array_0), None

        return [parameter_21, parameter_20, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37]

    def op_add_9(self, parameter_21, parameter_20, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37):
    
        # EarlyReturn(0, 50)

        # pd_op.add: (-1x84x-1x-1xf32) <- (-1x84x-1x-1xf32, 1x84x1x1xf32)
        add_9 = conv2d_9 + reshape_36

        return [parameter_21, parameter_20, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, add_9]

    def op_transpose_9(self, parameter_21, parameter_20, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, add_9):
    
        # EarlyReturn(0, 51)

        # pd_op.transpose: (-1x-1x-1x84xf32) <- (-1x84x-1x-1xf32)
        transpose_9 = paddle.transpose(add_9, perm=[0, 2, 3, 1])

        return [parameter_21, parameter_20, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, transpose_9]

    def op_reshape_19(self, parameter_21, parameter_20, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, transpose_9):
    
        # EarlyReturn(0, 52)

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x84xi64) <- (-1x-1x-1x84xf32, 3xi64)
        reshape_38, reshape_39 = paddle.reshape(transpose_9, full_int_array_2), None

        return [parameter_21, parameter_20, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39]

    def op_conv2d_10(self, parameter_21, parameter_20, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39):
    
        # EarlyReturn(0, 53)

        # pd_op.conv2d: (-1x16x-1x-1xf32) <- (-1x256x-1x-1xf32, 16x256x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(data_5, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_21, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10]

    def op_reshape_20(self, parameter_21, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10):
    
        # EarlyReturn(0, 54)

        # pd_op.reshape: (1x16x1x1xf32, 0x16xi64) <- (16xf32, 4xi64)
        reshape_40, reshape_41 = paddle.reshape(parameter_21, full_int_array_0), None

        return [parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41]

    def op_add_10(self, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41):
    
        # EarlyReturn(0, 55)

        # pd_op.add: (-1x16x-1x-1xf32) <- (-1x16x-1x-1xf32, 1x16x1x1xf32)
        add_10 = conv2d_10 + reshape_40

        return [parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, add_10]

    def op_transpose_10(self, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, add_10):
    
        # EarlyReturn(0, 56)

        # pd_op.transpose: (-1x-1x-1x16xf32) <- (-1x16x-1x-1xf32)
        transpose_10 = paddle.transpose(add_10, perm=[0, 2, 3, 1])

        return [parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, transpose_10]

    def op_reshape_21(self, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, full_int_array_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, transpose_10):
    
        # EarlyReturn(0, 57)

        # pd_op.reshape: (-1x-1x4xf32, 0x-1x-1x-1x16xi64) <- (-1x-1x-1x16xf32, 3xi64)
        reshape_42, reshape_43 = paddle.reshape(transpose_10, full_int_array_1), None

        return [parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43]

    def op_conv2d_11(self, parameter_23, parameter_22, data_5, conv2d_0, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43):
    
        # EarlyReturn(0, 58)

        # pd_op.conv2d: (-1x84x-1x-1xf32) <- (-1x256x-1x-1xf32, 84x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(data_5, parameter_22, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_23, conv2d_0, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43, conv2d_11]

    def op_reshape_22(self, parameter_23, conv2d_0, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43, conv2d_11):
    
        # EarlyReturn(0, 59)

        # pd_op.reshape: (1x84x1x1xf32, 0x84xi64) <- (84xf32, 4xi64)
        reshape_44, reshape_45 = paddle.reshape(parameter_23, full_int_array_0), None

        return [conv2d_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43, conv2d_11, reshape_44, reshape_45]

    def op_add_11(self, conv2d_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43, conv2d_11, reshape_44, reshape_45):
    
        # EarlyReturn(0, 60)

        # pd_op.add: (-1x84x-1x-1xf32) <- (-1x84x-1x-1xf32, 1x84x1x1xf32)
        add_11 = conv2d_11 + reshape_44

        return [conv2d_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43, conv2d_11, reshape_44, reshape_45, add_11]

    def op_transpose_11(self, conv2d_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43, conv2d_11, reshape_44, reshape_45, add_11):
    
        # EarlyReturn(0, 61)

        # pd_op.transpose: (-1x-1x-1x84xf32) <- (-1x84x-1x-1xf32)
        transpose_11 = paddle.transpose(add_11, perm=[0, 2, 3, 1])

        return [conv2d_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43, conv2d_11, reshape_44, reshape_45, transpose_11]

    def op_reshape_23(self, conv2d_0, reshape_0, reshape_1, reshape_2, reshape_3, conv2d_1, reshape_4, reshape_5, full_int_array_2, reshape_6, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_10, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_14, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_18, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_22, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_26, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_30, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_34, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_38, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_42, reshape_43, conv2d_11, reshape_44, reshape_45, transpose_11):
    
        # EarlyReturn(0, 62)

        # pd_op.reshape: (-1x-1x21xf32, 0x-1x-1x-1x84xi64) <- (-1x-1x-1x84xf32, 3xi64)
        reshape_46, reshape_47 = paddle.reshape(transpose_11, full_int_array_2), None

        return [conv2d_0, reshape_0, reshape_1, reshape_3, conv2d_1, reshape_4, reshape_5, reshape_7, conv2d_2, reshape_8, reshape_9, reshape_11, conv2d_3, reshape_12, reshape_13, reshape_15, conv2d_4, reshape_16, reshape_17, reshape_19, conv2d_5, reshape_20, reshape_21, reshape_23, conv2d_6, reshape_24, reshape_25, reshape_27, conv2d_7, reshape_28, reshape_29, reshape_31, conv2d_8, reshape_32, reshape_33, reshape_35, conv2d_9, reshape_36, reshape_37, reshape_39, conv2d_10, reshape_40, reshape_41, reshape_43, conv2d_11, reshape_44, reshape_45, reshape_47, reshape_2, reshape_10, reshape_18, reshape_26, reshape_34, reshape_42, reshape_6, reshape_14, reshape_22, reshape_30, reshape_38, reshape_46]

is_module_block_and_last_stage_passed = (
    True and not last_stage_failed
)
@unittest.skipIf(not is_module_block_and_last_stage_passed, "last stage failed")
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_1
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([126, 1024, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([126, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([126], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([84, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([24, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([24, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([84, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([126], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([16, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([24, 1024, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([16, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([84], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([84, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([16, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([126], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([126, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # data_5
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_1
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[84], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[126, 1024, 3, 3], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[126, 256, 3, 3], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[126], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[84, 512, 3, 3], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[24, 256, 3, 3], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[24, 512, 3, 3], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[84, 256, 3, 3], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[126], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[16, 256, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[24, 1024, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[84], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[16, 256, 3, 3], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[84], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[84, 256, 3, 3], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[16, 512, 3, 3], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[126], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[126, 512, 3, 3], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            # data_5
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
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