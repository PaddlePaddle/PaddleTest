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
    return [31][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_0, parameter_4, parameter_1, data_0, data_1):

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x256x-1x-1xf32, 2x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32, 0x2xi64) <- (2xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_0), None

        # pd_op.add: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1x2x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        # pd_op.multiply: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        multiply_0 = add_0 * parameter_2

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 16, paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(multiply_0, full_0, 0, True)

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x256x-1x-1xf32, 2x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(data_0, parameter_3, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x2x1x1xf32, 0x2xi64) <- (2xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(parameter_4, full_int_array_0), None

        # pd_op.add: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1x2x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        # pd_op.multiply: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        multiply_1 = add_1 * parameter_2

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_1, full_1, 1, True)

        # pd_op.elu: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32)
        elu_0 = paddle._C_ops.elu(scale_1, 1)

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(elu_0, assign_0, 0, True)

        # pd_op.conv2d: (-1x1x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(data_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x1x1x1xf32, 0x1xi64) <- (1xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_6, full_int_array_0), None

        # pd_op.add: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, 1x1x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        # pd_op.divide: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1xf32)
        divide_0 = add_2 / data_1

        # pd_op.sign: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        sign_0 = paddle._C_ops.sign(divide_0)

        # pd_op.abs: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        abs_0 = paddle._C_ops.abs(divide_0)

        # pd_op.floor: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        floor_0 = paddle._C_ops.floor(abs_0)

        # pd_op.multiply: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1x1x-1x-1xf32)
        multiply_2 = sign_0 * floor_0

        # pd_op.multiply: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1xf32)
        multiply_3 = multiply_2 * data_1

        # pd_op.subtract: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1x1x-1x-1xf32)
        subtract_0 = add_2 - multiply_3

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x2x-1x-1xf32, -1x2x-1x-1xf32, -1x1x-1x-1xf32]) <- (-1x2x-1x-1xf32, -1x2x-1x-1xf32, -1x1x-1x-1xf32)
        combine_0 = [scale_0, scale_2, subtract_0]

        # pd_op.concat: (-1x5x-1x-1xf32) <- ([-1x2x-1x-1xf32, -1x2x-1x-1xf32, -1x1x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)

        # pd_op.flatten: (-1x5x-1xf32, 0x-1x5x-1x-1xf32) <- (-1x5x-1x-1xf32)
        flatten_0, flatten_1 = paddle._C_ops.flatten(concat_0, 2, 3), None

        # pd_op.transpose: (-1x-1x5xf32) <- (-1x5x-1xf32)
        transpose_0 = paddle.transpose(flatten_0, perm=[0, 2, 1])
        return conv2d_0, reshape_0, reshape_1, add_0, full_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, assign_0, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, full_2, flatten_1, scale_0, scale_2, subtract_0, concat_0, transpose_0



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

    def forward(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_0, parameter_4, parameter_1, data_0, data_1):
        args = [parameter_5, parameter_3, parameter_2, parameter_6, parameter_0, parameter_4, parameter_1, data_0, data_1]
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
            self.op_multiply_0,
            self.op_full_0,
            self.op_assign_0,
            self.op_scale_0,
            self.op_conv2d_1,
            self.op_reshape_1,
            self.op_add_1,
            self.op_multiply_1,
            self.op_full_1,
            self.op_scale_1,
            self.op_elu_0,
            self.op_scale_2,
            self.op_conv2d_2,
            self.op_reshape_2,
            self.op_add_2,
            self.op_divide_0,
            self.op_sign_0,
            self.op_abs_0,
            self.op_floor_0,
            self.op_multiply_2,
            self.op_multiply_3,
            self.op_subtract_0,
            self.op_full_2,
            self.op_combine_0,
            self.op_concat_0,
            self.op_flatten_0,
            self.op_transpose_0,
        ]

    def op_conv2d_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_0, parameter_4, parameter_1, data_0, data_1):
    
        # EarlyReturn(0, 0)

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x256x-1x-1xf32, 2x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, parameter_1, data_0, data_1, conv2d_0]

    def op_full_int_array_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, parameter_1, data_0, data_1, conv2d_0):
    
        # EarlyReturn(0, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        return [parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, parameter_1, data_0, data_1, conv2d_0, full_int_array_0]

    def op_reshape_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, parameter_1, data_0, data_1, conv2d_0, full_int_array_0):
    
        # EarlyReturn(0, 2)

        # pd_op.reshape: (1x2x1x1xf32, 0x2xi64) <- (2xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(parameter_1, full_int_array_0), None

        return [parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1]

    def op_add_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1):
    
        # EarlyReturn(0, 3)

        # pd_op.add: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1x2x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        return [parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0]

    def op_multiply_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0):
    
        # EarlyReturn(0, 4)

        # pd_op.multiply: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        multiply_0 = add_0 * parameter_2

        return [parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, multiply_0]

    def op_full_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, multiply_0):
    
        # EarlyReturn(0, 5)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 16, paddle.float32, paddle.core.CPUPlace())

        return [parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, multiply_0, full_0]

    def op_assign_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, multiply_0, full_0):
    
        # EarlyReturn(0, 6)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        return [parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, multiply_0, full_0, assign_0]

    def op_scale_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, multiply_0, full_0, assign_0):
    
        # EarlyReturn(0, 7)

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(multiply_0, full_0, 0, True)

        return [parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0]

    def op_conv2d_1(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0):
    
        # EarlyReturn(0, 8)

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x256x-1x-1xf32, 2x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(data_0, parameter_3, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_5, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1]

    def op_reshape_1(self, parameter_5, parameter_2, parameter_6, parameter_4, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1):
    
        # EarlyReturn(0, 9)

        # pd_op.reshape: (1x2x1x1xf32, 0x2xi64) <- (2xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(parameter_4, full_int_array_0), None

        return [parameter_5, parameter_2, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3]

    def op_add_1(self, parameter_5, parameter_2, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3):
    
        # EarlyReturn(0, 10)

        # pd_op.add: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1x2x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        return [parameter_5, parameter_2, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1]

    def op_multiply_1(self, parameter_5, parameter_2, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1):
    
        # EarlyReturn(0, 11)

        # pd_op.multiply: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        multiply_1 = add_1 * parameter_2

        return [parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, multiply_1]

    def op_full_1(self, parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, multiply_1):
    
        # EarlyReturn(0, 12)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        return [parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, multiply_1, full_1]

    def op_scale_1(self, parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, multiply_1, full_1):
    
        # EarlyReturn(0, 13)

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_1, full_1, 1, True)

        return [parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1]

    def op_elu_0(self, parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1):
    
        # EarlyReturn(0, 14)

        # pd_op.elu: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32)
        elu_0 = paddle._C_ops.elu(scale_1, 1)

        return [parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0]

    def op_scale_2(self, parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0):
    
        # EarlyReturn(0, 15)

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(elu_0, assign_0, 0, True)

        return [parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2]

    def op_conv2d_2(self, parameter_5, parameter_6, data_0, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2):
    
        # EarlyReturn(0, 16)

        # pd_op.conv2d: (-1x1x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(data_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_6, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2]

    def op_reshape_2(self, parameter_6, data_1, conv2d_0, full_int_array_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2):
    
        # EarlyReturn(0, 17)

        # pd_op.reshape: (1x1x1x1xf32, 0x1xi64) <- (1xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_6, full_int_array_0), None

        return [data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5]

    def op_add_2(self, data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5):
    
        # EarlyReturn(0, 18)

        # pd_op.add: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, 1x1x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        return [data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2]

    def op_divide_0(self, data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2):
    
        # EarlyReturn(0, 19)

        # pd_op.divide: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1xf32)
        divide_0 = add_2 / data_1

        return [data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0]

    def op_sign_0(self, data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0):
    
        # EarlyReturn(0, 20)

        # pd_op.sign: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        sign_0 = paddle._C_ops.sign(divide_0)

        return [data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0]

    def op_abs_0(self, data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0):
    
        # EarlyReturn(0, 21)

        # pd_op.abs: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        abs_0 = paddle._C_ops.abs(divide_0)

        return [data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, abs_0]

    def op_floor_0(self, data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, abs_0):
    
        # EarlyReturn(0, 22)

        # pd_op.floor: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        floor_0 = paddle._C_ops.floor(abs_0)

        return [data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0]

    def op_multiply_2(self, data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0):
    
        # EarlyReturn(0, 23)

        # pd_op.multiply: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1x1x-1x-1xf32)
        multiply_2 = sign_0 * floor_0

        return [data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2]

    def op_multiply_3(self, data_1, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2):
    
        # EarlyReturn(0, 24)

        # pd_op.multiply: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1xf32)
        multiply_3 = multiply_2 * data_1

        return [conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3]

    def op_subtract_0(self, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3):
    
        # EarlyReturn(0, 25)

        # pd_op.subtract: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1x1x-1x-1xf32)
        subtract_0 = add_2 - multiply_3

        return [conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0]

    def op_full_2(self, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0):
    
        # EarlyReturn(0, 26)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        return [conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0, full_2]

    def op_combine_0(self, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0, full_2):
    
        # EarlyReturn(0, 27)

        # builtin.combine: ([-1x2x-1x-1xf32, -1x2x-1x-1xf32, -1x1x-1x-1xf32]) <- (-1x2x-1x-1xf32, -1x2x-1x-1xf32, -1x1x-1x-1xf32)
        combine_0 = [scale_0, scale_2, subtract_0]

        return [conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0, full_2, combine_0]

    def op_concat_0(self, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0, full_2, combine_0):
    
        # EarlyReturn(0, 28)

        # pd_op.concat: (-1x5x-1x-1xf32) <- ([-1x2x-1x-1xf32, -1x2x-1x-1xf32, -1x1x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)

        return [conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0, full_2, concat_0]

    def op_flatten_0(self, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0, full_2, concat_0):
    
        # EarlyReturn(0, 29)

        # pd_op.flatten: (-1x5x-1xf32, 0x-1x5x-1x-1xf32) <- (-1x5x-1x-1xf32)
        flatten_0, flatten_1 = paddle._C_ops.flatten(concat_0, 2, 3), None

        return [conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0, full_2, concat_0, flatten_0, flatten_1]

    def op_transpose_0(self, conv2d_0, reshape_0, reshape_1, add_0, full_0, assign_0, scale_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, scale_2, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, subtract_0, full_2, concat_0, flatten_0, flatten_1):
    
        # EarlyReturn(0, 30)

        # pd_op.transpose: (-1x-1x5xf32) <- (-1x5x-1xf32)
        transpose_0 = paddle.transpose(flatten_0, perm=[0, 2, 1])

        return [conv2d_0, reshape_0, reshape_1, add_0, full_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, assign_0, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, full_2, flatten_1, scale_0, scale_2, subtract_0, concat_0, transpose_0]

if True and not (IsCinnStageEnableDiff() and LastCINNStageFailed()):

    class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
        def prepare_data(self):
            self.inputs = [
                # parameter_5
                paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
                # parameter_3
                paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
                # parameter_2
                paddle.uniform([1], dtype='float32', min=0, max=0.5),
                # parameter_6
                paddle.uniform([1], dtype='float32', min=0, max=0.5),
                # parameter_0
                paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
                # parameter_4
                paddle.uniform([2], dtype='float32', min=0, max=0.5),
                # parameter_1
                paddle.uniform([2], dtype='float32', min=0, max=0.5),
                # data_0
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
                # data_1
                paddle.uniform([1], dtype='float32', min=0, max=0.5),
            ]
            for input in self.inputs:
                input.stop_gradient = True

        def apply_to_static(self, net, use_cinn):
            build_strategy = paddle.static.BuildStrategy()
            input_spec = [
                # parameter_5
                paddle.static.InputSpec(shape=[1, 256, 3, 3], dtype='float32'),
                # parameter_3
                paddle.static.InputSpec(shape=[2, 256, 3, 3], dtype='float32'),
                # parameter_2
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                # parameter_6
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                # parameter_0
                paddle.static.InputSpec(shape=[2, 256, 3, 3], dtype='float32'),
                # parameter_4
                paddle.static.InputSpec(shape=[2], dtype='float32'),
                # parameter_1
                paddle.static.InputSpec(shape=[2], dtype='float32'),
                # data_0
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                # data_1
                paddle.static.InputSpec(shape=[None], dtype='float32'),
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