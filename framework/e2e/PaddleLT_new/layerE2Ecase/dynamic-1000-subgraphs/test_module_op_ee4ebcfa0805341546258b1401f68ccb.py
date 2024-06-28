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
    return [44][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_2, parameter_0, parameter_5, parameter_1, parameter_4, parameter_3, data_0, data_1, data_2):

        # pd_op.layer_norm: (1x32768x64xf32, 1x32768xf32, 1x32768xf32) <- (1x32768x64xf32, 64xf32, 64xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = paddle._C_ops.layer_norm(data_0, parameter_0, parameter_1, 1e-06, 2), None, None

        # pd_op.matmul: (1x32768x64xf32) <- (1x32768x64xf32, 64x64xf32)
        matmul_0 = paddle.matmul(layer_norm_0, parameter_2, transpose_x=False, transpose_y=False)

        # pd_op.add: (1x32768x64xf32) <- (1x32768x64xf32, 64xf32)
        add_0 = matmul_0 + parameter_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 32768, 1, 64]

        # pd_op.reshape: (1x32768x1x64xf32, 0x1x32768x64xi64) <- (1x32768x64xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(add_0, full_int_array_0), None

        # pd_op.transpose: (1x1x32768x64xf32) <- (1x32768x1x64xf32)
        transpose_0 = paddle.transpose(reshape_0, perm=[0, 2, 1, 3])

        # pd_op.transpose: (1x64x32768xf32) <- (1x32768x64xf32)
        transpose_1 = paddle.transpose(layer_norm_0, perm=[0, 2, 1])

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full([], 1, paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full([], 64, paddle.int64, paddle.core.CPUPlace())

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(data_1, paddle.int64)

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(data_2, paddle.int64)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, full_1, cast_0, cast_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (1x64x-1x-1xf32, 0x1x64x32768xi64) <- (1x64x32768xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(transpose_1, stack_0), None

        # pd_op.conv2d: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, 64x64x8x8xf32)
        conv2d_0 = paddle._C_ops.conv2d(reshape_2, parameter_4, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xi64) <- (64xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_5, full_int_array_1), None

        # pd_op.add: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, 1x64x1x1xf32)
        add_1 = conv2d_0 + reshape_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, 64, -1]

        # pd_op.reshape: (1x64x-1xf32, 0x1x64x-1x-1xi64) <- (1x64x-1x-1xf32, 3xi64)
        reshape_6, reshape_7 = paddle.reshape(add_1, full_int_array_2), None

        # pd_op.transpose: (1x-1x64xf32) <- (1x64x-1xf32)
        transpose_2 = paddle.transpose(reshape_6, perm=[0, 2, 1])

        # pd_op.layer_norm: (1x-1x64xf32, 1x-1xf32, 1x-1xf32) <- (1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = paddle._C_ops.layer_norm(transpose_2, parameter_6, parameter_7, 1e-05, 2), None, None

        # pd_op.matmul: (1x-1x128xf32) <- (1x-1x64xf32, 64x128xf32)
        matmul_1 = paddle.matmul(layer_norm_3, parameter_8, transpose_x=False, transpose_y=False)

        # pd_op.add: (1x-1x128xf32) <- (1x-1x128xf32, 128xf32)
        add_2 = matmul_1 + parameter_9

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_3 = [1, -1, 2, 1, 64]

        # pd_op.reshape: (1x-1x2x1x64xf32, 0x1x-1x128xi64) <- (1x-1x128xf32, 5xi64)
        reshape_8, reshape_9 = paddle.reshape(add_2, full_int_array_3), None

        # pd_op.transpose: (2x1x1x-1x64xf32) <- (1x-1x2x1x64xf32)
        transpose_3 = paddle.transpose(reshape_8, perm=[2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_5

        # pd_op.slice: (1x1x-1x64xf32) <- (2x1x1x-1x64xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(transpose_3, [0], full_int_array_4, full_int_array_5, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.slice: (1x1x-1x64xf32) <- (2x1x1x-1x64xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(transpose_3, [0], assign_0, full_int_array_6, [1], [0])

        # pd_op.transpose: (1x1x64x-1xf32) <- (1x1x-1x64xf32)
        transpose_4 = paddle.transpose(slice_0, perm=[0, 1, 3, 2])

        # pd_op.matmul: (1x1x32768x-1xf32) <- (1x1x32768x64xf32, 1x1x64x-1xf32)
        matmul_2 = paddle.matmul(transpose_0, transpose_4, transpose_x=False, transpose_y=False)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 0.125, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (1x1x32768x-1xf32) <- (1x1x32768x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_2, full_2, 0, True)

        # pd_op.softmax: (1x1x32768x-1xf32) <- (1x1x32768x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)

        # pd_op.matmul: (1x1x32768x64xf32) <- (1x1x32768x-1xf32, 1x1x-1x64xf32)
        matmul_3 = paddle.matmul(softmax_0, slice_1, transpose_x=False, transpose_y=False)

        # pd_op.transpose: (1x32768x1x64xf32) <- (1x1x32768x64xf32)
        transpose_5 = paddle.transpose(matmul_3, perm=[0, 2, 1, 3])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [1, 32768, 64]

        # pd_op.reshape: (1x32768x64xf32, 0x1x32768x1x64xi64) <- (1x32768x1x64xf32, 3xi64)
        reshape_10, reshape_11 = paddle.reshape(transpose_5, full_int_array_7), None

        # pd_op.matmul: (1x32768x64xf32) <- (1x32768x64xf32, 64x64xf32)
        matmul_4 = paddle.matmul(reshape_10, parameter_10, transpose_x=False, transpose_y=False)

        # pd_op.add: (1x32768x64xf32) <- (1x32768x64xf32, 64xf32)
        add_3 = matmul_4 + parameter_11
        return layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, reshape_10, reshape_11, matmul_4, add_3



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

    def forward(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_2, parameter_0, parameter_5, parameter_1, parameter_4, parameter_3, data_0, data_1, data_2):
        args = [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_2, parameter_0, parameter_5, parameter_1, parameter_4, parameter_3, data_0, data_1, data_2]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_layer_norm_0,
            self.op_matmul_0,
            self.op_add_0,
            self.op_full_int_array_0,
            self.op_reshape_0,
            self.op_transpose_0,
            self.op_transpose_1,
            self.op_full_0,
            self.op_full_1,
            self.op_cast_0,
            self.op_cast_1,
            self.op_combine_0,
            self.op_stack_0,
            self.op_reshape_1,
            self.op_conv2d_0,
            self.op_full_int_array_1,
            self.op_reshape_2,
            self.op_add_1,
            self.op_full_int_array_2,
            self.op_reshape_3,
            self.op_transpose_2,
            self.op_layer_norm_1,
            self.op_matmul_1,
            self.op_add_2,
            self.op_full_int_array_3,
            self.op_reshape_4,
            self.op_transpose_3,
            self.op_full_int_array_4,
            self.op_full_int_array_5,
            self.op_assign_0,
            self.op_slice_0,
            self.op_full_int_array_6,
            self.op_slice_1,
            self.op_transpose_4,
            self.op_matmul_2,
            self.op_full_2,
            self.op_scale_0,
            self.op_softmax_0,
            self.op_matmul_3,
            self.op_transpose_5,
            self.op_full_int_array_7,
            self.op_reshape_5,
            self.op_matmul_4,
            self.op_add_3,
        ]

    def op_layer_norm_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_2, parameter_0, parameter_5, parameter_1, parameter_4, parameter_3, data_0, data_1, data_2):
    
        # EarlyReturn(0, 0)

        # pd_op.layer_norm: (1x32768x64xf32, 1x32768xf32, 1x32768xf32) <- (1x32768x64xf32, 64xf32, 64xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = paddle._C_ops.layer_norm(data_0, parameter_0, parameter_1, 1e-06, 2), None, None

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_2, parameter_5, parameter_4, parameter_3, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2]

    def op_matmul_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_2, parameter_5, parameter_4, parameter_3, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2):
    
        # EarlyReturn(0, 1)

        # pd_op.matmul: (1x32768x64xf32) <- (1x32768x64xf32, 64x64xf32)
        matmul_0 = paddle.matmul(layer_norm_0, parameter_2, transpose_x=False, transpose_y=False)

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, parameter_3, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0]

    def op_add_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, parameter_3, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0):
    
        # EarlyReturn(0, 2)

        # pd_op.add: (1x32768x64xf32) <- (1x32768x64xf32, 64xf32)
        add_0 = matmul_0 + parameter_3

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, add_0]

    def op_full_int_array_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, add_0):
    
        # EarlyReturn(0, 3)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 32768, 1, 64]

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, add_0, full_int_array_0]

    def op_reshape_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, add_0, full_int_array_0):
    
        # EarlyReturn(0, 4)

        # pd_op.reshape: (1x32768x1x64xf32, 0x1x32768x64xi64) <- (1x32768x64xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(add_0, full_int_array_0), None

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_0, reshape_1]

    def op_transpose_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_0, reshape_1):
    
        # EarlyReturn(0, 5)

        # pd_op.transpose: (1x1x32768x64xf32) <- (1x32768x1x64xf32)
        transpose_0 = paddle.transpose(reshape_0, perm=[0, 2, 1, 3])

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0]

    def op_transpose_1(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0):
    
        # EarlyReturn(0, 6)

        # pd_op.transpose: (1x64x32768xf32) <- (1x32768x64xf32)
        transpose_1 = paddle.transpose(layer_norm_0, perm=[0, 2, 1])

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1]

    def op_full_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1):
    
        # EarlyReturn(0, 7)

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full([], 1, paddle.int64, paddle.core.CPUPlace())

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, full_0]

    def op_full_1(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, full_0):
    
        # EarlyReturn(0, 8)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full([], 64, paddle.int64, paddle.core.CPUPlace())

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, full_0, full_1]

    def op_cast_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_1, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, full_0, full_1):
    
        # EarlyReturn(0, 9)

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(data_1, paddle.int64)

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, full_0, full_1, cast_0]

    def op_cast_1(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, data_2, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, full_0, full_1, cast_0):
    
        # EarlyReturn(0, 10)

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(data_2, paddle.int64)

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, full_0, full_1, cast_0, cast_1]

    def op_combine_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, full_0, full_1, cast_0, cast_1):
    
        # EarlyReturn(0, 11)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, full_1, cast_0, cast_1]

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, combine_0]

    def op_stack_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, combine_0):
    
        # EarlyReturn(0, 12)

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, stack_0]

    def op_reshape_1(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, transpose_1, stack_0):
    
        # EarlyReturn(0, 13)

        # pd_op.reshape: (1x64x-1x-1xf32, 0x1x64x32768xi64) <- (1x64x32768xf32, 4xi64)
        reshape_2, reshape_3 = paddle.reshape(transpose_1, stack_0), None

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3]

    def op_conv2d_0(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, parameter_4, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3):
    
        # EarlyReturn(0, 14)

        # pd_op.conv2d: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, 64x64x8x8xf32)
        conv2d_0 = paddle._C_ops.conv2d(reshape_2, parameter_4, [8, 8], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0]

    def op_full_int_array_1(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0):
    
        # EarlyReturn(0, 15)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, full_int_array_1]

    def op_reshape_2(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, parameter_5, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, full_int_array_1):
    
        # EarlyReturn(0, 16)

        # pd_op.reshape: (1x64x1x1xf32, 0x64xi64) <- (64xf32, 4xi64)
        reshape_4, reshape_5 = paddle.reshape(parameter_5, full_int_array_1), None

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5]

    def op_add_1(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5):
    
        # EarlyReturn(0, 17)

        # pd_op.add: (1x64x-1x-1xf32) <- (1x64x-1x-1xf32, 1x64x1x1xf32)
        add_1 = conv2d_0 + reshape_4

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, add_1]

    def op_full_int_array_2(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, add_1):
    
        # EarlyReturn(0, 18)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, 64, -1]

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, add_1, full_int_array_2]

    def op_reshape_3(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, add_1, full_int_array_2):
    
        # EarlyReturn(0, 19)

        # pd_op.reshape: (1x64x-1xf32, 0x1x64x-1x-1xi64) <- (1x64x-1x-1xf32, 3xi64)
        reshape_6, reshape_7 = paddle.reshape(add_1, full_int_array_2), None

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_6, reshape_7]

    def op_transpose_2(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_6, reshape_7):
    
        # EarlyReturn(0, 20)

        # pd_op.transpose: (1x-1x64xf32) <- (1x64x-1xf32)
        transpose_2 = paddle.transpose(reshape_6, perm=[0, 2, 1])

        return [parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2]

    def op_layer_norm_1(self, parameter_11, parameter_8, parameter_9, parameter_7, parameter_6, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2):
    
        # EarlyReturn(0, 21)

        # pd_op.layer_norm: (1x-1x64xf32, 1x-1xf32, 1x-1xf32) <- (1x-1x64xf32, 64xf32, 64xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = paddle._C_ops.layer_norm(transpose_2, parameter_6, parameter_7, 1e-05, 2), None, None

        return [parameter_11, parameter_8, parameter_9, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5]

    def op_matmul_1(self, parameter_11, parameter_8, parameter_9, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5):
    
        # EarlyReturn(0, 22)

        # pd_op.matmul: (1x-1x128xf32) <- (1x-1x64xf32, 64x128xf32)
        matmul_1 = paddle.matmul(layer_norm_3, parameter_8, transpose_x=False, transpose_y=False)

        return [parameter_11, parameter_9, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1]

    def op_add_2(self, parameter_11, parameter_9, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1):
    
        # EarlyReturn(0, 23)

        # pd_op.add: (1x-1x128xf32) <- (1x-1x128xf32, 128xf32)
        add_2 = matmul_1 + parameter_9

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, add_2]

    def op_full_int_array_3(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, add_2):
    
        # EarlyReturn(0, 24)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_3 = [1, -1, 2, 1, 64]

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, add_2, full_int_array_3]

    def op_reshape_4(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, add_2, full_int_array_3):
    
        # EarlyReturn(0, 25)

        # pd_op.reshape: (1x-1x2x1x64xf32, 0x1x-1x128xi64) <- (1x-1x128xf32, 5xi64)
        reshape_8, reshape_9 = paddle.reshape(add_2, full_int_array_3), None

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_8, reshape_9]

    def op_transpose_3(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_8, reshape_9):
    
        # EarlyReturn(0, 26)

        # pd_op.transpose: (2x1x1x-1x64xf32) <- (1x-1x2x1x64xf32)
        transpose_3 = paddle.transpose(reshape_8, perm=[2, 0, 3, 1, 4])

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3]

    def op_full_int_array_4(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3):
    
        # EarlyReturn(0, 27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4]

    def op_full_int_array_5(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4):
    
        # EarlyReturn(0, 28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5]

    def op_assign_0(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5):
    
        # EarlyReturn(0, 29)

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_5

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0]

    def op_slice_0(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0):
    
        # EarlyReturn(0, 30)

        # pd_op.slice: (1x1x-1x64xf32) <- (2x1x1x-1x64xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(transpose_3, [0], full_int_array_4, full_int_array_5, [1], [0])

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, slice_0]

    def op_full_int_array_6(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, slice_0):
    
        # EarlyReturn(0, 31)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, slice_0, full_int_array_6]

    def op_slice_1(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, slice_0, full_int_array_6):
    
        # EarlyReturn(0, 32)

        # pd_op.slice: (1x1x-1x64xf32) <- (2x1x1x-1x64xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(transpose_3, [0], assign_0, full_int_array_6, [1], [0])

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, slice_0, full_int_array_6, slice_1]

    def op_transpose_4(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, slice_0, full_int_array_6, slice_1):
    
        # EarlyReturn(0, 33)

        # pd_op.transpose: (1x1x64x-1xf32) <- (1x1x-1x64xf32)
        transpose_4 = paddle.transpose(slice_0, perm=[0, 1, 3, 2])

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4]

    def op_matmul_2(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4):
    
        # EarlyReturn(0, 34)

        # pd_op.matmul: (1x1x32768x-1xf32) <- (1x1x32768x64xf32, 1x1x64x-1xf32)
        matmul_2 = paddle.matmul(transpose_0, transpose_4, transpose_x=False, transpose_y=False)

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, matmul_2]

    def op_full_2(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, matmul_2):
    
        # EarlyReturn(0, 35)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 0.125, paddle.float32, paddle.core.CPUPlace())

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, matmul_2, full_2]

    def op_scale_0(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, matmul_2, full_2):
    
        # EarlyReturn(0, 36)

        # pd_op.scale: (1x1x32768x-1xf32) <- (1x1x32768x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(matmul_2, full_2, 0, True)

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, scale_0]

    def op_softmax_0(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, scale_0):
    
        # EarlyReturn(0, 37)

        # pd_op.softmax: (1x1x32768x-1xf32) <- (1x1x32768x-1xf32)
        softmax_0 = paddle._C_ops.softmax(scale_0, -1)

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0]

    def op_matmul_3(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0):
    
        # EarlyReturn(0, 38)

        # pd_op.matmul: (1x1x32768x64xf32) <- (1x1x32768x-1xf32, 1x1x-1x64xf32)
        matmul_3 = paddle.matmul(softmax_0, slice_1, transpose_x=False, transpose_y=False)

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, matmul_3]

    def op_transpose_5(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, matmul_3):
    
        # EarlyReturn(0, 39)

        # pd_op.transpose: (1x32768x1x64xf32) <- (1x1x32768x64xf32)
        transpose_5 = paddle.transpose(matmul_3, perm=[0, 2, 1, 3])

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, transpose_5]

    def op_full_int_array_7(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, transpose_5):
    
        # EarlyReturn(0, 40)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_7 = [1, 32768, 64]

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, transpose_5, full_int_array_7]

    def op_reshape_5(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, transpose_5, full_int_array_7):
    
        # EarlyReturn(0, 41)

        # pd_op.reshape: (1x32768x64xf32, 0x1x32768x1x64xi64) <- (1x32768x1x64xf32, 3xi64)
        reshape_10, reshape_11 = paddle.reshape(transpose_5, full_int_array_7), None

        return [parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, reshape_10, reshape_11]

    def op_matmul_4(self, parameter_11, parameter_10, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, reshape_10, reshape_11):
    
        # EarlyReturn(0, 42)

        # pd_op.matmul: (1x32768x64xf32) <- (1x32768x64xf32, 64x64xf32)
        matmul_4 = paddle.matmul(reshape_10, parameter_10, transpose_x=False, transpose_y=False)

        return [parameter_11, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, reshape_10, reshape_11, matmul_4]

    def op_add_3(self, parameter_11, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, reshape_10, reshape_11, matmul_4):
    
        # EarlyReturn(0, 43)

        # pd_op.add: (1x32768x64xf32) <- (1x32768x64xf32, 64xf32)
        add_3 = matmul_4 + parameter_11

        return [layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, reshape_1, transpose_0, reshape_2, reshape_3, conv2d_0, reshape_4, reshape_5, reshape_7, transpose_2, layer_norm_3, layer_norm_4, layer_norm_5, matmul_1, reshape_9, transpose_3, full_int_array_4, full_int_array_5, assign_0, full_int_array_6, slice_1, transpose_4, full_2, softmax_0, reshape_10, reshape_11, matmul_4, add_3]

if True and not (IsCinnStageEnableDiff() and LastCINNStageFailed()):

    class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
        def prepare_data(self):
            self.inputs = [
                # parameter_11
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                # parameter_8
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
                # parameter_9
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                # parameter_7
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                # parameter_6
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                # parameter_10
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                # parameter_2
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                # parameter_0
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                # parameter_5
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                # parameter_1
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                # parameter_4
                paddle.uniform([64, 64, 8, 8], dtype='float32', min=0, max=0.5),
                # parameter_3
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                # data_0
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                # data_1
                paddle.to_tensor([128], dtype='int32').reshape([]),
                # data_2
                paddle.to_tensor([256], dtype='int32').reshape([]),
            ]
            for input in self.inputs:
                input.stop_gradient = True

        def apply_to_static(self, net, use_cinn):
            build_strategy = paddle.static.BuildStrategy()
            input_spec = [
                # parameter_11
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                # parameter_8
                paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
                # parameter_9
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                # parameter_7
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                # parameter_6
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                # parameter_10
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
                # parameter_2
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
                # parameter_0
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                # parameter_5
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                # parameter_1
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                # parameter_4
                paddle.static.InputSpec(shape=[64, 64, 8, 8], dtype='float32'),
                # parameter_3
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                # data_0
                paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
                # data_1
                paddle.static.InputSpec(shape=[], dtype='int32'),
                # data_2
                paddle.static.InputSpec(shape=[], dtype='int32'),
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