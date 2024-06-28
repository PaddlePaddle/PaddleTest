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
    return [37][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_0, data_1, data_2, data_3, data_4):

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_0, reshape_1 = paddle.reshape(data_0, full_int_array_0), None

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(data_1, full_int_array_0), None

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_4, reshape_5 = paddle.reshape(data_2, full_int_array_0), None

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_6, reshape_7 = paddle.reshape(data_3, full_int_array_0), None

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_8, reshape_9 = paddle.reshape(data_4, full_int_array_0), None

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32]) <- (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32)
        combine_0 = [reshape_0, reshape_2, reshape_4, reshape_6, reshape_8]

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_0 = paddle.transpose(data_5, perm=[0, 2, 3, 1])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [1, -1, 1]

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_10, reshape_11 = paddle.reshape(transpose_0, full_int_array_1), None

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_1 = paddle.transpose(data_6, perm=[0, 2, 3, 1])

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_12, reshape_13 = paddle.reshape(transpose_1, full_int_array_1), None

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_2 = paddle.transpose(data_7, perm=[0, 2, 3, 1])

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_14, reshape_15 = paddle.reshape(transpose_2, full_int_array_1), None

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_3 = paddle.transpose(data_8, perm=[0, 2, 3, 1])

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_16, reshape_17 = paddle.reshape(transpose_3, full_int_array_1), None

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_4 = paddle.transpose(data_9, perm=[0, 2, 3, 1])

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_18, reshape_19 = paddle.reshape(transpose_4, full_int_array_1), None

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_1

        # builtin.combine: ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32]) <- (1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32)
        combine_1 = [reshape_10, reshape_12, reshape_14, reshape_16, reshape_18]

        # pd_op.concat: (1x-1x1xf32) <- ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_5 = paddle.transpose(data_10, perm=[0, 2, 3, 1])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, -1, 4]

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_20, reshape_21 = paddle.reshape(transpose_5, full_int_array_2), None

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_6 = paddle.transpose(data_11, perm=[0, 2, 3, 1])

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_22, reshape_23 = paddle.reshape(transpose_6, full_int_array_2), None

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_7 = paddle.transpose(data_12, perm=[0, 2, 3, 1])

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_24, reshape_25 = paddle.reshape(transpose_7, full_int_array_2), None

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_8 = paddle.transpose(data_13, perm=[0, 2, 3, 1])

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_26, reshape_27 = paddle.reshape(transpose_8, full_int_array_2), None

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_9 = paddle.transpose(data_14, perm=[0, 2, 3, 1])

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_28, reshape_29 = paddle.reshape(transpose_9, full_int_array_2), None

        # builtin.combine: ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32]) <- (1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32)
        combine_2 = [reshape_20, reshape_22, reshape_24, reshape_26, reshape_28]

        # pd_op.concat: (1x-1x4xf32) <- ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, assign_0)
        return reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27, reshape_28, reshape_29, assign_0, concat_0, concat_1, concat_2



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

    def forward(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_0, data_1, data_2, data_3, data_4):
        args = [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_0, data_1, data_2, data_3, data_4]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_full_int_array_0,
            self.op_reshape_0,
            self.op_reshape_1,
            self.op_reshape_2,
            self.op_reshape_3,
            self.op_reshape_4,
            self.op_full_0,
            self.op_combine_0,
            self.op_concat_0,
            self.op_transpose_0,
            self.op_full_int_array_1,
            self.op_reshape_5,
            self.op_transpose_1,
            self.op_reshape_6,
            self.op_transpose_2,
            self.op_reshape_7,
            self.op_transpose_3,
            self.op_reshape_8,
            self.op_transpose_4,
            self.op_reshape_9,
            self.op_full_1,
            self.op_assign_0,
            self.op_combine_1,
            self.op_concat_1,
            self.op_transpose_5,
            self.op_full_int_array_2,
            self.op_reshape_10,
            self.op_transpose_6,
            self.op_reshape_11,
            self.op_transpose_7,
            self.op_reshape_12,
            self.op_transpose_8,
            self.op_reshape_13,
            self.op_transpose_9,
            self.op_reshape_14,
            self.op_combine_2,
            self.op_concat_2,
        ]

    def op_full_int_array_0(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_0, data_1, data_2, data_3, data_4):
    
        # EarlyReturn(0, 0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        return [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_0, data_1, data_2, data_3, data_4, full_int_array_0]

    def op_reshape_0(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_0, data_1, data_2, data_3, data_4, full_int_array_0):
    
        # EarlyReturn(0, 1)

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_0, reshape_1 = paddle.reshape(data_0, full_int_array_0), None

        return [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_1, data_2, data_3, data_4, full_int_array_0, reshape_0, reshape_1]

    def op_reshape_1(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_1, data_2, data_3, data_4, full_int_array_0, reshape_0, reshape_1):
    
        # EarlyReturn(0, 2)

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(data_1, full_int_array_0), None

        return [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_2, data_3, data_4, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3]

    def op_reshape_2(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_2, data_3, data_4, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3):
    
        # EarlyReturn(0, 3)

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_4, reshape_5 = paddle.reshape(data_2, full_int_array_0), None

        return [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_3, data_4, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5]

    def op_reshape_3(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_3, data_4, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5):
    
        # EarlyReturn(0, 4)

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_6, reshape_7 = paddle.reshape(data_3, full_int_array_0), None

        return [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_4, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7]

    def op_reshape_4(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, data_4, full_int_array_0, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7):
    
        # EarlyReturn(0, 5)

        # pd_op.reshape: (-1x4xf32, 0x-1x-1xi64) <- (-1x-1xf32, 2xi64)
        reshape_8, reshape_9 = paddle.reshape(data_4, full_int_array_0), None

        return [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9]

    def op_full_0(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9):
    
        # EarlyReturn(0, 6)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        return [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0]

    def op_combine_0(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0):
    
        # EarlyReturn(0, 7)

        # builtin.combine: ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32]) <- (-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32)
        combine_0 = [reshape_0, reshape_2, reshape_4, reshape_6, reshape_8]

        return [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, combine_0]

    def op_concat_0(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, combine_0):
    
        # EarlyReturn(0, 8)

        # pd_op.concat: (-1x4xf32) <- ([-1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32, -1x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        return [data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0]

    def op_transpose_0(self, data_5, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0):
    
        # EarlyReturn(0, 9)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_0 = paddle.transpose(data_5, perm=[0, 2, 3, 1])

        return [data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, transpose_0]

    def op_full_int_array_1(self, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, transpose_0):
    
        # EarlyReturn(0, 10)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [1, -1, 1]

        return [data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, transpose_0, full_int_array_1]

    def op_reshape_5(self, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, transpose_0, full_int_array_1):
    
        # EarlyReturn(0, 11)

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_10, reshape_11 = paddle.reshape(transpose_0, full_int_array_1), None

        return [data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11]

    def op_transpose_1(self, data_6, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11):
    
        # EarlyReturn(0, 12)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_1 = paddle.transpose(data_6, perm=[0, 2, 3, 1])

        return [data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, transpose_1]

    def op_reshape_6(self, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, transpose_1):
    
        # EarlyReturn(0, 13)

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_12, reshape_13 = paddle.reshape(transpose_1, full_int_array_1), None

        return [data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13]

    def op_transpose_2(self, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13):
    
        # EarlyReturn(0, 14)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_2 = paddle.transpose(data_7, perm=[0, 2, 3, 1])

        return [data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, transpose_2]

    def op_reshape_7(self, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, transpose_2):
    
        # EarlyReturn(0, 15)

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_14, reshape_15 = paddle.reshape(transpose_2, full_int_array_1), None

        return [data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15]

    def op_transpose_3(self, data_8, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15):
    
        # EarlyReturn(0, 16)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_3 = paddle.transpose(data_8, perm=[0, 2, 3, 1])

        return [data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, transpose_3]

    def op_reshape_8(self, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, transpose_3):
    
        # EarlyReturn(0, 17)

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_16, reshape_17 = paddle.reshape(transpose_3, full_int_array_1), None

        return [data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17]

    def op_transpose_4(self, data_9, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17):
    
        # EarlyReturn(0, 18)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_4 = paddle.transpose(data_9, perm=[0, 2, 3, 1])

        return [data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, transpose_4]

    def op_reshape_9(self, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, full_int_array_1, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, transpose_4):
    
        # EarlyReturn(0, 19)

        # pd_op.reshape: (1x-1x1xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_18, reshape_19 = paddle.reshape(transpose_4, full_int_array_1), None

        return [data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19]

    def op_full_1(self, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19):
    
        # EarlyReturn(0, 20)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

        return [data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1]

    def op_assign_0(self, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1):
    
        # EarlyReturn(0, 21)

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_1

        return [data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0]

    def op_combine_1(self, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0):
    
        # EarlyReturn(0, 22)

        # builtin.combine: ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32]) <- (1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32)
        combine_1 = [reshape_10, reshape_12, reshape_14, reshape_16, reshape_18]

        return [data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, combine_1]

    def op_concat_1(self, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, combine_1):
    
        # EarlyReturn(0, 23)

        # pd_op.concat: (1x-1x1xf32) <- ([1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32, 1x-1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)

        return [data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1]

    def op_transpose_5(self, data_10, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1):
    
        # EarlyReturn(0, 24)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_5 = paddle.transpose(data_10, perm=[0, 2, 3, 1])

        return [data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, transpose_5]

    def op_full_int_array_2(self, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, transpose_5):
    
        # EarlyReturn(0, 25)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, -1, 4]

        return [data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, transpose_5, full_int_array_2]

    def op_reshape_10(self, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, transpose_5, full_int_array_2):
    
        # EarlyReturn(0, 26)

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_20, reshape_21 = paddle.reshape(transpose_5, full_int_array_2), None

        return [data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21]

    def op_transpose_6(self, data_11, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21):
    
        # EarlyReturn(0, 27)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_6 = paddle.transpose(data_11, perm=[0, 2, 3, 1])

        return [data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, transpose_6]

    def op_reshape_11(self, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, transpose_6):
    
        # EarlyReturn(0, 28)

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_22, reshape_23 = paddle.reshape(transpose_6, full_int_array_2), None

        return [data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23]

    def op_transpose_7(self, data_12, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23):
    
        # EarlyReturn(0, 29)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_7 = paddle.transpose(data_12, perm=[0, 2, 3, 1])

        return [data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, transpose_7]

    def op_reshape_12(self, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, transpose_7):
    
        # EarlyReturn(0, 30)

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_24, reshape_25 = paddle.reshape(transpose_7, full_int_array_2), None

        return [data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25]

    def op_transpose_8(self, data_13, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25):
    
        # EarlyReturn(0, 31)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_8 = paddle.transpose(data_13, perm=[0, 2, 3, 1])

        return [data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, transpose_8]

    def op_reshape_13(self, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, transpose_8):
    
        # EarlyReturn(0, 32)

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_26, reshape_27 = paddle.reshape(transpose_8, full_int_array_2), None

        return [data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27]

    def op_transpose_9(self, data_14, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27):
    
        # EarlyReturn(0, 33)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_9 = paddle.transpose(data_14, perm=[0, 2, 3, 1])

        return [reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27, transpose_9]

    def op_reshape_14(self, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, full_int_array_2, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27, transpose_9):
    
        # EarlyReturn(0, 34)

        # pd_op.reshape: (1x-1x4xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_28, reshape_29 = paddle.reshape(transpose_9, full_int_array_2), None

        return [reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27, reshape_28, reshape_29]

    def op_combine_2(self, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27, reshape_28, reshape_29):
    
        # EarlyReturn(0, 35)

        # builtin.combine: ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32]) <- (1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32)
        combine_2 = [reshape_20, reshape_22, reshape_24, reshape_26, reshape_28]

        return [reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27, reshape_28, reshape_29, combine_2]

    def op_concat_2(self, reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, concat_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, assign_0, concat_1, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27, reshape_28, reshape_29, combine_2):
    
        # EarlyReturn(0, 36)

        # pd_op.concat: (1x-1x4xf32) <- ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, assign_0)

        return [reshape_0, reshape_1, reshape_2, reshape_3, reshape_4, reshape_5, reshape_6, reshape_7, reshape_8, reshape_9, full_0, reshape_10, reshape_11, reshape_12, reshape_13, reshape_14, reshape_15, reshape_16, reshape_17, reshape_18, reshape_19, full_1, reshape_20, reshape_21, reshape_22, reshape_23, reshape_24, reshape_25, reshape_26, reshape_27, reshape_28, reshape_29, assign_0, concat_0, concat_1, concat_2]

if True and not (IsCinnStageEnableDiff() and LastCINNStageFailed()):

    class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
        def prepare_data(self):
            self.inputs = [
                # data_5
                paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
                # data_6
                paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
                # data_7
                paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
                # data_8
                paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
                # data_9
                paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
                # data_10
                paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
                # data_11
                paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
                # data_12
                paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
                # data_13
                paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
                # data_14
                paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
                # data_0
                paddle.uniform([139392, 4], dtype='float32', min=0, max=0.5),
                # data_1
                paddle.uniform([34848, 4], dtype='float32', min=0, max=0.5),
                # data_2
                paddle.uniform([8712, 4], dtype='float32', min=0, max=0.5),
                # data_3
                paddle.uniform([2178, 4], dtype='float32', min=0, max=0.5),
                # data_4
                paddle.uniform([528, 4], dtype='float32', min=0, max=0.5),
            ]
            for input in self.inputs:
                input.stop_gradient = True

        def apply_to_static(self, net, use_cinn):
            build_strategy = paddle.static.BuildStrategy()
            input_spec = [
                # data_5
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_6
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_7
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_8
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_9
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_10
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_11
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_12
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_13
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_14
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                # data_0
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                # data_1
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                # data_2
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                # data_3
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                # data_4
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
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