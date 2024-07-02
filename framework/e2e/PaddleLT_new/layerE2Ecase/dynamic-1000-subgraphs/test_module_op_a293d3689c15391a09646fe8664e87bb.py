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
    return [53][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):

        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full([1], -2, paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1x-1xi64) <- (-1x-1x-1xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(data_0, full_0, False, False, paddle.int64)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 2, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1xi64) <- (-1x-1xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_1, full_1, 0, True)

        # pd_op.add: (-1x-1xi64) <- (-1x-1xi64, -1x-1xi64)
        add_0 = argmax_0 + scale_0

        # pd_op.flatten: (-1xi32, 0x-1x-1x-1xf32) <- (-1x-1x-1xi32)
        flatten_0, flatten_1 = paddle._C_ops.flatten(data_2, 0, 2), None

        # pd_op.flatten: (-1xi64, 0x-1x-1xf32) <- (-1x-1xi64)
        flatten_2, flatten_3 = paddle._C_ops.flatten(add_0, 0, 1), None

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        # pd_op.gather: (-1xi32) <- (-1xi32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(flatten_0, flatten_2, full_2)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 3549]

        # pd_op.reshape: (1x3549xi32, 0x-1xi64) <- (-1xi32, 2xi64)
        reshape_0, reshape_1 = paddle.reshape(gather_0, full_int_array_0), None

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full([], 0, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.greater_than: (-1x-1xb) <- (-1x-1xf32, xf32)
        greater_than_0 = data_3 > full_3

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 80, paddle.float32, paddle.core.CPUPlace())

        # pd_op.full_like: (1x3549xi32) <- (1x3549xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(reshape_0, full_4, paddle.int32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        # pd_op.full_like: (1x3549xi32) <- (1x3549xi32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(reshape_0, full_5, paddle.int32, paddle.framework._current_expected_place())

        # pd_op.full_like: (1x3549xi32) <- (1x3549xi32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(full_like_0, full_5, paddle.int32, paddle.framework._current_expected_place())

        # pd_op.full_like: (-1x-1xb) <- (-1x-1xb, 1xf32)
        full_like_3 = paddle._C_ops.full_like(greater_than_0, full_5, paddle.bool, paddle.framework._current_expected_place())

        # pd_op.cast: (-1x-1xi32) <- (-1x-1xb)
        cast_0 = paddle._C_ops.cast(full_like_3, paddle.int32)

        # pd_op.cast: (-1x-1xi32) <- (-1x-1xb)
        cast_1 = paddle._C_ops.cast(greater_than_0, paddle.int32)

        # pd_op.add: (1x3549xi32) <- (1x3549xi32, 1x3549xi32)
        add_1 = full_like_1 + full_like_2

        # pd_op.add: (-1x3549xi32) <- (1x3549xi32, -1x-1xi32)
        add_2 = add_1 + cast_0

        # pd_op.add: (-1x3549xi32) <- (1x3549xi32, -1x3549xi32)
        add_3 = reshape_0 + add_2

        # pd_op.add: (-1x3549xi32) <- (1x3549xi32, -1x3549xi32)
        add_4 = full_like_0 + add_2

        # pd_op.add: (-1x3549xi32) <- (-1x-1xi32, -1x3549xi32)
        add_5 = cast_1 + add_2

        # pd_op.cast: (-1x3549xb) <- (-1x3549xi32)
        cast_2 = paddle._C_ops.cast(add_5, paddle.bool)

        # pd_op.where: (-1x3549xi32) <- (-1x3549xb, -1x3549xi32, -1x3549xi32)
        where_0 = paddle._C_ops.where(cast_2, add_3, add_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 4]

        # pd_op.reshape: (-1x4xf32, 0x-1x-1x-1xi64) <- (-1x-1x-1xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(data_4, full_int_array_1), None

        # pd_op.gather: (-1x4xf32) <- (-1x4xf32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(reshape_2, flatten_2, full_2)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, 3549, 4]

        # pd_op.reshape: (1x3549x4xf32, 0x-1x4xi64) <- (-1x4xf32, 3xi64)
        reshape_4, reshape_5 = paddle.reshape(gather_1, full_int_array_2), None

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], 81, paddle.int32, paddle.core.CPUPlace())

        # pd_op.one_hot: (-1x3549x81xf32) <- (-1x3549xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(where_0 % full_6, full_6)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_7

        # pd_op.arange: (80xi64) <- (1xf32, 1xf32, 1xf32)
        arange_0 = paddle.arange(full_5, full_4, full_7, dtype=paddle.int64)

        # pd_op.index_select: (-1x3549x80xf32) <- (-1x3549x81xf32, 80xi64)
        index_select_0 = paddle._C_ops.index_select(one_hot_0, arange_0, -1)

        # pd_op.multiply: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x-1xf32)
        multiply_0 = data_5 * data_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_3

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_3

        # pd_op.max: (-1x-1x1xf32) <- (-1x-1x-1xf32, 1xi64)
        max_0 = paddle._C_ops.max(multiply_0, full_int_array_3, True)

        # pd_op.multiply: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x-1xf32)
        multiply_1 = data_6 * data_0

        # pd_op.max: (-1x-1x1xf32) <- (-1x-1x-1xf32, 1xi64)
        max_1 = paddle._C_ops.max(multiply_1, assign_2, True)

        # pd_op.scale: (-1x-1x1xf32) <- (-1x-1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(max_0, assign_0, 1e-09, True)

        # pd_op.divide: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x1xf32)
        divide_0 = multiply_0 / scale_1

        # pd_op.multiply: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x1xf32)
        multiply_2 = divide_0 * max_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-2]

        # pd_op.max: (-1x-1xf32) <- (-1x-1x-1xf32, 1xi64)
        max_2 = paddle._C_ops.max(multiply_2, full_int_array_4, False)

        # pd_op.unsqueeze: (-1x-1x1xf32, 0x-1x-1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_0, unsqueeze_1 = paddle.unsqueeze(max_2, assign_1), None

        # pd_op.multiply: (-1x3549x80xf32) <- (-1x3549x80xf32, -1x-1x1xf32)
        multiply_3 = index_select_0 * unsqueeze_0
        return index_select_0, multiply_0, full_int_array_3, max_0, multiply_1, assign_2, max_1, assign_0, scale_1, divide_0, multiply_2, full_int_array_4, max_2, assign_1, unsqueeze_0, unsqueeze_1, where_0, reshape_4, multiply_3



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

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        args = [data_0, data_1, data_2, data_3, data_4, data_5, data_6]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_full_0,
            self.op_argmax_0,
            self.op_full_1,
            self.op_scale_0,
            self.op_add_0,
            self.op_flatten_0,
            self.op_flatten_1,
            self.op_full_2,
            self.op_gather_0,
            self.op_full_int_array_0,
            self.op_reshape_0,
            self.op_full_3,
            self.op_greater_than_0,
            self.op_full_4,
            self.op_full_like_0,
            self.op_full_5,
            self.op_full_like_1,
            self.op_full_like_2,
            self.op_full_like_3,
            self.op_cast_0,
            self.op_cast_1,
            self.op_add_1,
            self.op_add_2,
            self.op_add_3,
            self.op_add_4,
            self.op_add_5,
            self.op_cast_2,
            self.op_where_0,
            self.op_full_int_array_1,
            self.op_reshape_1,
            self.op_gather_1,
            self.op_full_int_array_2,
            self.op_reshape_2,
            self.op_full_6,
            self.op_one_hot_0,
            self.op_full_7,
            self.op_assign_0,
            self.op_arange_0,
            self.op_index_select_0,
            self.op_multiply_0,
            self.op_full_int_array_3,
            self.op_assign_1,
            self.op_assign_2,
            self.op_max_0,
            self.op_multiply_1,
            self.op_max_1,
            self.op_scale_1,
            self.op_divide_0,
            self.op_multiply_2,
            self.op_full_int_array_4,
            self.op_max_2,
            self.op_unsqueeze_0,
            self.op_multiply_3,
        ]

    def op_full_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
    
        # EarlyReturn(0, 0)

        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full([1], -2, paddle.int64, paddle.core.CPUPlace())

        return [data_0, data_1, data_2, data_3, data_4, data_5, data_6, full_0]

    def op_argmax_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, full_0):
    
        # EarlyReturn(0, 1)

        # pd_op.argmax: (-1x-1xi64) <- (-1x-1x-1xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(data_0, full_0, False, False, paddle.int64)

        return [data_0, data_1, data_2, data_3, data_4, data_5, data_6, argmax_0]

    def op_full_1(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, argmax_0):
    
        # EarlyReturn(0, 2)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 2, paddle.float32, paddle.core.CPUPlace())

        return [data_0, data_1, data_2, data_3, data_4, data_5, data_6, argmax_0, full_1]

    def op_scale_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, argmax_0, full_1):
    
        # EarlyReturn(0, 3)

        # pd_op.scale: (-1x-1xi64) <- (-1x-1xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_1, full_1, 0, True)

        return [data_0, data_2, data_3, data_4, data_5, data_6, argmax_0, scale_0]

    def op_add_0(self, data_0, data_2, data_3, data_4, data_5, data_6, argmax_0, scale_0):
    
        # EarlyReturn(0, 4)

        # pd_op.add: (-1x-1xi64) <- (-1x-1xi64, -1x-1xi64)
        add_0 = argmax_0 + scale_0

        return [data_0, data_2, data_3, data_4, data_5, data_6, add_0]

    def op_flatten_0(self, data_0, data_2, data_3, data_4, data_5, data_6, add_0):
    
        # EarlyReturn(0, 5)

        # pd_op.flatten: (-1xi32, 0x-1x-1x-1xf32) <- (-1x-1x-1xi32)
        flatten_0, flatten_1 = paddle._C_ops.flatten(data_2, 0, 2), None

        return [data_0, data_3, data_4, data_5, data_6, add_0, flatten_0]

    def op_flatten_1(self, data_0, data_3, data_4, data_5, data_6, add_0, flatten_0):
    
        # EarlyReturn(0, 6)

        # pd_op.flatten: (-1xi64, 0x-1x-1xf32) <- (-1x-1xi64)
        flatten_2, flatten_3 = paddle._C_ops.flatten(add_0, 0, 1), None

        return [data_0, data_3, data_4, data_5, data_6, flatten_0, flatten_2]

    def op_full_2(self, data_0, data_3, data_4, data_5, data_6, flatten_0, flatten_2):
    
        # EarlyReturn(0, 7)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        return [data_0, data_3, data_4, data_5, data_6, flatten_0, flatten_2, full_2]

    def op_gather_0(self, data_0, data_3, data_4, data_5, data_6, flatten_0, flatten_2, full_2):
    
        # EarlyReturn(0, 8)

        # pd_op.gather: (-1xi32) <- (-1xi32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(flatten_0, flatten_2, full_2)

        return [data_0, data_3, data_4, data_5, data_6, flatten_2, full_2, gather_0]

    def op_full_int_array_0(self, data_0, data_3, data_4, data_5, data_6, flatten_2, full_2, gather_0):
    
        # EarlyReturn(0, 9)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 3549]

        return [data_0, data_3, data_4, data_5, data_6, flatten_2, full_2, gather_0, full_int_array_0]

    def op_reshape_0(self, data_0, data_3, data_4, data_5, data_6, flatten_2, full_2, gather_0, full_int_array_0):
    
        # EarlyReturn(0, 10)

        # pd_op.reshape: (1x3549xi32, 0x-1xi64) <- (-1xi32, 2xi64)
        reshape_0, reshape_1 = paddle.reshape(gather_0, full_int_array_0), None

        return [data_0, data_3, data_4, data_5, data_6, flatten_2, full_2, reshape_0]

    def op_full_3(self, data_0, data_3, data_4, data_5, data_6, flatten_2, full_2, reshape_0):
    
        # EarlyReturn(0, 11)

        # pd_op.full: (xf32) <- ()
        full_3 = paddle._C_ops.full([], 0, paddle.float32, paddle.framework._current_expected_place())

        return [data_0, data_3, data_4, data_5, data_6, flatten_2, full_2, reshape_0, full_3]

    def op_greater_than_0(self, data_0, data_3, data_4, data_5, data_6, flatten_2, full_2, reshape_0, full_3):
    
        # EarlyReturn(0, 12)

        # pd_op.greater_than: (-1x-1xb) <- (-1x-1xf32, xf32)
        greater_than_0 = data_3 > full_3

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0]

    def op_full_4(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0):
    
        # EarlyReturn(0, 13)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 80, paddle.float32, paddle.core.CPUPlace())

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4]

    def op_full_like_0(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4):
    
        # EarlyReturn(0, 14)

        # pd_op.full_like: (1x3549xi32) <- (1x3549xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(reshape_0, full_4, paddle.int32, paddle.framework._current_expected_place())

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0]

    def op_full_5(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0):
    
        # EarlyReturn(0, 15)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5]

    def op_full_like_1(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5):
    
        # EarlyReturn(0, 16)

        # pd_op.full_like: (1x3549xi32) <- (1x3549xi32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(reshape_0, full_5, paddle.int32, paddle.framework._current_expected_place())

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5, full_like_1]

    def op_full_like_2(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5, full_like_1):
    
        # EarlyReturn(0, 17)

        # pd_op.full_like: (1x3549xi32) <- (1x3549xi32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(full_like_0, full_5, paddle.int32, paddle.framework._current_expected_place())

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5, full_like_1, full_like_2]

    def op_full_like_3(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5, full_like_1, full_like_2):
    
        # EarlyReturn(0, 18)

        # pd_op.full_like: (-1x-1xb) <- (-1x-1xb, 1xf32)
        full_like_3 = paddle._C_ops.full_like(greater_than_0, full_5, paddle.bool, paddle.framework._current_expected_place())

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5, full_like_1, full_like_2, full_like_3]

    def op_cast_0(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5, full_like_1, full_like_2, full_like_3):
    
        # EarlyReturn(0, 19)

        # pd_op.cast: (-1x-1xi32) <- (-1x-1xb)
        cast_0 = paddle._C_ops.cast(full_like_3, paddle.int32)

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5, full_like_1, full_like_2, cast_0]

    def op_cast_1(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, greater_than_0, full_4, full_like_0, full_5, full_like_1, full_like_2, cast_0):
    
        # EarlyReturn(0, 20)

        # pd_op.cast: (-1x-1xi32) <- (-1x-1xb)
        cast_1 = paddle._C_ops.cast(greater_than_0, paddle.int32)

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, full_4, full_like_0, full_5, full_like_1, full_like_2, cast_0, cast_1]

    def op_add_1(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, full_4, full_like_0, full_5, full_like_1, full_like_2, cast_0, cast_1):
    
        # EarlyReturn(0, 21)

        # pd_op.add: (1x3549xi32) <- (1x3549xi32, 1x3549xi32)
        add_1 = full_like_1 + full_like_2

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, full_4, full_like_0, full_5, cast_0, cast_1, add_1]

    def op_add_2(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, full_4, full_like_0, full_5, cast_0, cast_1, add_1):
    
        # EarlyReturn(0, 22)

        # pd_op.add: (-1x3549xi32) <- (1x3549xi32, -1x-1xi32)
        add_2 = add_1 + cast_0

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, full_4, full_like_0, full_5, cast_1, add_2]

    def op_add_3(self, data_0, data_4, data_5, data_6, flatten_2, full_2, reshape_0, full_4, full_like_0, full_5, cast_1, add_2):
    
        # EarlyReturn(0, 23)

        # pd_op.add: (-1x3549xi32) <- (1x3549xi32, -1x3549xi32)
        add_3 = reshape_0 + add_2

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_like_0, full_5, cast_1, add_2, add_3]

    def op_add_4(self, data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_like_0, full_5, cast_1, add_2, add_3):
    
        # EarlyReturn(0, 24)

        # pd_op.add: (-1x3549xi32) <- (1x3549xi32, -1x3549xi32)
        add_4 = full_like_0 + add_2

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, cast_1, add_2, add_3, add_4]

    def op_add_5(self, data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, cast_1, add_2, add_3, add_4):
    
        # EarlyReturn(0, 25)

        # pd_op.add: (-1x3549xi32) <- (-1x-1xi32, -1x3549xi32)
        add_5 = cast_1 + add_2

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, add_3, add_4, add_5]

    def op_cast_2(self, data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, add_3, add_4, add_5):
    
        # EarlyReturn(0, 26)

        # pd_op.cast: (-1x3549xb) <- (-1x3549xi32)
        cast_2 = paddle._C_ops.cast(add_5, paddle.bool)

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, add_3, add_4, cast_2]

    def op_where_0(self, data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, add_3, add_4, cast_2):
    
        # EarlyReturn(0, 27)

        # pd_op.where: (-1x3549xi32) <- (-1x3549xb, -1x3549xi32, -1x3549xi32)
        where_0 = paddle._C_ops.where(cast_2, add_3, add_4)

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, where_0]

    def op_full_int_array_1(self, data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, where_0):
    
        # EarlyReturn(0, 28)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 4]

        return [data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, where_0, full_int_array_1]

    def op_reshape_1(self, data_0, data_4, data_5, data_6, flatten_2, full_2, full_4, full_5, where_0, full_int_array_1):
    
        # EarlyReturn(0, 29)

        # pd_op.reshape: (-1x4xf32, 0x-1x-1x-1xi64) <- (-1x-1x-1xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(data_4, full_int_array_1), None

        return [data_0, data_5, data_6, flatten_2, full_2, full_4, full_5, where_0, reshape_2]

    def op_gather_1(self, data_0, data_5, data_6, flatten_2, full_2, full_4, full_5, where_0, reshape_2):
    
        # EarlyReturn(0, 30)

        # pd_op.gather: (-1x4xf32) <- (-1x4xf32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(reshape_2, flatten_2, full_2)

        return [data_0, data_5, data_6, full_4, full_5, where_0, gather_1]

    def op_full_int_array_2(self, data_0, data_5, data_6, full_4, full_5, where_0, gather_1):
    
        # EarlyReturn(0, 31)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, 3549, 4]

        return [data_0, data_5, data_6, full_4, full_5, where_0, gather_1, full_int_array_2]

    def op_reshape_2(self, data_0, data_5, data_6, full_4, full_5, where_0, gather_1, full_int_array_2):
    
        # EarlyReturn(0, 32)

        # pd_op.reshape: (1x3549x4xf32, 0x-1x4xi64) <- (-1x4xf32, 3xi64)
        reshape_4, reshape_5 = paddle.reshape(gather_1, full_int_array_2), None

        return [data_0, data_5, data_6, full_4, full_5, where_0, reshape_4]

    def op_full_6(self, data_0, data_5, data_6, full_4, full_5, where_0, reshape_4):
    
        # EarlyReturn(0, 33)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], 81, paddle.int32, paddle.core.CPUPlace())

        return [data_0, data_5, data_6, full_4, full_5, where_0, reshape_4, full_6]

    def op_one_hot_0(self, data_0, data_5, data_6, full_4, full_5, where_0, reshape_4, full_6):
    
        # EarlyReturn(0, 34)

        # pd_op.one_hot: (-1x3549x81xf32) <- (-1x3549xi32, 1xi32)
        one_hot_0 = paddle._C_ops.one_hot(where_0 % full_6, full_6)

        return [data_0, data_5, data_6, full_4, full_5, where_0, reshape_4, one_hot_0]

    def op_full_7(self, data_0, data_5, data_6, full_4, full_5, where_0, reshape_4, one_hot_0):
    
        # EarlyReturn(0, 35)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        return [data_0, data_5, data_6, full_4, full_5, where_0, reshape_4, one_hot_0, full_7]

    def op_assign_0(self, data_0, data_5, data_6, full_4, full_5, where_0, reshape_4, one_hot_0, full_7):
    
        # EarlyReturn(0, 36)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_7

        return [data_0, data_5, data_6, full_4, full_5, where_0, reshape_4, one_hot_0, full_7, assign_0]

    def op_arange_0(self, data_0, data_5, data_6, full_4, full_5, where_0, reshape_4, one_hot_0, full_7, assign_0):
    
        # EarlyReturn(0, 37)

        # pd_op.arange: (80xi64) <- (1xf32, 1xf32, 1xf32)
        arange_0 = paddle.arange(full_5, full_4, full_7, dtype=paddle.int64)

        return [data_0, data_5, data_6, where_0, reshape_4, one_hot_0, assign_0, arange_0]

    def op_index_select_0(self, data_0, data_5, data_6, where_0, reshape_4, one_hot_0, assign_0, arange_0):
    
        # EarlyReturn(0, 38)

        # pd_op.index_select: (-1x3549x80xf32) <- (-1x3549x81xf32, 80xi64)
        index_select_0 = paddle._C_ops.index_select(one_hot_0, arange_0, -1)

        return [data_0, data_5, data_6, where_0, reshape_4, assign_0, index_select_0]

    def op_multiply_0(self, data_0, data_5, data_6, where_0, reshape_4, assign_0, index_select_0):
    
        # EarlyReturn(0, 39)

        # pd_op.multiply: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x-1xf32)
        multiply_0 = data_5 * data_0

        return [data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0]

    def op_full_int_array_3(self, data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0):
    
        # EarlyReturn(0, 40)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-1]

        return [data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3]

    def op_assign_1(self, data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3):
    
        # EarlyReturn(0, 41)

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_3

        return [data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1]

    def op_assign_2(self, data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1):
    
        # EarlyReturn(0, 42)

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_3

        return [data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2]

    def op_max_0(self, data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2):
    
        # EarlyReturn(0, 43)

        # pd_op.max: (-1x-1x1xf32) <- (-1x-1x-1xf32, 1xi64)
        max_0 = paddle._C_ops.max(multiply_0, full_int_array_3, True)

        return [data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0]

    def op_multiply_1(self, data_0, data_6, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0):
    
        # EarlyReturn(0, 44)

        # pd_op.multiply: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x-1xf32)
        multiply_1 = data_6 * data_0

        return [where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1]

    def op_max_1(self, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1):
    
        # EarlyReturn(0, 45)

        # pd_op.max: (-1x-1x1xf32) <- (-1x-1x-1xf32, 1xi64)
        max_1 = paddle._C_ops.max(multiply_1, assign_2, True)

        return [where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1]

    def op_scale_1(self, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1):
    
        # EarlyReturn(0, 46)

        # pd_op.scale: (-1x-1x1xf32) <- (-1x-1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(max_0, assign_0, 1e-09, True)

        return [where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1]

    def op_divide_0(self, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1):
    
        # EarlyReturn(0, 47)

        # pd_op.divide: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x1xf32)
        divide_0 = multiply_0 / scale_1

        return [where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0]

    def op_multiply_2(self, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0):
    
        # EarlyReturn(0, 48)

        # pd_op.multiply: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1x1xf32)
        multiply_2 = divide_0 * max_1

        return [where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0, multiply_2]

    def op_full_int_array_4(self, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0, multiply_2):
    
        # EarlyReturn(0, 49)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-2]

        return [where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0, multiply_2, full_int_array_4]

    def op_max_2(self, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0, multiply_2, full_int_array_4):
    
        # EarlyReturn(0, 50)

        # pd_op.max: (-1x-1xf32) <- (-1x-1x-1xf32, 1xi64)
        max_2 = paddle._C_ops.max(multiply_2, full_int_array_4, False)

        return [where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0, multiply_2, full_int_array_4, max_2]

    def op_unsqueeze_0(self, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0, multiply_2, full_int_array_4, max_2):
    
        # EarlyReturn(0, 51)

        # pd_op.unsqueeze: (-1x-1x1xf32, 0x-1x-1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_0, unsqueeze_1 = paddle.unsqueeze(max_2, assign_1), None

        return [where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0, multiply_2, full_int_array_4, max_2, unsqueeze_0, unsqueeze_1]

    def op_multiply_3(self, where_0, reshape_4, assign_0, index_select_0, multiply_0, full_int_array_3, assign_1, assign_2, max_0, multiply_1, max_1, scale_1, divide_0, multiply_2, full_int_array_4, max_2, unsqueeze_0, unsqueeze_1):
    
        # EarlyReturn(0, 52)

        # pd_op.multiply: (-1x3549x80xf32) <- (-1x3549x80xf32, -1x-1x1xf32)
        multiply_3 = index_select_0 * unsqueeze_0

        return [index_select_0, multiply_0, full_int_array_3, max_0, multiply_1, assign_2, max_1, assign_0, scale_1, divide_0, multiply_2, full_int_array_4, max_2, assign_1, unsqueeze_0, unsqueeze_1, where_0, reshape_4, multiply_3]

is_module_block_and_last_stage_passed = (
    True and not last_stage_failed
)
@unittest.skipIf(not is_module_block_and_last_stage_passed, "last stage failed")
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # data_0
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.to_tensor([9], dtype='int64').reshape([1, 1]),
            # data_2
            paddle.to_tensor([6, 6], dtype='int32').reshape([1, 2, 1]),
            # data_3
            paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([1, 2, 4], dtype='float32', min=0, max=0.5),
            # data_5
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            # data_6
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # data_0
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            # data_2
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            # data_3
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_5
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_6
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