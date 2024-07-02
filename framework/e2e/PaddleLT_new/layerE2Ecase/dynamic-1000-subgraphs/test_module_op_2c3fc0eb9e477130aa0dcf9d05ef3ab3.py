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
    return [26][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, data_1, data_0, data_2):

        # pd_op.cast: (-1xi64) <- (-1xf32)
        cast_0 = paddle._C_ops.cast(data_0, paddle.int64)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_0, 1, True)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(scale_0, paddle.float32)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_0 = cast_1 - data_0

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_2 = paddle._C_ops.cast(cast_0, paddle.float32)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_1 = data_0 - cast_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.unsqueeze: (-1x1xi64, 0x-1xi64) <- (-1xi64, 1xi64)
        unsqueeze_0, unsqueeze_1 = paddle.unsqueeze(cast_0, full_int_array_0), None

        # pd_op.cross_entropy_with_softmax: (-1x-1xf32, -1x1xf32) <- (-1x-1xf32, -1x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_1 = paddle._C_ops.cross_entropy_with_softmax(data_1, unsqueeze_0, False, True, True, -100, -1)

        # pd_op.squeeze: (-1xf32, 0x-1x1xf32) <- (-1x1xf32, 1xi64)
        squeeze_0, squeeze_1 = paddle._C_ops.squeeze(cross_entropy_with_softmax_1, assign_1), None

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_0 = squeeze_0 * subtract_0

        # pd_op.unsqueeze: (-1x1xi64, 0x-1xi64) <- (-1xi64, 1xi64)
        unsqueeze_2, unsqueeze_3 = paddle.unsqueeze(scale_0, full_int_array_0), None

        # pd_op.cross_entropy_with_softmax: (-1x-1xf32, -1x1xf32) <- (-1x-1xf32, -1x1xi64)
        cross_entropy_with_softmax_2, cross_entropy_with_softmax_3 = paddle._C_ops.cross_entropy_with_softmax(data_1, unsqueeze_2, False, True, True, -100, -1)

        # pd_op.squeeze: (-1xf32, 0x-1x1xf32) <- (-1x1xf32, 1xi64)
        squeeze_2, squeeze_3 = paddle._C_ops.squeeze(cross_entropy_with_softmax_3, assign_0), None

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_1 = squeeze_2 * subtract_1

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_0 = multiply_0 + multiply_1

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 0.25, paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_0, full_1, 0, True)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_2 = scale_1 * data_2

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_1 = []

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_1, None, False)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(sum_0, assign_2, 0, True)
        return cast_1, subtract_0, cast_2, subtract_1, unsqueeze_0, cross_entropy_with_softmax_0, assign_1, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, assign_0, squeeze_2, squeeze_3, multiply_1, full_1, scale_1, multiply_2, full_int_array_1, assign_2, scale_2



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

    def forward(self, data_1, data_0, data_2):
        args = [data_1, data_0, data_2]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_cast_0,
            self.op_full_0,
            self.op_scale_0,
            self.op_cast_1,
            self.op_subtract_0,
            self.op_cast_2,
            self.op_subtract_1,
            self.op_full_int_array_0,
            self.op_assign_0,
            self.op_assign_1,
            self.op_unsqueeze_0,
            self.op_cross_entropy_with_softmax_0,
            self.op_squeeze_0,
            self.op_multiply_0,
            self.op_unsqueeze_1,
            self.op_cross_entropy_with_softmax_1,
            self.op_squeeze_1,
            self.op_multiply_1,
            self.op_add_0,
            self.op_full_1,
            self.op_assign_2,
            self.op_scale_1,
            self.op_multiply_2,
            self.op_full_int_array_1,
            self.op_sum_0,
            self.op_scale_2,
        ]

    def op_cast_0(self, data_1, data_0, data_2):
    
        # EarlyReturn(0, 0)

        # pd_op.cast: (-1xi64) <- (-1xf32)
        cast_0 = paddle._C_ops.cast(data_0, paddle.int64)

        return [data_1, data_0, data_2, cast_0]

    def op_full_0(self, data_1, data_0, data_2, cast_0):
    
        # EarlyReturn(0, 1)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        return [data_1, data_0, data_2, cast_0, full_0]

    def op_scale_0(self, data_1, data_0, data_2, cast_0, full_0):
    
        # EarlyReturn(0, 2)

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_0, 1, True)

        return [data_1, data_0, data_2, cast_0, scale_0]

    def op_cast_1(self, data_1, data_0, data_2, cast_0, scale_0):
    
        # EarlyReturn(0, 3)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(scale_0, paddle.float32)

        return [data_1, data_0, data_2, cast_0, scale_0, cast_1]

    def op_subtract_0(self, data_1, data_0, data_2, cast_0, scale_0, cast_1):
    
        # EarlyReturn(0, 4)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_0 = cast_1 - data_0

        return [data_1, data_0, data_2, cast_0, scale_0, cast_1, subtract_0]

    def op_cast_2(self, data_1, data_0, data_2, cast_0, scale_0, cast_1, subtract_0):
    
        # EarlyReturn(0, 5)

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_2 = paddle._C_ops.cast(cast_0, paddle.float32)

        return [data_1, data_0, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2]

    def op_subtract_1(self, data_1, data_0, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2):
    
        # EarlyReturn(0, 6)

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_1 = data_0 - cast_2

        return [data_1, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2, subtract_1]

    def op_full_int_array_0(self, data_1, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2, subtract_1):
    
        # EarlyReturn(0, 7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        return [data_1, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0]

    def op_assign_0(self, data_1, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0):
    
        # EarlyReturn(0, 8)

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        return [data_1, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0]

    def op_assign_1(self, data_1, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0):
    
        # EarlyReturn(0, 9)

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        return [data_1, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1]

    def op_unsqueeze_0(self, data_1, data_2, cast_0, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1):
    
        # EarlyReturn(0, 10)

        # pd_op.unsqueeze: (-1x1xi64, 0x-1xi64) <- (-1xi64, 1xi64)
        unsqueeze_0, unsqueeze_1 = paddle.unsqueeze(cast_0, full_int_array_0), None

        return [data_1, data_2, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1, unsqueeze_0]

    def op_cross_entropy_with_softmax_0(self, data_1, data_2, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1, unsqueeze_0):
    
        # EarlyReturn(0, 11)

        # pd_op.cross_entropy_with_softmax: (-1x-1xf32, -1x1xf32) <- (-1x-1xf32, -1x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_1 = paddle._C_ops.cross_entropy_with_softmax(data_1, unsqueeze_0, False, True, True, -100, -1)

        return [data_1, data_2, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, cross_entropy_with_softmax_1]

    def op_squeeze_0(self, data_1, data_2, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, cross_entropy_with_softmax_1):
    
        # EarlyReturn(0, 12)

        # pd_op.squeeze: (-1xf32, 0x-1x1xf32) <- (-1x1xf32, 1xi64)
        squeeze_0, squeeze_1 = paddle._C_ops.squeeze(cross_entropy_with_softmax_1, assign_1), None

        return [data_1, data_2, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1]

    def op_multiply_0(self, data_1, data_2, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1):
    
        # EarlyReturn(0, 13)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_0 = squeeze_0 * subtract_0

        return [data_1, data_2, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0]

    def op_unsqueeze_1(self, data_1, data_2, scale_0, cast_1, subtract_0, cast_2, subtract_1, full_int_array_0, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0):
    
        # EarlyReturn(0, 14)

        # pd_op.unsqueeze: (-1x1xi64, 0x-1xi64) <- (-1xi64, 1xi64)
        unsqueeze_2, unsqueeze_3 = paddle.unsqueeze(scale_0, full_int_array_0), None

        return [data_1, data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2]

    def op_cross_entropy_with_softmax_1(self, data_1, data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2):
    
        # EarlyReturn(0, 15)

        # pd_op.cross_entropy_with_softmax: (-1x-1xf32, -1x1xf32) <- (-1x-1xf32, -1x1xi64)
        cross_entropy_with_softmax_2, cross_entropy_with_softmax_3 = paddle._C_ops.cross_entropy_with_softmax(data_1, unsqueeze_2, False, True, True, -100, -1)

        return [data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, cross_entropy_with_softmax_3]

    def op_squeeze_1(self, data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, cross_entropy_with_softmax_3):
    
        # EarlyReturn(0, 16)

        # pd_op.squeeze: (-1xf32, 0x-1x1xf32) <- (-1x1xf32, 1xi64)
        squeeze_2, squeeze_3 = paddle._C_ops.squeeze(cross_entropy_with_softmax_3, assign_0), None

        return [data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3]

    def op_multiply_1(self, data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3):
    
        # EarlyReturn(0, 17)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_1 = squeeze_2 * subtract_1

        return [data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1]

    def op_add_0(self, data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1):
    
        # EarlyReturn(0, 18)

        # pd_op.add: (-1xf32) <- (-1xf32, -1xf32)
        add_0 = multiply_0 + multiply_1

        return [data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, add_0]

    def op_full_1(self, data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, add_0):
    
        # EarlyReturn(0, 19)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 0.25, paddle.float32, paddle.core.CPUPlace())

        return [data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, add_0, full_1]

    def op_assign_2(self, data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, add_0, full_1):
    
        # EarlyReturn(0, 20)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_1

        return [data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, add_0, full_1, assign_2]

    def op_scale_1(self, data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, add_0, full_1, assign_2):
    
        # EarlyReturn(0, 21)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(add_0, full_1, 0, True)

        return [data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, full_1, assign_2, scale_1]

    def op_multiply_2(self, data_2, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, full_1, assign_2, scale_1):
    
        # EarlyReturn(0, 22)

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_2 = scale_1 * data_2

        return [cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, full_1, assign_2, scale_1, multiply_2]

    def op_full_int_array_1(self, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, full_1, assign_2, scale_1, multiply_2):
    
        # EarlyReturn(0, 23)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_1 = []

        return [cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, full_1, assign_2, scale_1, multiply_2, full_int_array_1]

    def op_sum_0(self, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, full_1, assign_2, scale_1, multiply_2, full_int_array_1):
    
        # EarlyReturn(0, 24)

        # pd_op.sum: (xf32) <- (-1xf32, 0xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_1, None, False)

        return [cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, full_1, assign_2, scale_1, multiply_2, full_int_array_1, sum_0]

    def op_scale_2(self, cast_1, subtract_0, cast_2, subtract_1, assign_0, assign_1, unsqueeze_0, cross_entropy_with_softmax_0, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, squeeze_2, squeeze_3, multiply_1, full_1, assign_2, scale_1, multiply_2, full_int_array_1, sum_0):
    
        # EarlyReturn(0, 25)

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(sum_0, assign_2, 0, True)

        return [cast_1, subtract_0, cast_2, subtract_1, unsqueeze_0, cross_entropy_with_softmax_0, assign_1, squeeze_0, squeeze_1, multiply_0, unsqueeze_2, cross_entropy_with_softmax_2, assign_0, squeeze_2, squeeze_3, multiply_1, full_1, scale_1, multiply_2, full_int_array_1, assign_2, scale_2]

is_module_block_and_last_stage_passed = (
    True and not last_stage_failed
)
@unittest.skipIf(not is_module_block_and_last_stage_passed, "last stage failed")
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # data_1
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # data_1
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_2
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