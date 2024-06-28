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
    return [35][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7):

        # pd_op.maximum: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        maximum_0 = paddle.maximum(data_0, data_1)

        # pd_op.maximum: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        maximum_1 = paddle.maximum(data_2, data_3)

        # pd_op.minimum: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        minimum_0 = paddle._C_ops.minimum(data_4, data_5)

        # pd_op.minimum: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        minimum_1 = paddle._C_ops.minimum(data_6, data_7)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_0 = minimum_0 - maximum_0

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_0

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 3.40282e+38, paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_1

        # pd_op.clip: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_0, full_1)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_1 = minimum_1 - maximum_1

        # pd_op.clip: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, assign_2, assign_5)

        # pd_op.multiply: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        multiply_0 = clip_0 * clip_1

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_2 = data_4 - data_0

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_3 = data_6 - data_2

        # pd_op.multiply: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        multiply_1 = subtract_2 * subtract_3

        # pd_op.clip: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(multiply_1, assign_1, assign_4)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_4 = data_5 - data_1

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_5 = data_7 - data_3

        # pd_op.multiply: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        multiply_2 = subtract_4 * subtract_5

        # pd_op.clip: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32, 1xf32)
        clip_3 = paddle._C_ops.clip(multiply_2, assign_0, assign_3)

        # pd_op.add: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        add_0 = clip_2 + clip_3

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_6 = add_0 - multiply_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_6, full_2, 1e-09, True)

        # pd_op.divide: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        divide_0 = multiply_0 / scale_0

        # pd_op.multiply: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        multiply_3 = divide_0 * divide_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], -1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_3, full_3, 1, True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 2.5, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_4, 0, True)
        return maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, full_1, clip_0, subtract_1, assign_2, assign_5, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, assign_1, assign_4, clip_2, subtract_4, subtract_5, multiply_2, assign_0, assign_3, clip_3, add_0, full_2, scale_0, divide_0, full_3, full_4, scale_2



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

    def forward(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7):
        args = [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_maximum_0,
            self.op_maximum_1,
            self.op_minimum_0,
            self.op_minimum_1,
            self.op_subtract_0,
            self.op_full_0,
            self.op_assign_0,
            self.op_assign_1,
            self.op_assign_2,
            self.op_full_1,
            self.op_assign_3,
            self.op_assign_4,
            self.op_assign_5,
            self.op_clip_0,
            self.op_subtract_1,
            self.op_clip_1,
            self.op_multiply_0,
            self.op_subtract_2,
            self.op_subtract_3,
            self.op_multiply_1,
            self.op_clip_2,
            self.op_subtract_4,
            self.op_subtract_5,
            self.op_multiply_2,
            self.op_clip_3,
            self.op_add_0,
            self.op_subtract_6,
            self.op_full_2,
            self.op_scale_0,
            self.op_divide_0,
            self.op_multiply_3,
            self.op_full_3,
            self.op_scale_1,
            self.op_full_4,
            self.op_scale_2,
        ]

    def op_maximum_0(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7):
    
        # EarlyReturn(0, 0)

        # pd_op.maximum: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        maximum_0 = paddle.maximum(data_0, data_1)

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0]

    def op_maximum_1(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0):
    
        # EarlyReturn(0, 1)

        # pd_op.maximum: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        maximum_1 = paddle.maximum(data_2, data_3)

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1]

    def op_minimum_0(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1):
    
        # EarlyReturn(0, 2)

        # pd_op.minimum: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        minimum_0 = paddle._C_ops.minimum(data_4, data_5)

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0]

    def op_minimum_1(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0):
    
        # EarlyReturn(0, 3)

        # pd_op.minimum: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        minimum_1 = paddle._C_ops.minimum(data_6, data_7)

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1]

    def op_subtract_0(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1):
    
        # EarlyReturn(0, 4)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_0 = minimum_0 - maximum_0

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0]

    def op_full_0(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0):
    
        # EarlyReturn(0, 5)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0]

    def op_assign_0(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0):
    
        # EarlyReturn(0, 6)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0]

    def op_assign_1(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0):
    
        # EarlyReturn(0, 7)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_1 = full_0

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1]

    def op_assign_2(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1):
    
        # EarlyReturn(0, 8)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_2 = full_0

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2]

    def op_full_1(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2):
    
        # EarlyReturn(0, 9)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], 3.40282e+38, paddle.float32, paddle.core.CPUPlace())

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1]

    def op_assign_3(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1):
    
        # EarlyReturn(0, 10)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_1

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3]

    def op_assign_4(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3):
    
        # EarlyReturn(0, 11)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_1

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4]

    def op_assign_5(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4):
    
        # EarlyReturn(0, 12)

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_1

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5]

    def op_clip_0(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5):
    
        # EarlyReturn(0, 13)

        # pd_op.clip: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_0, full_0, full_1)

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0]

    def op_subtract_1(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0):
    
        # EarlyReturn(0, 14)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_1 = minimum_1 - maximum_1

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1]

    def op_clip_1(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1):
    
        # EarlyReturn(0, 15)

        # pd_op.clip: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_1, assign_2, assign_5)

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1]

    def op_multiply_0(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1):
    
        # EarlyReturn(0, 16)

        # pd_op.multiply: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        multiply_0 = clip_0 * clip_1

        return [data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0]

    def op_subtract_2(self, data_0, data_2, data_4, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0):
    
        # EarlyReturn(0, 17)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_2 = data_4 - data_0

        return [data_2, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2]

    def op_subtract_3(self, data_2, data_6, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2):
    
        # EarlyReturn(0, 18)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_3 = data_6 - data_2

        return [data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3]

    def op_multiply_1(self, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3):
    
        # EarlyReturn(0, 19)

        # pd_op.multiply: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        multiply_1 = subtract_2 * subtract_3

        return [data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1]

    def op_clip_2(self, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1):
    
        # EarlyReturn(0, 20)

        # pd_op.clip: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(multiply_1, assign_1, assign_4)

        return [data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2]

    def op_subtract_4(self, data_1, data_3, data_5, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2):
    
        # EarlyReturn(0, 21)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_4 = data_5 - data_1

        return [data_3, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4]

    def op_subtract_5(self, data_3, data_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4):
    
        # EarlyReturn(0, 22)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_5 = data_7 - data_3

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5]

    def op_multiply_2(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5):
    
        # EarlyReturn(0, 23)

        # pd_op.multiply: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        multiply_2 = subtract_4 * subtract_5

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2]

    def op_clip_3(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2):
    
        # EarlyReturn(0, 24)

        # pd_op.clip: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32, 1xf32)
        clip_3 = paddle._C_ops.clip(multiply_2, assign_0, assign_3)

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3]

    def op_add_0(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3):
    
        # EarlyReturn(0, 25)

        # pd_op.add: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        add_0 = clip_2 + clip_3

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0]

    def op_subtract_6(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0):
    
        # EarlyReturn(0, 26)

        # pd_op.subtract: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        subtract_6 = add_0 - multiply_0

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, subtract_6]

    def op_full_2(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, subtract_6):
    
        # EarlyReturn(0, 27)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, subtract_6, full_2]

    def op_scale_0(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, subtract_6, full_2):
    
        # EarlyReturn(0, 28)

        # pd_op.scale: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_6, full_2, 1e-09, True)

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0]

    def op_divide_0(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0):
    
        # EarlyReturn(0, 29)

        # pd_op.divide: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        divide_0 = multiply_0 / scale_0

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0]

    def op_multiply_3(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0):
    
        # EarlyReturn(0, 30)

        # pd_op.multiply: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, -1x-1x-1x-1x-1xf32)
        multiply_3 = divide_0 * divide_0

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0, multiply_3]

    def op_full_3(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0, multiply_3):
    
        # EarlyReturn(0, 31)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], -1, paddle.float32, paddle.core.CPUPlace())

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0, multiply_3, full_3]

    def op_scale_1(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0, multiply_3, full_3):
    
        # EarlyReturn(0, 32)

        # pd_op.scale: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_3, full_3, 1, True)

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0, full_3, scale_1]

    def op_full_4(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0, full_3, scale_1):
    
        # EarlyReturn(0, 33)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 2.5, paddle.float32, paddle.core.CPUPlace())

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0, full_3, scale_1, full_4]

    def op_scale_2(self, maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, assign_0, assign_1, assign_2, full_1, assign_3, assign_4, assign_5, clip_0, subtract_1, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, clip_2, subtract_4, subtract_5, multiply_2, clip_3, add_0, full_2, scale_0, divide_0, full_3, scale_1, full_4):
    
        # EarlyReturn(0, 34)

        # pd_op.scale: (-1x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_4, 0, True)

        return [maximum_0, maximum_1, minimum_0, minimum_1, subtract_0, full_0, full_1, clip_0, subtract_1, assign_2, assign_5, clip_1, multiply_0, subtract_2, subtract_3, multiply_1, assign_1, assign_4, clip_2, subtract_4, subtract_5, multiply_2, assign_0, assign_3, clip_3, add_0, full_2, scale_0, divide_0, full_3, full_4, scale_2]

if True and not (IsCinnStageEnableDiff() and LastCINNStageFailed()):

    class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
        def prepare_data(self):
            self.inputs = [
                # data_0
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                # data_2
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                # data_4
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                # data_6
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                # data_1
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                # data_3
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                # data_5
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                # data_7
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]
            for input in self.inputs:
                input.stop_gradient = True

        def apply_to_static(self, net, use_cinn):
            build_strategy = paddle.static.BuildStrategy()
            input_spec = [
                # data_0
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                # data_2
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                # data_4
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                # data_6
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                # data_1
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                # data_3
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                # data_5
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                # data_7
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
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