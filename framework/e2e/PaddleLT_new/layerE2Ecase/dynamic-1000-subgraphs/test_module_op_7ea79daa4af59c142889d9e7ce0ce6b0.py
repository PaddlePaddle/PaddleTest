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
    env = {
        k:v
        for k, v in env.items()
        if v is not None
    }
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
        return True, f"last stage failed."
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
class TestTryRun(unittest.TestCase):
    def test_panic(self):
        if not AthenaTryRunEnabled():
            return
        if try_run_exit_code == 0:
            # All unittest cases passed.
            return
        if try_run_exit_code > 0:
            # program failed but not panic.
            return
        # program panicked.
        kOutputLimit = 65536
        message = try_run_stderr[-kOutputLimit:]
        raise RuntimeError(f"panicked. last {kOutputLimit} characters of stderr: \n{message}")

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
    return [43][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_207_0_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_0 = paddle._C_ops.pow(data_0, float('2'))

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_0 = paddle._C_ops.mean(pow_0, [1], True)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [28, 28]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_5 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_6 = full_int_array_0

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(mean_0, full_int_array_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [22, 784]

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_0, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_1 = paddle._C_ops.pow(data_1, float('2'))

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_1 = paddle._C_ops.mean(pow_1, [1], True)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(mean_1, assign_6, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_1, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_2 = paddle._C_ops.pow(data_2, float('2'))

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_2 = paddle._C_ops.mean(pow_2, [1], True)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(mean_2, assign_5, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_2, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_3 = paddle._C_ops.pow(data_3, float('2'))

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_3 = paddle._C_ops.mean(pow_3, [1], True)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(mean_3, assign_4, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_3, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_4 = paddle._C_ops.pow(data_4, float('2'))

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_4 = paddle._C_ops.mean(pow_4, [1], True)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(mean_4, assign_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_4, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_5 = paddle._C_ops.pow(data_5, float('2'))

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_5 = paddle._C_ops.mean(pow_5, [1], True)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(mean_5, assign_2, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_5, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_6 = paddle._C_ops.pow(data_6, float('2'))

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_6 = paddle._C_ops.mean(pow_6, [1], True)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(mean_6, assign_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_6, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_7 = paddle._C_ops.pow(data_7, float('2'))

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_7 = paddle._C_ops.mean(pow_7, [1], True)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(mean_7, assign_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_7, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32]) <- (22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32)
        combine_0 = [reshape_0, reshape_2, reshape_4, reshape_6, reshape_8, reshape_10, reshape_12, reshape_14]

        # pd_op.stack: (22x8x784xf32) <- ([22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)
        return pow_0, mean_0, full_int_array_0, pool2d_0, reshape_0, reshape_1, pow_1, mean_1, assign_6, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, assign_5, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, assign_4, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, assign_3, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, assign_2, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, assign_1, pool2d_6, reshape_12, reshape_13, pow_7, mean_7, assign_0, pool2d_7, reshape_14, reshape_15, stack_0



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

class Block_builtin_module_207_0_0(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        args = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_pow_0,
            self.op_mean_0,
            self.op_full_int_array_0,
            self.op_assign_0,
            self.op_assign_1,
            self.op_assign_2,
            self.op_assign_3,
            self.op_assign_4,
            self.op_assign_5,
            self.op_assign_6,
            self.op_pool2d_0,
            self.op_full_int_array_1,
            self.op_reshape_0,
            self.op_pow_1,
            self.op_mean_1,
            self.op_pool2d_1,
            self.op_reshape_1,
            self.op_pow_2,
            self.op_mean_2,
            self.op_pool2d_2,
            self.op_reshape_2,
            self.op_pow_3,
            self.op_mean_3,
            self.op_pool2d_3,
            self.op_reshape_3,
            self.op_pow_4,
            self.op_mean_4,
            self.op_pool2d_4,
            self.op_reshape_4,
            self.op_pow_5,
            self.op_mean_5,
            self.op_pool2d_5,
            self.op_reshape_5,
            self.op_pow_6,
            self.op_mean_6,
            self.op_pool2d_6,
            self.op_reshape_6,
            self.op_pow_7,
            self.op_mean_7,
            self.op_pool2d_7,
            self.op_reshape_7,
            self.op_combine_0,
            self.op_stack_0,
        ]

    def op_pow_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
    
        # EarlyReturn(0, 0)

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_0 = paddle._C_ops.pow(data_0, float('2'))

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0]

    def op_mean_0(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0):
    
        # EarlyReturn(0, 1)

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_0 = paddle._C_ops.mean(pow_0, [1], True)

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0]

    def op_full_int_array_0(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0):
    
        # EarlyReturn(0, 2)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [28, 28]

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0]

    def op_assign_0(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0):
    
        # EarlyReturn(0, 3)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0]

    def op_assign_1(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0):
    
        # EarlyReturn(0, 4)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1]

    def op_assign_2(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1):
    
        # EarlyReturn(0, 5)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2]

    def op_assign_3(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2):
    
        # EarlyReturn(0, 6)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_0

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3]

    def op_assign_4(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3):
    
        # EarlyReturn(0, 7)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_0

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4]

    def op_assign_5(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4):
    
        # EarlyReturn(0, 8)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_5 = full_int_array_0

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5]

    def op_assign_6(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5):
    
        # EarlyReturn(0, 9)

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_6 = full_int_array_0

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6]

    def op_pool2d_0(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6):
    
        # EarlyReturn(0, 10)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(mean_0, full_int_array_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0]

    def op_full_int_array_1(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0):
    
        # EarlyReturn(0, 11)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [22, 784]

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1]

    def op_reshape_0(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1):
    
        # EarlyReturn(0, 12)

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_0, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        return [data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1]

    def op_pow_1(self, data_1, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1):
    
        # EarlyReturn(0, 13)

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_1 = paddle._C_ops.pow(data_1, float('2'))

        return [data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1]

    def op_mean_1(self, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1):
    
        # EarlyReturn(0, 14)

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_1 = paddle._C_ops.mean(pow_1, [1], True)

        return [data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1]

    def op_pool2d_1(self, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1):
    
        # EarlyReturn(0, 15)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(mean_1, assign_6, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1]

    def op_reshape_1(self, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1):
    
        # EarlyReturn(0, 16)

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_1, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        return [data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3]

    def op_pow_2(self, data_2, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3):
    
        # EarlyReturn(0, 17)

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_2 = paddle._C_ops.pow(data_2, float('2'))

        return [data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2]

    def op_mean_2(self, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2):
    
        # EarlyReturn(0, 18)

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_2 = paddle._C_ops.mean(pow_2, [1], True)

        return [data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2]

    def op_pool2d_2(self, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2):
    
        # EarlyReturn(0, 19)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(mean_2, assign_5, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2]

    def op_reshape_2(self, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2):
    
        # EarlyReturn(0, 20)

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_2, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        return [data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5]

    def op_pow_3(self, data_3, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5):
    
        # EarlyReturn(0, 21)

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_3 = paddle._C_ops.pow(data_3, float('2'))

        return [data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3]

    def op_mean_3(self, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3):
    
        # EarlyReturn(0, 22)

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_3 = paddle._C_ops.mean(pow_3, [1], True)

        return [data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3]

    def op_pool2d_3(self, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3):
    
        # EarlyReturn(0, 23)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(mean_3, assign_4, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3]

    def op_reshape_3(self, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3):
    
        # EarlyReturn(0, 24)

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_3, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        return [data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7]

    def op_pow_4(self, data_4, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7):
    
        # EarlyReturn(0, 25)

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_4 = paddle._C_ops.pow(data_4, float('2'))

        return [data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4]

    def op_mean_4(self, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4):
    
        # EarlyReturn(0, 26)

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_4 = paddle._C_ops.mean(pow_4, [1], True)

        return [data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4]

    def op_pool2d_4(self, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4):
    
        # EarlyReturn(0, 27)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(mean_4, assign_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4]

    def op_reshape_4(self, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4):
    
        # EarlyReturn(0, 28)

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_4, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        return [data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9]

    def op_pow_5(self, data_5, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9):
    
        # EarlyReturn(0, 29)

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_5 = paddle._C_ops.pow(data_5, float('2'))

        return [data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5]

    def op_mean_5(self, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5):
    
        # EarlyReturn(0, 30)

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_5 = paddle._C_ops.mean(pow_5, [1], True)

        return [data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5]

    def op_pool2d_5(self, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5):
    
        # EarlyReturn(0, 31)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(mean_5, assign_2, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5]

    def op_reshape_5(self, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5):
    
        # EarlyReturn(0, 32)

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_5, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        return [data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11]

    def op_pow_6(self, data_6, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11):
    
        # EarlyReturn(0, 33)

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_6 = paddle._C_ops.pow(data_6, float('2'))

        return [data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6]

    def op_mean_6(self, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6):
    
        # EarlyReturn(0, 34)

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_6 = paddle._C_ops.mean(pow_6, [1], True)

        return [data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6]

    def op_pool2d_6(self, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6):
    
        # EarlyReturn(0, 35)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(mean_6, assign_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6]

    def op_reshape_6(self, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6):
    
        # EarlyReturn(0, 36)

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_6, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        return [data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13]

    def op_pow_7(self, data_7, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13):
    
        # EarlyReturn(0, 37)

        # pd_op.pow: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        pow_7 = paddle._C_ops.pow(data_7, float('2'))

        return [pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7]

    def op_mean_7(self, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7):
    
        # EarlyReturn(0, 38)

        # pd_op.mean: (-1x1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        mean_7 = paddle._C_ops.mean(pow_7, [1], True)

        return [pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7, mean_7]

    def op_pool2d_7(self, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7, mean_7):
    
        # EarlyReturn(0, 39)

        # pd_op.pool2d: (-1x1x28x28xf32) <- (-1x1x-1x-1xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(mean_7, assign_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        return [pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7, mean_7, pool2d_7]

    def op_reshape_7(self, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, full_int_array_1, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7, mean_7, pool2d_7):
    
        # EarlyReturn(0, 40)

        # pd_op.reshape: (22x784xf32, 0x-1x1x28x28xi64) <- (-1x1x28x28xf32, 2xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(pool2d_7, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        return [pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7, mean_7, pool2d_7, reshape_14, reshape_15]

    def op_combine_0(self, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7, mean_7, pool2d_7, reshape_14, reshape_15):
    
        # EarlyReturn(0, 41)

        # builtin.combine: ([22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32]) <- (22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32)
        combine_0 = [reshape_0, reshape_2, reshape_4, reshape_6, reshape_8, reshape_10, reshape_12, reshape_14]

        return [pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7, mean_7, pool2d_7, reshape_14, reshape_15, combine_0]

    def op_stack_0(self, pow_0, mean_0, full_int_array_0, assign_0, assign_1, assign_2, assign_3, assign_4, assign_5, assign_6, pool2d_0, reshape_0, reshape_1, pow_1, mean_1, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, pool2d_6, reshape_12, reshape_13, pow_7, mean_7, pool2d_7, reshape_14, reshape_15, combine_0):
    
        # EarlyReturn(0, 42)

        # pd_op.stack: (22x8x784xf32) <- ([22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32, 22x784xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 1)

        return [pow_0, mean_0, full_int_array_0, pool2d_0, reshape_0, reshape_1, pow_1, mean_1, assign_6, pool2d_1, reshape_2, reshape_3, pow_2, mean_2, assign_5, pool2d_2, reshape_4, reshape_5, pow_3, mean_3, assign_4, pool2d_3, reshape_6, reshape_7, pow_4, mean_4, assign_3, pool2d_4, reshape_8, reshape_9, pow_5, mean_5, assign_2, pool2d_5, reshape_10, reshape_11, pow_6, mean_6, assign_1, pool2d_6, reshape_12, reshape_13, pow_7, mean_7, assign_0, pool2d_7, reshape_14, reshape_15, stack_0]

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_207_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # data_0
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            # data_5
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            # data_6
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            # data_7
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # data_0
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            # data_5
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            # data_6
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            # data_7
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = Block_builtin_module_207_0_0()
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
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()