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
    return [59][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_459_0_0(self, parameter_24, parameter_22, parameter_20, parameter_18, constant_3, parameter_16, constant_2, parameter_13, parameter_12, parameter_11, parameter_10, parameter_9, parameter_8, constant_1, parameter_7, parameter_6, parameter_5, parameter_4, parameter_3, parameter_2, parameter_1, parameter_0, constant_0, parameter_14, parameter_15, parameter_17, parameter_19, parameter_21, parameter_23, feed_0):

        # pd_op.cast: (-1x400x100xf16) <- (-1x400x100xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.unsqueeze_: (-1x400x1x100xf16, None) <- (-1x400x100xf16, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(cast_0, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x1x100xf16) <- (-1x400x1x100xf16, 256x100x1x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(unsqueeze__0, parameter_0, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.add_: (-1x256x1x100xf16) <- (-1x256x1x100xf16, 1x256x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_0, parameter_1)

        # pd_op.squeeze_: (-1x256x100xf16, None) <- (-1x256x1x100xf16, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__0, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x256x100xf16) <- (-1x256x100xf16)
        relu__0 = paddle._C_ops.relu_(squeeze__0)

        # pd_op.unsqueeze_: (-1x256x1x100xf16, None) <- (-1x256x100xf16, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu__0, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x1x100xf16) <- (-1x256x1x100xf16, 256x64x1x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(unsqueeze__2, parameter_2, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.add_: (-1x256x1x100xf16) <- (-1x256x1x100xf16, 1x256x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_1, parameter_3)

        # pd_op.squeeze_: (-1x256x100xf16, None) <- (-1x256x1x100xf16, 1xi64)
        squeeze__2, squeeze__3 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__1, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x256x100xf16) <- (-1x256x100xf16)
        relu__1 = paddle._C_ops.relu_(squeeze__2)

        # pd_op.unsqueeze: (-1x256x1x100xf16, None) <- (-1x256x100xf16, 1xi64)
        unsqueeze_0, unsqueeze_1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(relu__1, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x1x100xf16) <- (-1x256x1x100xf16, 256x64x1x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(unsqueeze_0, parameter_4, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.add_: (-1x256x1x100xf16) <- (-1x256x1x100xf16, 1x256x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_2, parameter_5)

        # pd_op.squeeze_: (-1x256x100xf16, None) <- (-1x256x1x100xf16, 1xi64)
        squeeze__4, squeeze__5 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__2, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x256x100xf16) <- (-1x256x100xf16)
        relu__2 = paddle._C_ops.relu_(squeeze__4)

        # pd_op.unsqueeze_: (-1x256x1x100xf16, None) <- (-1x256x100xf16, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu__2, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x1x1x100xf16) <- (-1x256x1x100xf16, 1x256x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(unsqueeze__4, parameter_6, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1x1x100xf16) <- (-1x1x1x100xf16, 1x1x1x1xf16)
        add__3 = paddle._C_ops.add_(conv2d_3, parameter_7)

        # pd_op.squeeze_: (-1x1x100xf16, None) <- (-1x1x1x100xf16, 1xi64)
        squeeze__6, squeeze__7 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__3, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x1x100xf16) <- (-1x1x100xf16)
        sigmoid__0 = paddle._C_ops.sigmoid_(squeeze__6)

        # pd_op.squeeze_: (-1x100xf16, None) <- (-1x1x100xf16, 1xi64)
        squeeze__8, squeeze__9 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(sigmoid__0, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unsqueeze: (-1x256x1x100xf16, None) <- (-1x256x100xf16, 1xi64)
        unsqueeze_2, unsqueeze_3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(relu__1, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x1x100xf16) <- (-1x256x1x100xf16, 256x64x1x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(unsqueeze_2, parameter_8, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.add_: (-1x256x1x100xf16) <- (-1x256x1x100xf16, 1x256x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_4, parameter_9)

        # pd_op.squeeze_: (-1x256x100xf16, None) <- (-1x256x1x100xf16, 1xi64)
        squeeze__10, squeeze__11 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__4, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x256x100xf16) <- (-1x256x100xf16)
        relu__3 = paddle._C_ops.relu_(squeeze__10)

        # pd_op.unsqueeze_: (-1x256x1x100xf16, None) <- (-1x256x100xf16, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu__3, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x1x1x100xf16) <- (-1x256x1x100xf16, 1x256x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(unsqueeze__6, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x1x1x100xf16) <- (-1x1x1x100xf16, 1x1x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_5, parameter_11)

        # pd_op.squeeze_: (-1x1x100xf16, None) <- (-1x1x1x100xf16, 1xi64)
        squeeze__12, squeeze__13 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__5, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x1x100xf16) <- (-1x1x100xf16)
        sigmoid__1 = paddle._C_ops.sigmoid_(squeeze__12)

        # pd_op.squeeze_: (-1x100xf16, None) <- (-1x1x100xf16, 1xi64)
        squeeze__14, squeeze__15 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(sigmoid__1, constant_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unsqueeze_: (-1x256x1x100xf16, None) <- (-1x256x100xf16, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu__1, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x1x100xf16) <- (-1x256x1x100xf16, 128x256x1x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(unsqueeze__8, parameter_12, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x1x100xf16) <- (-1x128x1x100xf16, 1x128x1x1xf16)
        add__6 = paddle._C_ops.add_(conv2d_6, parameter_13)

        # pd_op.squeeze_: (-1x128x100xf16, None) <- (-1x128x1x100xf16, 1xi64)
        squeeze__16, squeeze__17 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__6, constant_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x128x100xf16) <- (-1x128x100xf16)
        relu__4 = paddle._C_ops.relu_(squeeze__16)

        # pd_op.matmul: (-1x128x320000xf16) <- (-1x128x100xf16, 100x320000xf16)
        matmul_0 = paddle.matmul(relu__4, parameter_14, transpose_x=False, transpose_y=False)

        # pd_op.reshape_: (-1x128x-1x100x100xf16, 0x-1x128x320000xf16) <- (-1x128x320000xf16, 5xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_0, constant_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv3d: (-1x512x-1x100x100xf16) <- (-1x128x-1x100x100xf16, 512x128x32x1x1xf16)
        conv3d_0 = paddle._C_ops.conv3d(reshape__0, parameter_15, [32, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

        # pd_op.add: (-1x512x-1x100x100xf16) <- (-1x512x-1x100x100xf16, 1x512x1x1x1xf16)
        add_0 = conv3d_0 + parameter_16

        # pd_op.relu: (-1x512x-1x100x100xf16) <- (-1x512x-1x100x100xf16)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.squeeze_: (-1x512x100x100xf16, None) <- (-1x512x-1x100x100xf16, 1xi64)
        squeeze__18, squeeze__19 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(relu_0, constant_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x100x100xf16) <- (-1x512x100x100xf16, 128x512x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(squeeze__18, parameter_17, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x100x100xf16) <- (-1x128x100x100xf16, 1x128x1x1xf16)
        add__7 = paddle._C_ops.add_(conv2d_7, parameter_18)

        # pd_op.relu_: (-1x128x100x100xf16) <- (-1x128x100x100xf16)
        relu__5 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x128x100x100xf16) <- (-1x128x100x100xf16, 128x128x3x3xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu__5, parameter_19, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x100x100xf16) <- (-1x128x100x100xf16, 1x128x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_8, parameter_20)

        # pd_op.relu_: (-1x128x100x100xf16) <- (-1x128x100x100xf16)
        relu__6 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x128x100x100xf16) <- (-1x128x100x100xf16, 128x128x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__6, parameter_21, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x100x100xf16) <- (-1x128x100x100xf16, 1x128x1x1xf16)
        add__9 = paddle._C_ops.add_(conv2d_9, parameter_22)

        # pd_op.relu_: (-1x128x100x100xf16) <- (-1x128x100x100xf16)
        relu__7 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x2x100x100xf16) <- (-1x128x100x100xf16, 2x128x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu__7, parameter_23, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x2x100x100xf16) <- (-1x2x100x100xf16, 1x2x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_10, parameter_24)

        # pd_op.sigmoid_: (-1x2x100x100xf16) <- (-1x2x100x100xf16)
        sigmoid__2 = paddle._C_ops.sigmoid_(add__10)

        # pd_op.cast: (-1x2x100x100xf32) <- (-1x2x100x100xf16)
        cast_1 = paddle._C_ops.cast(sigmoid__2, paddle.float32)

        # pd_op.cast: (-1x100xf32) <- (-1x100xf16)
        cast_2 = paddle._C_ops.cast(squeeze__8, paddle.float32)

        # pd_op.cast: (-1x100xf32) <- (-1x100xf16)
        cast_3 = paddle._C_ops.cast(squeeze__14, paddle.float32)
        return cast_1, cast_2, cast_3



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

class ModuleOp(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_24, parameter_22, parameter_20, parameter_18, constant_3, parameter_16, constant_2, parameter_13, parameter_12, parameter_11, parameter_10, parameter_9, parameter_8, constant_1, parameter_7, parameter_6, parameter_5, parameter_4, parameter_3, parameter_2, parameter_1, parameter_0, constant_0, parameter_14, parameter_15, parameter_17, parameter_19, parameter_21, parameter_23, feed_0):
        return self.builtin_module_459_0_0(parameter_24, parameter_22, parameter_20, parameter_18, constant_3, parameter_16, constant_2, parameter_13, parameter_12, parameter_11, parameter_10, parameter_9, parameter_8, constant_1, parameter_7, parameter_6, parameter_5, parameter_4, parameter_3, parameter_2, parameter_1, parameter_0, constant_0, parameter_14, parameter_15, parameter_17, parameter_19, parameter_21, parameter_23, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_459_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_24
            paddle.uniform([1, 2, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_22
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_3
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            # parameter_16
            paddle.uniform([1, 512, 1, 1, 1], dtype='float16', min=0, max=0.5),
            # constant_2
            paddle.to_tensor([0, 0, -1, 100, 100], dtype='int64').reshape([5]),
            # parameter_13
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_12
            paddle.uniform([128, 256, 1, 3], dtype='float16', min=0, max=0.5),
            # parameter_11
            paddle.uniform([1, 1, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_10
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_8
            paddle.uniform([256, 64, 1, 3], dtype='float16', min=0, max=0.5),
            # constant_1
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            # parameter_7
            paddle.uniform([1, 1, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_6
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_5
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([256, 64, 1, 3], dtype='float16', min=0, max=0.5),
            # parameter_3
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_2
            paddle.uniform([256, 64, 1, 3], dtype='float16', min=0, max=0.5),
            # parameter_1
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_0
            paddle.uniform([256, 100, 1, 3], dtype='float16', min=0, max=0.5),
            # constant_0
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
            # parameter_14
            paddle.uniform([100, 320000], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([512, 128, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_17
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_21
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([2, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 400, 100], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_24
            paddle.static.InputSpec(shape=[1, 2, 1, 1], dtype='float16'),
            # parameter_22
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # constant_3
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_16
            paddle.static.InputSpec(shape=[1, 512, 1, 1, 1], dtype='float16'),
            # constant_2
            paddle.static.InputSpec(shape=[5], dtype='int64'),
            # parameter_13
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            # parameter_12
            paddle.static.InputSpec(shape=[128, 256, 1, 3], dtype='float16'),
            # parameter_11
            paddle.static.InputSpec(shape=[1, 1, 1, 1], dtype='float16'),
            # parameter_10
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_8
            paddle.static.InputSpec(shape=[256, 64, 1, 3], dtype='float16'),
            # constant_1
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_7
            paddle.static.InputSpec(shape=[1, 1, 1, 1], dtype='float16'),
            # parameter_6
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_5
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[256, 64, 1, 3], dtype='float16'),
            # parameter_3
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_2
            paddle.static.InputSpec(shape=[256, 64, 1, 3], dtype='float16'),
            # parameter_1
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float16'),
            # parameter_0
            paddle.static.InputSpec(shape=[256, 100, 1, 3], dtype='float16'),
            # constant_0
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            # parameter_14
            paddle.static.InputSpec(shape=[100, 320000], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[512, 128, 32, 1, 1], dtype='float16'),
            # parameter_17
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_21
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[2, 128, 1, 1], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 400, 100], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = ModuleOp()
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