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
    return [6][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_84_0_0(self, parameter_2, parameter_1, parameter_0, parameter_3, data_0):

        # pd_op.layer_norm: (-1x-1x768xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(data_0, parameter_0, parameter_1, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x-1x2304xf32) <- (-1x-1x768xf32, 768x2304xf32)
        matmul_0 = paddle.matmul(layer_norm_0, parameter_2, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x-1x2304xf32) <- (-1x-1x2304xf32, 2304xf32)
        add_0 = matmul_0 + parameter_3

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x-1x768xf32, -1x-1x768xf32, -1x-1x768xf32]) <- (-1x-1x2304xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add_0, 3, full_0)

        # builtin.split: (-1x-1x768xf32, -1x-1x768xf32, -1x-1x768xf32) <- ([-1x-1x768xf32, -1x-1x768xf32, -1x-1x768xf32])
        split_0, split_1, split_2, = split_with_num_0
        return layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, full_0, split_0, split_1, split_2



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

class Block_builtin_module_84_0_0(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_2, parameter_1, parameter_0, parameter_3, data_0):
        args = [parameter_2, parameter_1, parameter_0, parameter_3, data_0]
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
            self.op_full_0,
            self.op_split_with_num_0,
            self.op_split_0,
        ]

    def op_layer_norm_0(self, parameter_2, parameter_1, parameter_0, parameter_3, data_0):
    
        # EarlyReturn(0, 0)

        # pd_op.layer_norm: (-1x-1x768xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x768xf32, 768xf32, 768xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(data_0, parameter_0, parameter_1, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        return [parameter_2, parameter_3, layer_norm_0, layer_norm_1, layer_norm_2]

    def op_matmul_0(self, parameter_2, parameter_3, layer_norm_0, layer_norm_1, layer_norm_2):
    
        # EarlyReturn(0, 1)

        # pd_op.matmul: (-1x-1x2304xf32) <- (-1x-1x768xf32, 768x2304xf32)
        matmul_0 = paddle.matmul(layer_norm_0, parameter_2, transpose_x=False, transpose_y=False)

        return [parameter_3, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0]

    def op_add_0(self, parameter_3, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0):
    
        # EarlyReturn(0, 2)

        # pd_op.add: (-1x-1x2304xf32) <- (-1x-1x2304xf32, 2304xf32)
        add_0 = matmul_0 + parameter_3

        return [layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, add_0]

    def op_full_0(self, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, add_0):
    
        # EarlyReturn(0, 3)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        return [layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, add_0, full_0]

    def op_split_with_num_0(self, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, add_0, full_0):
    
        # EarlyReturn(0, 4)

        # pd_op.split_with_num: ([-1x-1x768xf32, -1x-1x768xf32, -1x-1x768xf32]) <- (-1x-1x2304xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(add_0, 3, full_0)

        return [layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, full_0, split_with_num_0]

    def op_split_0(self, layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, full_0, split_with_num_0):
    
        # EarlyReturn(0, 5)

        # builtin.split: (-1x-1x768xf32, -1x-1x768xf32, -1x-1x768xf32) <- ([-1x-1x768xf32, -1x-1x768xf32, -1x-1x768xf32])
        split_0, split_1, split_2, = split_with_num_0

        return [layer_norm_0, layer_norm_1, layer_norm_2, matmul_0, full_0, split_0, split_1, split_2]

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_84_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_2
            paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([2304], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_2
            paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[2304], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = Block_builtin_module_84_0_0()
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