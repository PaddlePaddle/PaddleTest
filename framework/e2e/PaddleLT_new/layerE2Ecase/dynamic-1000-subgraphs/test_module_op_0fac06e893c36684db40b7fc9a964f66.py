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
        return True, f"last stage failed. stderr: {stderr}"
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
    return [21][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, data_0):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 1.11111, paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, 0, True)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_0 = paddle._C_ops.shape(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_0, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(slice_1, paddle.int64)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full([], 1, paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [cast_0, cast_1, full_1, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(stack_0, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 0.1, paddle.float32, paddle.framework._current_expected_place())

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(uniform_0, full_4)

        # pd_op.cast: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        cast_2 = paddle._C_ops.cast(scale_0, paddle.float32)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_3 = paddle._C_ops.cast(greater_equal_0, paddle.float32)

        # pd_op.multiply: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, -1x-1x1x1xf32)
        multiply_0 = cast_2 * cast_3
        return full_0, cast_2, cast_3, multiply_0



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

class Block_builtin_module_0_0_0(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        args = [data_0]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_full_0,
            self.op_scale_0,
            self.op_shape_0,
            self.op_full_int_array_0,
            self.op_full_int_array_1,
            self.op_slice_0,
            self.op_full_int_array_2,
            self.op_slice_1,
            self.op_cast_0,
            self.op_cast_1,
            self.op_full_1,
            self.op_combine_0,
            self.op_stack_0,
            self.op_full_2,
            self.op_full_3,
            self.op_uniform_0,
            self.op_full_4,
            self.op_greater_equal_0,
            self.op_cast_2,
            self.op_cast_3,
            self.op_multiply_0,
        ]

    def op_full_0(self, data_0):
    
        # EarlyReturn(0, 0)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], 1.11111, paddle.float32, paddle.core.CPUPlace())

        return [data_0, full_0]

    def op_scale_0(self, data_0, full_0):
    
        # EarlyReturn(0, 1)

        # pd_op.scale: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, 0, True)

        return [data_0, full_0, scale_0]

    def op_shape_0(self, data_0, full_0, scale_0):
    
        # EarlyReturn(0, 2)

        # pd_op.shape: (4xi32) <- (-1x-1x-1x-1xf32)
        shape_0 = paddle._C_ops.shape(data_0)

        return [full_0, scale_0, shape_0]

    def op_full_int_array_0(self, full_0, scale_0, shape_0):
    
        # EarlyReturn(0, 3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        return [full_0, scale_0, shape_0, full_int_array_0]

    def op_full_int_array_1(self, full_0, scale_0, shape_0, full_int_array_0):
    
        # EarlyReturn(0, 4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        return [full_0, scale_0, shape_0, full_int_array_0, full_int_array_1]

    def op_slice_0(self, full_0, scale_0, shape_0, full_int_array_0, full_int_array_1):
    
        # EarlyReturn(0, 5)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [full_0, scale_0, shape_0, full_int_array_1, slice_0]

    def op_full_int_array_2(self, full_0, scale_0, shape_0, full_int_array_1, slice_0):
    
        # EarlyReturn(0, 6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        return [full_0, scale_0, shape_0, full_int_array_1, slice_0, full_int_array_2]

    def op_slice_1(self, full_0, scale_0, shape_0, full_int_array_1, slice_0, full_int_array_2):
    
        # EarlyReturn(0, 7)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_0, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [full_0, scale_0, slice_0, slice_1]

    def op_cast_0(self, full_0, scale_0, slice_0, slice_1):
    
        # EarlyReturn(0, 8)

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)

        return [full_0, scale_0, slice_1, cast_0]

    def op_cast_1(self, full_0, scale_0, slice_1, cast_0):
    
        # EarlyReturn(0, 9)

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(slice_1, paddle.int64)

        return [full_0, scale_0, cast_0, cast_1]

    def op_full_1(self, full_0, scale_0, cast_0, cast_1):
    
        # EarlyReturn(0, 10)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full([], 1, paddle.int64, paddle.core.CPUPlace())

        return [full_0, scale_0, cast_0, cast_1, full_1]

    def op_combine_0(self, full_0, scale_0, cast_0, cast_1, full_1):
    
        # EarlyReturn(0, 11)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [cast_0, cast_1, full_1, full_1]

        return [full_0, scale_0, combine_0]

    def op_stack_0(self, full_0, scale_0, combine_0):
    
        # EarlyReturn(0, 12)

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        return [full_0, scale_0, stack_0]

    def op_full_2(self, full_0, scale_0, stack_0):
    
        # EarlyReturn(0, 13)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

        return [full_0, scale_0, stack_0, full_2]

    def op_full_3(self, full_0, scale_0, stack_0, full_2):
    
        # EarlyReturn(0, 14)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

        return [full_0, scale_0, stack_0, full_2, full_3]

    def op_uniform_0(self, full_0, scale_0, stack_0, full_2, full_3):
    
        # EarlyReturn(0, 15)

        # pd_op.uniform: (-1x-1x1x1xf32) <- (4xi64, 1xf32, 1xf32)
        uniform_0 = paddle._C_ops.uniform(stack_0, paddle.float32, full_2, full_3, 0, paddle.framework._current_expected_place())

        return [full_0, scale_0, uniform_0]

    def op_full_4(self, full_0, scale_0, uniform_0):
    
        # EarlyReturn(0, 16)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], 0.1, paddle.float32, paddle.framework._current_expected_place())

        return [full_0, scale_0, uniform_0, full_4]

    def op_greater_equal_0(self, full_0, scale_0, uniform_0, full_4):
    
        # EarlyReturn(0, 17)

        # pd_op.greater_equal: (-1x-1x1x1xb) <- (-1x-1x1x1xf32, 1xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(uniform_0, full_4)

        return [full_0, scale_0, greater_equal_0]

    def op_cast_2(self, full_0, scale_0, greater_equal_0):
    
        # EarlyReturn(0, 18)

        # pd_op.cast: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        cast_2 = paddle._C_ops.cast(scale_0, paddle.float32)

        return [full_0, greater_equal_0, cast_2]

    def op_cast_3(self, full_0, greater_equal_0, cast_2):
    
        # EarlyReturn(0, 19)

        # pd_op.cast: (-1x-1x1x1xf32) <- (-1x-1x1x1xb)
        cast_3 = paddle._C_ops.cast(greater_equal_0, paddle.float32)

        return [full_0, cast_2, cast_3]

    def op_multiply_0(self, full_0, cast_2, cast_3):
    
        # EarlyReturn(0, 20)

        # pd_op.multiply: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32, -1x-1x1x1xf32)
        multiply_0 = cast_2 * cast_3

        return [full_0, cast_2, cast_3, multiply_0]

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # data_0
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # data_0
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
        net = Block_builtin_module_0_0_0()
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
                # program paniced.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()