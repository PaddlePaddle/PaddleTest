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
    return [20][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, data_0, data_1, data_2):

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_0 = paddle.transpose(data_0, perm=[0, 2, 3, 1])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [1, -1, 128]

        # pd_op.reshape: (1x-1x128xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_0, reshape_1 = paddle.reshape(transpose_0, full_int_array_0), None

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.unsqueeze: (-1x-1x1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        unsqueeze_0, unsqueeze_1 = paddle.unsqueeze(data_1, full_int_array_1), None

        # pd_op.full: (1x500x1xi64) <- ()
        full_0 = paddle._C_ops.full([1, 500, 1], 0, paddle.int64, paddle.framework._current_expected_place())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1x500x1xi64]) <- (1x500x1xi64)
        combine_0 = [full_0]

        # pd_op.concat: (1x500x1xi64) <- ([1x500x1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], 2, paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1x500x1xi64, -1x-1x1xi64]) <- (1x500x1xi64, -1x-1x1xi64)
        combine_1 = [concat_0, unsqueeze_0]

        # pd_op.concat: (1x500x2xi64) <- ([1x500x1xi64, -1x-1x1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)

        # pd_op.gather_nd: (1x500x128xf32) <- (1x-1x128xf32, 1x500x2xi64)
        gather_nd_0 = paddle.gather_nd(reshape_0, concat_1)

        # pd_op.unsqueeze: (-1x-1x1xi32, 0x-1x-1xi32) <- (-1x-1xi32, 1xi64)
        unsqueeze_2, unsqueeze_3 = paddle.unsqueeze(data_2, full_int_array_1), None

        # pd_op.expand_as: (1x500x128xi32) <- (-1x-1x1xi32, None)
        expand_as_0 = paddle._C_ops.expand_as(unsqueeze_2, None, [1, 500, 128])

        # pd_op.full: (xi32) <- ()
        full_3 = paddle._C_ops.full([], 0, paddle.int32, paddle.framework._current_expected_place())

        # pd_op.greater_than: (1x500x128xb) <- (1x500x128xi32, xi32)
        greater_than_0 = expand_as_0 > full_3

        # pd_op.masked_select: (-1xf32) <- (1x500x128xf32, 1x500x128xb)
        masked_select_0 = paddle._C_ops.masked_select(gather_nd_0, greater_than_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 128]

        # pd_op.reshape: (-1x128xf32, 0x-1xi64) <- (-1xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(masked_select_0, full_int_array_2), None
        return reshape_0, reshape_1, gather_nd_0, greater_than_0, reshape_3, concat_1, reshape_2



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

    def forward(self, data_0, data_1, data_2):
        args = [data_0, data_1, data_2]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_transpose_0,
            self.op_full_int_array_0,
            self.op_reshape_0,
            self.op_full_int_array_1,
            self.op_unsqueeze_0,
            self.op_full_0,
            self.op_full_1,
            self.op_combine_0,
            self.op_concat_0,
            self.op_full_2,
            self.op_combine_1,
            self.op_concat_1,
            self.op_gather_nd_0,
            self.op_unsqueeze_1,
            self.op_expand_as_0,
            self.op_full_3,
            self.op_greater_than_0,
            self.op_masked_select_0,
            self.op_full_int_array_2,
            self.op_reshape_1,
        ]

    def op_transpose_0(self, data_0, data_1, data_2):
    
        # EarlyReturn(0, 0)

        # pd_op.transpose: (-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        transpose_0 = paddle.transpose(data_0, perm=[0, 2, 3, 1])

        return [data_1, data_2, transpose_0]

    def op_full_int_array_0(self, data_1, data_2, transpose_0):
    
        # EarlyReturn(0, 1)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [1, -1, 128]

        return [data_1, data_2, transpose_0, full_int_array_0]

    def op_reshape_0(self, data_1, data_2, transpose_0, full_int_array_0):
    
        # EarlyReturn(0, 2)

        # pd_op.reshape: (1x-1x128xf32, 0x-1x-1x-1x-1xi64) <- (-1x-1x-1x-1xf32, 3xi64)
        reshape_0, reshape_1 = paddle.reshape(transpose_0, full_int_array_0), None

        return [data_1, data_2, reshape_0, reshape_1]

    def op_full_int_array_1(self, data_1, data_2, reshape_0, reshape_1):
    
        # EarlyReturn(0, 3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        return [data_1, data_2, reshape_0, reshape_1, full_int_array_1]

    def op_unsqueeze_0(self, data_1, data_2, reshape_0, reshape_1, full_int_array_1):
    
        # EarlyReturn(0, 4)

        # pd_op.unsqueeze: (-1x-1x1xi64, 0x-1x-1xi64) <- (-1x-1xi64, 1xi64)
        unsqueeze_0, unsqueeze_1 = paddle.unsqueeze(data_1, full_int_array_1), None

        return [data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0]

    def op_full_0(self, data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0):
    
        # EarlyReturn(0, 5)

        # pd_op.full: (1x500x1xi64) <- ()
        full_0 = paddle._C_ops.full([1, 500, 1], 0, paddle.int64, paddle.framework._current_expected_place())

        return [data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, full_0]

    def op_full_1(self, data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, full_0):
    
        # EarlyReturn(0, 6)

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

        return [data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, full_0, full_1]

    def op_combine_0(self, data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, full_0, full_1):
    
        # EarlyReturn(0, 7)

        # builtin.combine: ([1x500x1xi64]) <- (1x500x1xi64)
        combine_0 = [full_0]

        return [data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, full_1, combine_0]

    def op_concat_0(self, data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, full_1, combine_0):
    
        # EarlyReturn(0, 8)

        # pd_op.concat: (1x500x1xi64) <- ([1x500x1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)

        return [data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, concat_0]

    def op_full_2(self, data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, concat_0):
    
        # EarlyReturn(0, 9)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], 2, paddle.int32, paddle.core.CPUPlace())

        return [data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, concat_0, full_2]

    def op_combine_1(self, data_2, reshape_0, reshape_1, full_int_array_1, unsqueeze_0, concat_0, full_2):
    
        # EarlyReturn(0, 10)

        # builtin.combine: ([1x500x1xi64, -1x-1x1xi64]) <- (1x500x1xi64, -1x-1x1xi64)
        combine_1 = [concat_0, unsqueeze_0]

        return [data_2, reshape_0, reshape_1, full_int_array_1, full_2, combine_1]

    def op_concat_1(self, data_2, reshape_0, reshape_1, full_int_array_1, full_2, combine_1):
    
        # EarlyReturn(0, 11)

        # pd_op.concat: (1x500x2xi64) <- ([1x500x1xi64, -1x-1x1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)

        return [data_2, reshape_0, reshape_1, full_int_array_1, concat_1]

    def op_gather_nd_0(self, data_2, reshape_0, reshape_1, full_int_array_1, concat_1):
    
        # EarlyReturn(0, 12)

        # pd_op.gather_nd: (1x500x128xf32) <- (1x-1x128xf32, 1x500x2xi64)
        gather_nd_0 = paddle.gather_nd(reshape_0, concat_1)

        return [data_2, reshape_0, reshape_1, full_int_array_1, concat_1, gather_nd_0]

    def op_unsqueeze_1(self, data_2, reshape_0, reshape_1, full_int_array_1, concat_1, gather_nd_0):
    
        # EarlyReturn(0, 13)

        # pd_op.unsqueeze: (-1x-1x1xi32, 0x-1x-1xi32) <- (-1x-1xi32, 1xi64)
        unsqueeze_2, unsqueeze_3 = paddle.unsqueeze(data_2, full_int_array_1), None

        return [reshape_0, reshape_1, concat_1, gather_nd_0, unsqueeze_2]

    def op_expand_as_0(self, reshape_0, reshape_1, concat_1, gather_nd_0, unsqueeze_2):
    
        # EarlyReturn(0, 14)

        # pd_op.expand_as: (1x500x128xi32) <- (-1x-1x1xi32, None)
        expand_as_0 = paddle._C_ops.expand_as(unsqueeze_2, None, [1, 500, 128])

        return [reshape_0, reshape_1, concat_1, gather_nd_0, expand_as_0]

    def op_full_3(self, reshape_0, reshape_1, concat_1, gather_nd_0, expand_as_0):
    
        # EarlyReturn(0, 15)

        # pd_op.full: (xi32) <- ()
        full_3 = paddle._C_ops.full([], 0, paddle.int32, paddle.framework._current_expected_place())

        return [reshape_0, reshape_1, concat_1, gather_nd_0, expand_as_0, full_3]

    def op_greater_than_0(self, reshape_0, reshape_1, concat_1, gather_nd_0, expand_as_0, full_3):
    
        # EarlyReturn(0, 16)

        # pd_op.greater_than: (1x500x128xb) <- (1x500x128xi32, xi32)
        greater_than_0 = expand_as_0 > full_3

        return [reshape_0, reshape_1, concat_1, gather_nd_0, greater_than_0]

    def op_masked_select_0(self, reshape_0, reshape_1, concat_1, gather_nd_0, greater_than_0):
    
        # EarlyReturn(0, 17)

        # pd_op.masked_select: (-1xf32) <- (1x500x128xf32, 1x500x128xb)
        masked_select_0 = paddle._C_ops.masked_select(gather_nd_0, greater_than_0)

        return [reshape_0, reshape_1, concat_1, gather_nd_0, greater_than_0, masked_select_0]

    def op_full_int_array_2(self, reshape_0, reshape_1, concat_1, gather_nd_0, greater_than_0, masked_select_0):
    
        # EarlyReturn(0, 18)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 128]

        return [reshape_0, reshape_1, concat_1, gather_nd_0, greater_than_0, masked_select_0, full_int_array_2]

    def op_reshape_1(self, reshape_0, reshape_1, concat_1, gather_nd_0, greater_than_0, masked_select_0, full_int_array_2):
    
        # EarlyReturn(0, 19)

        # pd_op.reshape: (-1x128xf32, 0x-1xi64) <- (-1xf32, 2xi64)
        reshape_2, reshape_3 = paddle.reshape(masked_select_0, full_int_array_2), None

        return [reshape_0, reshape_1, gather_nd_0, greater_than_0, reshape_3, concat_1, reshape_2]

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # data_0
            paddle.uniform([1, 128, 80, 144], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            # data_2
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # data_0
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            # data_2
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()