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
    return [15][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, parameter_0, parameter_1, data_0):

        # pd_op.shape: (3xi32) <- (-1x-1x96xf32)
        shape_0 = paddle._C_ops.shape(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_0, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.layer_norm: (-1x-1x96xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x96xf32, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = paddle._C_ops.layer_norm(data_0, parameter_0, parameter_1, 1e-05, 2), None, None

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full([], 56, paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full([], 96, paddle.int64, paddle.core.CPUPlace())

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [cast_0, full_0, full_0, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (-1x56x56x96xf32, 0x-1x-1x96xi64) <- (-1x-1x96xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(layer_norm_0, stack_0), None

        # pd_op.shape: (4xi32) <- (-1x56x56x96xf32)
        shape_1 = paddle._C_ops.shape(reshape_0)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_1, [0], full_int_array_0, full_int_array_1, [1], [0])
        return layer_norm_1, layer_norm_2, reshape_1, reshape_0, slice_0



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

    def forward(self, parameter_0, parameter_1, data_0):
        args = [parameter_0, parameter_1, data_0]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_shape_0,
            self.op_full_int_array_0,
            self.op_full_int_array_1,
            self.op_slice_0,
            self.op_full_int_array_2,
            self.op_slice_1,
            self.op_layer_norm_0,
            self.op_cast_0,
            self.op_full_0,
            self.op_full_1,
            self.op_combine_0,
            self.op_stack_0,
            self.op_reshape_0,
            self.op_shape_1,
            self.op_slice_2,
        ]

    def op_shape_0(self, parameter_0, parameter_1, data_0):
    
        # EarlyReturn(0, 0)

        # pd_op.shape: (3xi32) <- (-1x-1x96xf32)
        shape_0 = paddle._C_ops.shape(data_0)

        return [parameter_0, parameter_1, data_0, shape_0]

    def op_full_int_array_0(self, parameter_0, parameter_1, data_0, shape_0):
    
        # EarlyReturn(0, 1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        return [parameter_0, parameter_1, data_0, shape_0, full_int_array_0]

    def op_full_int_array_1(self, parameter_0, parameter_1, data_0, shape_0, full_int_array_0):
    
        # EarlyReturn(0, 2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        return [parameter_0, parameter_1, data_0, shape_0, full_int_array_0, full_int_array_1]

    def op_slice_0(self, parameter_0, parameter_1, data_0, shape_0, full_int_array_0, full_int_array_1):
    
        # EarlyReturn(0, 3)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [parameter_0, parameter_1, data_0, shape_0, full_int_array_0, full_int_array_1, slice_0]

    def op_full_int_array_2(self, parameter_0, parameter_1, data_0, shape_0, full_int_array_0, full_int_array_1, slice_0):
    
        # EarlyReturn(0, 4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        return [parameter_0, parameter_1, data_0, shape_0, full_int_array_0, full_int_array_1, slice_0, full_int_array_2]

    def op_slice_1(self, parameter_0, parameter_1, data_0, shape_0, full_int_array_0, full_int_array_1, slice_0, full_int_array_2):
    
        # EarlyReturn(0, 5)

        # pd_op.slice: (xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_0, [0], full_int_array_1, full_int_array_2, [1], [0])

        return [parameter_0, parameter_1, data_0, full_int_array_0, full_int_array_1, slice_0]

    def op_layer_norm_0(self, parameter_0, parameter_1, data_0, full_int_array_0, full_int_array_1, slice_0):
    
        # EarlyReturn(0, 6)

        # pd_op.layer_norm: (-1x-1x96xf32, -1x-1xf32, -1x-1xf32) <- (-1x-1x96xf32, 96xf32, 96xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = paddle._C_ops.layer_norm(data_0, parameter_0, parameter_1, 1e-05, 2), None, None

        return [full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2]

    def op_cast_0(self, full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2):
    
        # EarlyReturn(0, 7)

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)

        return [full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, cast_0]

    def op_full_0(self, full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, cast_0):
    
        # EarlyReturn(0, 8)

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full([], 56, paddle.int64, paddle.core.CPUPlace())

        return [full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, cast_0, full_0]

    def op_full_1(self, full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, cast_0, full_0):
    
        # EarlyReturn(0, 9)

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full([], 96, paddle.int64, paddle.core.CPUPlace())

        return [full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, cast_0, full_0, full_1]

    def op_combine_0(self, full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, cast_0, full_0, full_1):
    
        # EarlyReturn(0, 10)

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [cast_0, full_0, full_0, full_1]

        return [full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, combine_0]

    def op_stack_0(self, full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, combine_0):
    
        # EarlyReturn(0, 11)

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        return [full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, stack_0]

    def op_reshape_0(self, full_int_array_0, full_int_array_1, slice_0, layer_norm_0, layer_norm_1, layer_norm_2, stack_0):
    
        # EarlyReturn(0, 12)

        # pd_op.reshape: (-1x56x56x96xf32, 0x-1x-1x96xi64) <- (-1x-1x96xf32, 4xi64)
        reshape_0, reshape_1 = paddle.reshape(layer_norm_0, stack_0), None

        return [full_int_array_0, full_int_array_1, slice_0, layer_norm_1, layer_norm_2, reshape_0, reshape_1]

    def op_shape_1(self, full_int_array_0, full_int_array_1, slice_0, layer_norm_1, layer_norm_2, reshape_0, reshape_1):
    
        # EarlyReturn(0, 13)

        # pd_op.shape: (4xi32) <- (-1x56x56x96xf32)
        shape_1 = paddle._C_ops.shape(reshape_0)

        return [full_int_array_0, full_int_array_1, slice_0, layer_norm_1, layer_norm_2, reshape_0, reshape_1, shape_1]

    def op_slice_2(self, full_int_array_0, full_int_array_1, slice_0, layer_norm_1, layer_norm_2, reshape_0, reshape_1, shape_1):
    
        # EarlyReturn(0, 14)

        # pd_op.slice: (xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_1, [0], full_int_array_0, full_int_array_1, [1], [0])

        return [layer_norm_1, layer_norm_2, reshape_1, reshape_0, slice_0]

if True and not (IsCinnStageEnableDiff() and LastCINNStageFailed()):

    class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
        def prepare_data(self):
            self.inputs = [
                # parameter_0
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                # parameter_1
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                # data_0
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]
            for input in self.inputs:
                input.stop_gradient = True

        def apply_to_static(self, net, use_cinn):
            build_strategy = paddle.static.BuildStrategy()
            input_spec = [
                # parameter_0
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                # parameter_1
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                # data_0
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
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