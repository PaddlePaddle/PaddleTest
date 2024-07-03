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
    return [7][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_0_0_0(self, parameter_0, parameter_2, parameter_1, parameter_3, data_0):

        # pd_op.flatten: (-1x-1xf32, 0x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_0, flatten_1 = paddle._C_ops.flatten(data_0, 1, 3), None

        # pd_op.matmul: (-1x1024xf32) <- (-1x-1xf32, 12544x1024xf32)
        matmul_0 = paddle.matmul(flatten_0, parameter_0, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_0 = matmul_0 + parameter_1

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_1 = paddle.matmul(relu_0, parameter_2, transpose_x=False, transpose_y=False)

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_1 = matmul_1 + parameter_3

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_1 = paddle._C_ops.relu(add_1)
        return flatten_0, flatten_1, matmul_0, relu_0, matmul_1, relu_1



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

    def forward(self, parameter_0, parameter_2, parameter_1, parameter_3, data_0):
        args = [parameter_0, parameter_2, parameter_1, parameter_3, data_0]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_flatten_0,
            self.op_matmul_0,
            self.op_add_0,
            self.op_relu_0,
            self.op_matmul_1,
            self.op_add_1,
            self.op_relu_1,
        ]

    def op_flatten_0(self, parameter_0, parameter_2, parameter_1, parameter_3, data_0):
    
        # EarlyReturn(0, 0)

        # pd_op.flatten: (-1x-1xf32, 0x-1x-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_0, flatten_1 = paddle._C_ops.flatten(data_0, 1, 3), None

        return [parameter_0, parameter_2, parameter_1, parameter_3, flatten_0, flatten_1]

    def op_matmul_0(self, parameter_0, parameter_2, parameter_1, parameter_3, flatten_0, flatten_1):
    
        # EarlyReturn(0, 1)

        # pd_op.matmul: (-1x1024xf32) <- (-1x-1xf32, 12544x1024xf32)
        matmul_0 = paddle.matmul(flatten_0, parameter_0, transpose_x=False, transpose_y=False)

        return [parameter_2, parameter_1, parameter_3, flatten_0, flatten_1, matmul_0]

    def op_add_0(self, parameter_2, parameter_1, parameter_3, flatten_0, flatten_1, matmul_0):
    
        # EarlyReturn(0, 2)

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_0 = matmul_0 + parameter_1

        return [parameter_2, parameter_3, flatten_0, flatten_1, matmul_0, add_0]

    def op_relu_0(self, parameter_2, parameter_3, flatten_0, flatten_1, matmul_0, add_0):
    
        # EarlyReturn(0, 3)

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        return [parameter_2, parameter_3, flatten_0, flatten_1, matmul_0, relu_0]

    def op_matmul_1(self, parameter_2, parameter_3, flatten_0, flatten_1, matmul_0, relu_0):
    
        # EarlyReturn(0, 4)

        # pd_op.matmul: (-1x1024xf32) <- (-1x1024xf32, 1024x1024xf32)
        matmul_1 = paddle.matmul(relu_0, parameter_2, transpose_x=False, transpose_y=False)

        return [parameter_3, flatten_0, flatten_1, matmul_0, relu_0, matmul_1]

    def op_add_1(self, parameter_3, flatten_0, flatten_1, matmul_0, relu_0, matmul_1):
    
        # EarlyReturn(0, 5)

        # pd_op.add: (-1x1024xf32) <- (-1x1024xf32, 1024xf32)
        add_1 = matmul_1 + parameter_3

        return [flatten_0, flatten_1, matmul_0, relu_0, matmul_1, add_1]

    def op_relu_1(self, flatten_0, flatten_1, matmul_0, relu_0, matmul_1, add_1):
    
        # EarlyReturn(0, 6)

        # pd_op.relu: (-1x1024xf32) <- (-1x1024xf32)
        relu_1 = paddle._C_ops.relu(add_1)

        return [flatten_0, flatten_1, matmul_0, relu_0, matmul_1, relu_1]

is_module_block_and_last_stage_passed = (
    True and not last_stage_failed
)
@unittest.skipIf(not is_module_block_and_last_stage_passed, "last stage failed")
class Test_builtin_module_0_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([512, 256, 7, 7], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[12544, 1024], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
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

    def train(self, use_cinn):
        net = Block_builtin_module_0_0_0()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        paddle.seed(2024)
        out = net(*self.inputs)
        return out

if __name__ == '__main__':
    unittest.main()