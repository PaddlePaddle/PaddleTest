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
    if enable_cinn is None:
        return True
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

def ApplyToStatic(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=net.get_input_spec(),
        build_strategy=build_strategy,
        full_graph=True,
    )

class InstanceTrait:

    @classmethod
    def instance(cls):
        if cls.instance_ is None:
            cls.instance_ = cls()
        return cls.instance_

    @classmethod
    def static_instance_with_cinn(cls):
        if cls.static_instance_with_cinn_ is None:
            cls.static_instance_with_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=True
            )
        return cls.static_instance_with_cinn_

    @classmethod
    def static_instance_without_cinn(cls):
        if cls.static_instance_without_cinn_ is None:
            cls.static_instance_without_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=False
            )
        return cls.static_instance_without_cinn_


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

    def train(self, use_cinn):
        if GetEnvVarEnableJit():
            net = self.prepare_static_net(use_cinn)
        else:
            net = self.prepare_net()
        out = net(*self.inputs)
        return out
    
    def prepare_data(self):
        self.inputs = self.get_inputs()
        for input in self.inputs:
            input.stop_gradient = True

    def prepare_net(self):
        return self.get_test_class().instance()

    def prepare_static_net(self, use_cinn):
        if use_cinn:
            return self.get_test_class().static_instance_with_cinn()
        else:
            return self.get_test_class().static_instance_without_cinn()

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



if not (IsCinnStageEnableDiff() and LastCINNStageFailed()):
    class PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63dad6974cd91cf7d06b9749f75782e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63dad6974cd91cf7d06b9749f75782e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fde39fbd91468adede91f53ec460701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fde39fbd91468adede91f53ec460701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e352d87ec5213849f9e537ba9192d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e352d87ec5213849f9e537ba9192d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5916dcd3cb54cf854711fc1e609f9f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5916dcd3cb54cf854711fc1e609f9f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7149ede2f44fdf6f0cdafdf167b42a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7149ede2f44fdf6f0cdafdf167b42a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f0c18179067d8edbf9f77dbd52477a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f0c18179067d8edbf9f77dbd52477a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63dad6974cd91cf7d06b9749f75782e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63dad6974cd91cf7d06b9749f75782e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fec1bdcc7c8d0af8d49e28f5d4496cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fec1bdcc7c8d0af8d49e28f5d4496cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3882f97776243f5e381b53d32bacd8e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3882f97776243f5e381b53d32bacd8e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8e7e0f55c1f012e1e1292c0b8684963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8e7e0f55c1f012e1e1292c0b8684963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8e7e0f55c1f012e1e1292c0b8684963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8e7e0f55c1f012e1e1292c0b8684963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91386386388dd2371c8f8158e0cdbdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91386386388dd2371c8f8158e0cdbdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74f9458f8e1c8571cf326366690f0c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74f9458f8e1c8571cf326366690f0c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a715c644fb1de4df18cc81b1836e1f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a715c644fb1de4df18cc81b1836e1f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25ee4a34b04b01c12e40c268bfe0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25ee4a34b04b01c12e40c268bfe0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d39784cbb1345c66c89fb21626093f97(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c25db1396a3a3b94d33ffd9c4a85e256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.2974533438682556], [0.04537849500775337], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.4127102792263031], [0.41945791244506836], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_05db1eb03d79bb9d8648dcc41ca9f6c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.2677699625492096], [0.015402782708406448], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.1261843889951706], [0.05905040726065636], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.09990614652633667], [0.32696646451950073], [0.2620046138763428], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.29109078645706177], [0.030494073405861855], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_58a4f0d452abedda36a41f44a1cb04d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e88036ca06b6a24e282af95560a726a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.04978129267692566], [0.042689915746450424], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.11604061722755432], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.0034736385568976402], [0.2015986442565918], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.3369775414466858], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_a669ba71230e1ceac139a0ff018af2c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a669ba71230e1ceac139a0ff018af2c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a669ba71230e1ceac139a0ff018af2c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a669ba71230e1ceac139a0ff018af2c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7320b6b312a9155366e9d48b3bf31711(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_69a2d4ac0090888c9ad8e4f30508e252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_dbbafb61a9a38339b8542bdcac725678(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2081550508737564, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.285847932100296], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_75e819aaeba7ed19592064f827a558d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.07290997356176376, 0.22891433537006378, 0.06531837582588196, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_52103ca8a9739ebae86ec69f72ffaeb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.10902168601751328, 0.04308721795678139, 0.134864941239357, 0.4397190809249878, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c2868d4efbd053cb725dcb12bda1a09e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2868d4efbd053cb725dcb12bda1a09e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2868d4efbd053cb725dcb12bda1a09e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2868d4efbd053cb725dcb12bda1a09e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5916dcd3cb54cf854711fc1e609f9f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5916dcd3cb54cf854711fc1e609f9f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74f9458f8e1c8571cf326366690f0c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74f9458f8e1c8571cf326366690f0c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2089bbbdda1df3da7bbdf0be948ae22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2089bbbdda1df3da7bbdf0be948ae22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2089bbbdda1df3da7bbdf0be948ae22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2089bbbdda1df3da7bbdf0be948ae22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f0c18179067d8edbf9f77dbd52477a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f0c18179067d8edbf9f77dbd52477a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fde39fbd91468adede91f53ec460701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fde39fbd91468adede91f53ec460701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b9738fea752ad1d8f2ec682b14254f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_e370ddbe807e5f88e28f082e289e066b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_763ca661e8c265d2d99615d0ff8f1c88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_bb824789bc51930370a67e0e840c690e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_89d36a128c6de40c7bf0919f73d0bbab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.3532122075557709], [0.09650922566652298], [0.2863064706325531], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4361984133720398], [0.14439257979393005], [0.49514085054397583], [0.16213607788085938], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_feda047c66a40f3025bbaa68da054cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_25d9d82dfce4479b5b3b622077b18ba4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3966229259967804], [0.2040838748216629], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.4306377172470093], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4aa3aca05f4921f82f8dea9fbc84b337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_91386386388dd2371c8f8158e0cdbdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91386386388dd2371c8f8158e0cdbdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e352d87ec5213849f9e537ba9192d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e352d87ec5213849f9e537ba9192d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_420bdd54b77cfeb0c34316d30b3aee00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_420bdd54b77cfeb0c34316d30b3aee00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_420bdd54b77cfeb0c34316d30b3aee00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_420bdd54b77cfeb0c34316d30b3aee00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_345ce9107e3d11c02682f0cc6c7f7643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_345ce9107e3d11c02682f0cc6c7f7643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_065aa58244537410b0657d48ef15706c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_065aa58244537410b0657d48ef15706c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_065aa58244537410b0657d48ef15706c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_065aa58244537410b0657d48ef15706c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_525ee95859619cb9598ea9c1fdcba5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_525ee95859619cb9598ea9c1fdcba5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_525ee95859619cb9598ea9c1fdcba5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_525ee95859619cb9598ea9c1fdcba5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7149ede2f44fdf6f0cdafdf167b42a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7149ede2f44fdf6f0cdafdf167b42a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a715c644fb1de4df18cc81b1836e1f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a715c644fb1de4df18cc81b1836e1f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b03dc53a5fdeb5d8fdceb4c5c1590f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.36288416385650635], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.3282714784145355], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_85dcbd4a5b54fdcb4826fe28a8f99de9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_3fab0e90e04360dc3004a1ff8b861923(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b577424018ac2ea664fe3b91ccac131b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31906890869140625], [0.02482573315501213], [0.439250111579895], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.04625708982348442], [0.1838442087173462], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_3882f97776243f5e381b53d32bacd8e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3882f97776243f5e381b53d32bacd8e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25ee4a34b04b01c12e40c268bfe0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25ee4a34b04b01c12e40c268bfe0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_808e91ac6e67c3fa1a06d03136dd7eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_808e91ac6e67c3fa1a06d03136dd7eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_623254142c9cec186511655e0433aa8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_623254142c9cec186511655e0433aa8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_623254142c9cec186511655e0433aa8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_623254142c9cec186511655e0433aa8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b458caa365938c17ebe3f997550c4d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b458caa365938c17ebe3f997550c4d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b458caa365938c17ebe3f997550c4d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b458caa365938c17ebe3f997550c4d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ccc1284a14c340efe0178973c9602fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ccc1284a14c340efe0178973c9602fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ccc1284a14c340efe0178973c9602fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ccc1284a14c340efe0178973c9602fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_345ce9107e3d11c02682f0cc6c7f7643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_345ce9107e3d11c02682f0cc6c7f7643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ed078d1982e84a1548461d570565c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.14522317051887512], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d67cfc205b166ece1b284e89594ba4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.48723578453063965], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28212907910346985], [0.440621554851532], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f5f6883af7b670631a99590554296577(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7086c3223794c70a8f5c2454bdcacf65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.32093942165374756], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_1ac9f283686ae38d97f3495b7db79a94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ac9f283686ae38d97f3495b7db79a94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ac9f283686ae38d97f3495b7db79a94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ac9f283686ae38d97f3495b7db79a94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_808e91ac6e67c3fa1a06d03136dd7eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_808e91ac6e67c3fa1a06d03136dd7eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fec1bdcc7c8d0af8d49e28f5d4496cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fec1bdcc7c8d0af8d49e28f5d4496cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103d49111633544ba2757dcd0f09b1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103d49111633544ba2757dcd0f09b1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c464a4a644dbde9f04ac9bf9b18ea79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c464a4a644dbde9f04ac9bf9b18ea79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c464a4a644dbde9f04ac9bf9b18ea79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c464a4a644dbde9f04ac9bf9b18ea79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103d49111633544ba2757dcd0f09b1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103d49111633544ba2757dcd0f09b1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_32c47345b2e962ac22515bb98f95f624(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_828cbf528b7a7ebe7d092af1541789e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32c47345b2e962ac22515bb98f95f624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_828cbf528b7a7ebe7d092af1541789e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32c47345b2e962ac22515bb98f95f624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9e58804555ce783ae198c2d6cad381a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ce04d0716f897b2b310d21738c0a840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9e58804555ce783ae198c2d6cad381a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce04d0716f897b2b310d21738c0a840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9e58804555ce783ae198c2d6cad381a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f863b6c02a2a33751181e407f7503820(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15e26408ef951974a57e3960d5a96c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f863b6c02a2a33751181e407f7503820
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15e26408ef951974a57e3960d5a96c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f863b6c02a2a33751181e407f7503820
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08ca267f4f2d44366635b94ec1ae9a58(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32f776f61c989ca5a08747af0e086a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08ca267f4f2d44366635b94ec1ae9a58
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32f776f61c989ca5a08747af0e086a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08ca267f4f2d44366635b94ec1ae9a58
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d696da19788d5e8630828592ea4af5da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_862b25a1e631b20bdd5c55575f8d19a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d696da19788d5e8630828592ea4af5da
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862b25a1e631b20bdd5c55575f8d19a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d696da19788d5e8630828592ea4af5da
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81a0f7ded79965bce03f70d01c0768fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4bff057acf2c83d266c659a33c2bee48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81a0f7ded79965bce03f70d01c0768fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bff057acf2c83d266c659a33c2bee48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81a0f7ded79965bce03f70d01c0768fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_828cbf528b7a7ebe7d092af1541789e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32c47345b2e962ac22515bb98f95f624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_828cbf528b7a7ebe7d092af1541789e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32c47345b2e962ac22515bb98f95f624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef21ae3a6dbb9046e61a7d89122982dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_97a21f2914a656a7ecf994868a8119d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef21ae3a6dbb9046e61a7d89122982dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a21f2914a656a7ecf994868a8119d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef21ae3a6dbb9046e61a7d89122982dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_982e62db2e54dba72ba9173f1a94bcc8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ccdd66a715fad71a5ec742a78e0d7fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_982e62db2e54dba72ba9173f1a94bcc8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ccdd66a715fad71a5ec742a78e0d7fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_982e62db2e54dba72ba9173f1a94bcc8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_89d73834239d924c32258bd8b1df4fa8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17807b674b15175c0b351e16b3ae87c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89d73834239d924c32258bd8b1df4fa8
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17807b674b15175c0b351e16b3ae87c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89d73834239d924c32258bd8b1df4fa8
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17807b674b15175c0b351e16b3ae87c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89d73834239d924c32258bd8b1df4fa8
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17807b674b15175c0b351e16b3ae87c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89d73834239d924c32258bd8b1df4fa8
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_61a235c2c27af3308faa6780faba3693(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22926ea716420f2306eb64c31bdd6ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a235c2c27af3308faa6780faba3693
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22926ea716420f2306eb64c31bdd6ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a235c2c27af3308faa6780faba3693
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6b0ab0c33c4edccf1b9f078ed94e0fed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8a31fc3cd757de6adf27d45a29fadc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6b0ab0c33c4edccf1b9f078ed94e0fed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8a31fc3cd757de6adf27d45a29fadc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6b0ab0c33c4edccf1b9f078ed94e0fed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a4293ceb476c99858231453473ed33f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d29337c1788e66960d9d25a3879c4734(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a4293ceb476c99858231453473ed33f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d29337c1788e66960d9d25a3879c4734(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a4293ceb476c99858231453473ed33f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7a4193a004d60915430f36c6259757c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_795ca14a6a70bd511b77caa781ea2e88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a4193a004d60915430f36c6259757c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_795ca14a6a70bd511b77caa781ea2e88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a4193a004d60915430f36c6259757c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce818f5a99d731f8692e2985c78bfec3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.2974533438682556], [0.04537849500775337], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.4127102792263031], [0.41945791244506836], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_b039e9f8ccfdad03a9e7ef90538a144b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.2677699625492096], [0.015402782708406448], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.1261843889951706], [0.05905040726065636], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.09990614652633667], [0.32696646451950073], [0.2620046138763428], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.29109078645706177], [0.030494073405861855], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_5b81a03d86ef161ddc72b852b236131f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_b5d842432dd3623ede9dd0bcaec4c172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.04978129267692566], [0.042689915746450424], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.11604061722755432], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.0034736385568976402], [0.2015986442565918], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.3369775414466858], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_c1879197843bd29982f8ae05deef96a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_772c32e8136dbdb60381de5f736576f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1879197843bd29982f8ae05deef96a5
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_772c32e8136dbdb60381de5f736576f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1879197843bd29982f8ae05deef96a5
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_772c32e8136dbdb60381de5f736576f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1879197843bd29982f8ae05deef96a5
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_772c32e8136dbdb60381de5f736576f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1879197843bd29982f8ae05deef96a5
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42e9983bcd391597a36d4f0ada4d0d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b3e29deba9860728372688a99f8df446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2081550508737564, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.285847932100296], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_04afd3e713950caf763e8f8cf991e94e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.07290997356176376, 0.22891433537006378, 0.06531837582588196, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ed3b22b0d31239e571442328fc4b6b56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.10902168601751328, 0.04308721795678139, 0.134864941239357, 0.4397190809249878, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_f229cc4dc2f0f5ee4286d55d1e18d73b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b74c86ccc8b7b0053e528a30a843ab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f229cc4dc2f0f5ee4286d55d1e18d73b
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b74c86ccc8b7b0053e528a30a843ab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f229cc4dc2f0f5ee4286d55d1e18d73b
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b74c86ccc8b7b0053e528a30a843ab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f229cc4dc2f0f5ee4286d55d1e18d73b
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b74c86ccc8b7b0053e528a30a843ab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f229cc4dc2f0f5ee4286d55d1e18d73b
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32f776f61c989ca5a08747af0e086a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08ca267f4f2d44366635b94ec1ae9a58
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32f776f61c989ca5a08747af0e086a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08ca267f4f2d44366635b94ec1ae9a58
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8a31fc3cd757de6adf27d45a29fadc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6b0ab0c33c4edccf1b9f078ed94e0fed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8a31fc3cd757de6adf27d45a29fadc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6b0ab0c33c4edccf1b9f078ed94e0fed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86256248354bf1845f68e300333593df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ec31c668e6022e877a823940a10b7786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86256248354bf1845f68e300333593df
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec31c668e6022e877a823940a10b7786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86256248354bf1845f68e300333593df
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec31c668e6022e877a823940a10b7786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86256248354bf1845f68e300333593df
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec31c668e6022e877a823940a10b7786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86256248354bf1845f68e300333593df
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bff057acf2c83d266c659a33c2bee48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81a0f7ded79965bce03f70d01c0768fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bff057acf2c83d266c659a33c2bee48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81a0f7ded79965bce03f70d01c0768fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce04d0716f897b2b310d21738c0a840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9e58804555ce783ae198c2d6cad381a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce04d0716f897b2b310d21738c0a840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9e58804555ce783ae198c2d6cad381a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f429859c88336af9c3bb54b5a031cd99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5b69ce9525562a399bad43b6265e188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_e389861004caec73b4970634fe6bad31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_3a46fa44a9155b25a4b9ec3904853e30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b2f184936fba6807ea1e006dd4453313(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    
    class PrimitiveOp_0c76d2a5dbbdfc403876257f49009587(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c70c703e283b0c21ac49f9c685c3d450(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.3532122075557709], [0.09650922566652298], [0.2863064706325531], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4361984133720398], [0.14439257979393005], [0.49514085054397583], [0.16213607788085938], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_c430e89cc591bc47efa071b95c4ff3be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ba774e14dda8bd3e5f6193358c38266b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3966229259967804], [0.2040838748216629], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.4306377172470093], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2021d80feec9f66b61d588e80e4b6a90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_22926ea716420f2306eb64c31bdd6ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a235c2c27af3308faa6780faba3693
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22926ea716420f2306eb64c31bdd6ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61a235c2c27af3308faa6780faba3693
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15e26408ef951974a57e3960d5a96c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f863b6c02a2a33751181e407f7503820
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15e26408ef951974a57e3960d5a96c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f863b6c02a2a33751181e407f7503820
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ac091212e660f87c31de29d2ea94455(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a149c2eb4ee641a2a2df17673d00d74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ac091212e660f87c31de29d2ea94455
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a149c2eb4ee641a2a2df17673d00d74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ac091212e660f87c31de29d2ea94455
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a149c2eb4ee641a2a2df17673d00d74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ac091212e660f87c31de29d2ea94455
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a149c2eb4ee641a2a2df17673d00d74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ac091212e660f87c31de29d2ea94455
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c66271fd2a59c8f311159840d003a0e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07fb9bf3cb141f753a374951ffeea8cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c66271fd2a59c8f311159840d003a0e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07fb9bf3cb141f753a374951ffeea8cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c66271fd2a59c8f311159840d003a0e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7db9a189d36218d41a7f4db2464040c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c15afe2b51262ed352451147a85d8763(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7db9a189d36218d41a7f4db2464040c0
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c15afe2b51262ed352451147a85d8763(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7db9a189d36218d41a7f4db2464040c0
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c15afe2b51262ed352451147a85d8763(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7db9a189d36218d41a7f4db2464040c0
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c15afe2b51262ed352451147a85d8763(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7db9a189d36218d41a7f4db2464040c0
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e73b2046f0d3babd45dafd64374a190(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d574871fc65c78042a43e2f91edf8fc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e73b2046f0d3babd45dafd64374a190
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d574871fc65c78042a43e2f91edf8fc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e73b2046f0d3babd45dafd64374a190
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d574871fc65c78042a43e2f91edf8fc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e73b2046f0d3babd45dafd64374a190
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d574871fc65c78042a43e2f91edf8fc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e73b2046f0d3babd45dafd64374a190
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862b25a1e631b20bdd5c55575f8d19a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d696da19788d5e8630828592ea4af5da
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862b25a1e631b20bdd5c55575f8d19a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d696da19788d5e8630828592ea4af5da
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d29337c1788e66960d9d25a3879c4734(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a4293ceb476c99858231453473ed33f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d29337c1788e66960d9d25a3879c4734(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a4293ceb476c99858231453473ed33f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c09b68bb00be14af73332763dc0e330(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.36288416385650635], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.3282714784145355], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_833b7ae53aaf134d98ef05dd9a74a1db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_819868be715d5bdb46dad6908c09d58b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_0de4680f9dea3f7fa09e6fd819b79185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31906890869140625], [0.02482573315501213], [0.439250111579895], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.04625708982348442], [0.1838442087173462], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9ccdd66a715fad71a5ec742a78e0d7fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_982e62db2e54dba72ba9173f1a94bcc8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ccdd66a715fad71a5ec742a78e0d7fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_982e62db2e54dba72ba9173f1a94bcc8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_795ca14a6a70bd511b77caa781ea2e88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a4193a004d60915430f36c6259757c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_795ca14a6a70bd511b77caa781ea2e88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a4193a004d60915430f36c6259757c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_79ff0b3a4fa0f4401aa01a6987c9ec0d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0426f88549fdab68645be83780f00121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79ff0b3a4fa0f4401aa01a6987c9ec0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0426f88549fdab68645be83780f00121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79ff0b3a4fa0f4401aa01a6987c9ec0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fda0de0b6b48edb6b61a8c5034e3986f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_849166d264007215abf996eb825a49c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda0de0b6b48edb6b61a8c5034e3986f
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_849166d264007215abf996eb825a49c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda0de0b6b48edb6b61a8c5034e3986f
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_849166d264007215abf996eb825a49c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda0de0b6b48edb6b61a8c5034e3986f
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_849166d264007215abf996eb825a49c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda0de0b6b48edb6b61a8c5034e3986f
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_977c6fd58d545451fe725d7de010af2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c47b848c04345308f8d9cd0a3ad77b2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977c6fd58d545451fe725d7de010af2b
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c47b848c04345308f8d9cd0a3ad77b2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977c6fd58d545451fe725d7de010af2b
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c47b848c04345308f8d9cd0a3ad77b2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977c6fd58d545451fe725d7de010af2b
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c47b848c04345308f8d9cd0a3ad77b2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977c6fd58d545451fe725d7de010af2b
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a5df1c8353dfc2f88d7dc295677513af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a954b9a0f4afdfcb1ec5bfe55505a040(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5df1c8353dfc2f88d7dc295677513af
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a954b9a0f4afdfcb1ec5bfe55505a040(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5df1c8353dfc2f88d7dc295677513af
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a954b9a0f4afdfcb1ec5bfe55505a040(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5df1c8353dfc2f88d7dc295677513af
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a954b9a0f4afdfcb1ec5bfe55505a040(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5df1c8353dfc2f88d7dc295677513af
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07fb9bf3cb141f753a374951ffeea8cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c66271fd2a59c8f311159840d003a0e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07fb9bf3cb141f753a374951ffeea8cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c66271fd2a59c8f311159840d003a0e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ad5069168f2771bde956fe2a8dba9d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.14522317051887512], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_62408b85a33b2ce74fca58d178f5e9a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.48723578453063965], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28212907910346985], [0.440621554851532], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_82d9f6f74dd5d6b445a306a3b8a0ed1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_dbaa7db47caad8386c3ce7b734d62cf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.32093942165374756], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    
    class PrimitiveOp_4ea44031d1cf3979ee4089830913a483(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7064fdf073e6948c2950b053e2050743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ea44031d1cf3979ee4089830913a483
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7064fdf073e6948c2950b053e2050743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ea44031d1cf3979ee4089830913a483
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7064fdf073e6948c2950b053e2050743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ea44031d1cf3979ee4089830913a483
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7064fdf073e6948c2950b053e2050743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ea44031d1cf3979ee4089830913a483
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0426f88549fdab68645be83780f00121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79ff0b3a4fa0f4401aa01a6987c9ec0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0426f88549fdab68645be83780f00121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79ff0b3a4fa0f4401aa01a6987c9ec0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a21f2914a656a7ecf994868a8119d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef21ae3a6dbb9046e61a7d89122982dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a21f2914a656a7ecf994868a8119d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef21ae3a6dbb9046e61a7d89122982dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3bd1690ee6e4a3961d164e6cf3b3f5b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3afae5c721bf61e0d2b6b63e966cf71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3bd1690ee6e4a3961d164e6cf3b3f5b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3afae5c721bf61e0d2b6b63e966cf71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3bd1690ee6e4a3961d164e6cf3b3f5b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f2c061717c61d429185d990a59dc886b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_600fda635855e3476c0d6ed303f80a37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2c061717c61d429185d990a59dc886b
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600fda635855e3476c0d6ed303f80a37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2c061717c61d429185d990a59dc886b
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600fda635855e3476c0d6ed303f80a37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2c061717c61d429185d990a59dc886b
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600fda635855e3476c0d6ed303f80a37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2c061717c61d429185d990a59dc886b
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3afae5c721bf61e0d2b6b63e966cf71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3bd1690ee6e4a3961d164e6cf3b3f5b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3afae5c721bf61e0d2b6b63e966cf71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3bd1690ee6e4a3961d164e6cf3b3f5b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63dad6974cd91cf7d06b9749f75782e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63dad6974cd91cf7d06b9749f75782e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fde39fbd91468adede91f53ec460701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fde39fbd91468adede91f53ec460701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e352d87ec5213849f9e537ba9192d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e352d87ec5213849f9e537ba9192d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5916dcd3cb54cf854711fc1e609f9f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5916dcd3cb54cf854711fc1e609f9f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7149ede2f44fdf6f0cdafdf167b42a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7149ede2f44fdf6f0cdafdf167b42a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f0c18179067d8edbf9f77dbd52477a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f0c18179067d8edbf9f77dbd52477a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63dad6974cd91cf7d06b9749f75782e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63dad6974cd91cf7d06b9749f75782e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fec1bdcc7c8d0af8d49e28f5d4496cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fec1bdcc7c8d0af8d49e28f5d4496cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3882f97776243f5e381b53d32bacd8e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3882f97776243f5e381b53d32bacd8e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4fbf3e56c54f41f6787c4c8743a1068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4fbf3e56c54f41f6787c4c8743a1068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4fbf3e56c54f41f6787c4c8743a1068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4fbf3e56c54f41f6787c4c8743a1068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91386386388dd2371c8f8158e0cdbdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91386386388dd2371c8f8158e0cdbdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74f9458f8e1c8571cf326366690f0c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74f9458f8e1c8571cf326366690f0c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a715c644fb1de4df18cc81b1836e1f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a715c644fb1de4df18cc81b1836e1f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25ee4a34b04b01c12e40c268bfe0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25ee4a34b04b01c12e40c268bfe0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c25db1396a3a3b94d33ffd9c4a85e256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.2974533438682556], [0.04537849500775337], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.4127102792263031], [0.41945791244506836], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_05db1eb03d79bb9d8648dcc41ca9f6c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.2677699625492096], [0.015402782708406448], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.1261843889951706], [0.05905040726065636], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.09990614652633667], [0.32696646451950073], [0.2620046138763428], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.29109078645706177], [0.030494073405861855], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_58a4f0d452abedda36a41f44a1cb04d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e88036ca06b6a24e282af95560a726a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.04978129267692566], [0.042689915746450424], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.11604061722755432], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.0034736385568976402], [0.2015986442565918], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.3369775414466858], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_5c4f05bf4dc7ae89180be3171c13b16b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c4f05bf4dc7ae89180be3171c13b16b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c4f05bf4dc7ae89180be3171c13b16b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c4f05bf4dc7ae89180be3171c13b16b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69a2d4ac0090888c9ad8e4f30508e252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_dbbafb61a9a38339b8542bdcac725678(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2081550508737564, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.285847932100296], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_75e819aaeba7ed19592064f827a558d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.07290997356176376, 0.22891433537006378, 0.06531837582588196, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_52103ca8a9739ebae86ec69f72ffaeb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.10902168601751328, 0.04308721795678139, 0.134864941239357, 0.4397190809249878, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_6aef6784199fcd54ede30fb25f5f568b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6aef6784199fcd54ede30fb25f5f568b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6aef6784199fcd54ede30fb25f5f568b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6aef6784199fcd54ede30fb25f5f568b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5916dcd3cb54cf854711fc1e609f9f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5916dcd3cb54cf854711fc1e609f9f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74f9458f8e1c8571cf326366690f0c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74f9458f8e1c8571cf326366690f0c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0dcefdf4c56164ea10b594a01e578c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0dcefdf4c56164ea10b594a01e578c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0dcefdf4c56164ea10b594a01e578c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0dcefdf4c56164ea10b594a01e578c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f0c18179067d8edbf9f77dbd52477a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f0c18179067d8edbf9f77dbd52477a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fde39fbd91468adede91f53ec460701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fde39fbd91468adede91f53ec460701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b9738fea752ad1d8f2ec682b14254f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_e370ddbe807e5f88e28f082e289e066b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_763ca661e8c265d2d99615d0ff8f1c88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_bb824789bc51930370a67e0e840c690e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_89d36a128c6de40c7bf0919f73d0bbab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.3532122075557709], [0.09650922566652298], [0.2863064706325531], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4361984133720398], [0.14439257979393005], [0.49514085054397583], [0.16213607788085938], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_feda047c66a40f3025bbaa68da054cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_25d9d82dfce4479b5b3b622077b18ba4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3966229259967804], [0.2040838748216629], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.4306377172470093], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4aa3aca05f4921f82f8dea9fbc84b337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_91386386388dd2371c8f8158e0cdbdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91386386388dd2371c8f8158e0cdbdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e352d87ec5213849f9e537ba9192d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e352d87ec5213849f9e537ba9192d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_793fc432848bcb89c5834984266643ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_793fc432848bcb89c5834984266643ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_793fc432848bcb89c5834984266643ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_793fc432848bcb89c5834984266643ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_345ce9107e3d11c02682f0cc6c7f7643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_345ce9107e3d11c02682f0cc6c7f7643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6944ecfb2f97bed0f658bce6b7a06f2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6944ecfb2f97bed0f658bce6b7a06f2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6944ecfb2f97bed0f658bce6b7a06f2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6944ecfb2f97bed0f658bce6b7a06f2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_038afcada67a05dfb076b0e65415704e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_038afcada67a05dfb076b0e65415704e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_038afcada67a05dfb076b0e65415704e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_038afcada67a05dfb076b0e65415704e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7149ede2f44fdf6f0cdafdf167b42a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7149ede2f44fdf6f0cdafdf167b42a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a715c644fb1de4df18cc81b1836e1f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a715c644fb1de4df18cc81b1836e1f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b03dc53a5fdeb5d8fdceb4c5c1590f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.36288416385650635], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.3282714784145355], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_85dcbd4a5b54fdcb4826fe28a8f99de9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_3fab0e90e04360dc3004a1ff8b861923(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b577424018ac2ea664fe3b91ccac131b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31906890869140625], [0.02482573315501213], [0.439250111579895], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.04625708982348442], [0.1838442087173462], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_3882f97776243f5e381b53d32bacd8e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3882f97776243f5e381b53d32bacd8e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25ee4a34b04b01c12e40c268bfe0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25ee4a34b04b01c12e40c268bfe0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_808e91ac6e67c3fa1a06d03136dd7eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_808e91ac6e67c3fa1a06d03136dd7eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a20128f004b62ac0c65a1e7c167818fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a20128f004b62ac0c65a1e7c167818fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a20128f004b62ac0c65a1e7c167818fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a20128f004b62ac0c65a1e7c167818fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59bf452e4a2651f7bc8fd6248e9266(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59bf452e4a2651f7bc8fd6248e9266(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59bf452e4a2651f7bc8fd6248e9266(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59bf452e4a2651f7bc8fd6248e9266(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bc43f9305c5da7bcdd07b790ed98589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bc43f9305c5da7bcdd07b790ed98589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bc43f9305c5da7bcdd07b790ed98589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bc43f9305c5da7bcdd07b790ed98589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_345ce9107e3d11c02682f0cc6c7f7643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_345ce9107e3d11c02682f0cc6c7f7643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ed078d1982e84a1548461d570565c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.14522317051887512], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d67cfc205b166ece1b284e89594ba4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.48723578453063965], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28212907910346985], [0.440621554851532], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f5f6883af7b670631a99590554296577(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7086c3223794c70a8f5c2454bdcacf65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.32093942165374756], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7010780ce3f0763c784cc5c86ac6ef7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7010780ce3f0763c784cc5c86ac6ef7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7010780ce3f0763c784cc5c86ac6ef7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7010780ce3f0763c784cc5c86ac6ef7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_808e91ac6e67c3fa1a06d03136dd7eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_808e91ac6e67c3fa1a06d03136dd7eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fec1bdcc7c8d0af8d49e28f5d4496cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fec1bdcc7c8d0af8d49e28f5d4496cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103d49111633544ba2757dcd0f09b1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103d49111633544ba2757dcd0f09b1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e42b681a89be2ba63b3a486554b99f5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e42b681a89be2ba63b3a486554b99f5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e42b681a89be2ba63b3a486554b99f5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e42b681a89be2ba63b3a486554b99f5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103d49111633544ba2757dcd0f09b1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103d49111633544ba2757dcd0f09b1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b93828341902b0b37fd9aef39ee3f41
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()