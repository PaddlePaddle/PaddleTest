import os
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


    class TestPrimitiveOp_9d287e1ae9e2eb30bc3bbc91cb5dc5a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d287e1ae9e2eb30bc3bbc91cb5dc5a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d287e1ae9e2eb30bc3bbc91cb5dc5a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d287e1ae9e2eb30bc3bbc91cb5dc5a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_72d04221d1548db01486e7ff6b07a5bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.32824042439460754], [0.11407893896102905], [0.04877207800745964], [0.24639740586280823], [0.4660728871822357], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.3819327652454376], [0.053689487278461456], [0.17333781719207764], [0.3545892834663391], [0.06132432818412781], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_4c855bf6685f22eda2c144425f5fddaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.477491170167923], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.23870912194252014], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e55ede414881c50f6ccbab19d32a8cb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33700039982795715], [0.13599033653736115], [0.16089512407779694], [0.4550243616104126], [0.2522077262401581], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.1544954627752304], [0.3554002642631531], [0.06358063966035843], [0.38996535539627075], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_8c560ef05b27b9260b5dbada88d15250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.2988268733024597], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.3334523141384125], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_3dc0021c8e21d0fdfda73e5257876926(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dc0021c8e21d0fdfda73e5257876926(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dc0021c8e21d0fdfda73e5257876926(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dc0021c8e21d0fdfda73e5257876926(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_dd2ce49d3a55bf74c0d514ad8b233d57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45263877511024475, 0.18129011988639832, 0.45884010195732117, 0.49139586091041565, 0.3470446765422821, 0.2975061535835266], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_92d92881f4f654d702cc5e0be767460e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.4360129237174988, 0.43007832765579224, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.35412517189979553, 0.2729951739311218, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f4e1f3884a85f6362f0299708b5d43d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.20852655172348022, 0.4761844575405121, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e1de4d669a201a2d1ae77cc53f200505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.06498825550079346, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_3d61e86387afbe2dcfcb9738ed079dd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d61e86387afbe2dcfcb9738ed079dd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d61e86387afbe2dcfcb9738ed079dd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d61e86387afbe2dcfcb9738ed079dd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e3fd782d0b37dccbaa0bf6fbd8454fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3fd782d0b37dccbaa0bf6fbd8454fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3fd782d0b37dccbaa0bf6fbd8454fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3fd782d0b37dccbaa0bf6fbd8454fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8df184385a5249ac1ee8ec02577ed5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9c1dd07e9b15ccb9ff8b5fa8b4bda0ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_550d9dd31e474c8357aca02d99ccc334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_66d6b78962d9380c9c0d161020f5fa5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_935b1008220f71ab47d724080e22ddde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.2049286961555481], [0.05052289366722107], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.38781505823135376], [0.12123280763626099], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_285c6d8054a8e9e91a547e31e582b9d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_016a4507bad3bb00461b4284b2678d63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06345830112695694], [0.37558823823928833], [0.1739027202129364], [0.027879230678081512], [0.49161165952682495], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.19266913831233978], [0.3696064352989197], [0.3602122664451599], [0.343851238489151], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_1730a2e3607493133365161b31b2aa02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.13699816167354584], [0.30248284339904785], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.4184499979019165], [0.4230078458786011], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_88359994a17ae5bfd69ea986bd4f9e09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88359994a17ae5bfd69ea986bd4f9e09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88359994a17ae5bfd69ea986bd4f9e09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88359994a17ae5bfd69ea986bd4f9e09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2da868aa9a1fa09f3f639cf8daef7f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2da868aa9a1fa09f3f639cf8daef7f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2da868aa9a1fa09f3f639cf8daef7f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2da868aa9a1fa09f3f639cf8daef7f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8826211499358eb9ecbea3dc183e3791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8826211499358eb9ecbea3dc183e3791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8826211499358eb9ecbea3dc183e3791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8826211499358eb9ecbea3dc183e3791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f404990f952b8401e6dfb6149e922ef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.08265111595392227], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.1922953873872757], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_d46be9040244eb19dcd1dc1b8971e428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.3689388930797577], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.03426346927881241], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9f7cfa08fb1aeb9fec67db5bcfc8946c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_060a64c847eef0c6a07910eb2660a67e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_5f817bdedcd4a64125d36e5142a6675d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f817bdedcd4a64125d36e5142a6675d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f817bdedcd4a64125d36e5142a6675d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f817bdedcd4a64125d36e5142a6675d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf69b83b2bf3a08f850aaff70642110(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf69b83b2bf3a08f850aaff70642110(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf69b83b2bf3a08f850aaff70642110(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf69b83b2bf3a08f850aaff70642110(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19086c5004d153b354272736ebd346e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19086c5004d153b354272736ebd346e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19086c5004d153b354272736ebd346e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19086c5004d153b354272736ebd346e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_66e8d21b92ff6c50d451fc82100bb1e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.43085747957229614], [0.18588005006313324], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4131563901901245], [0.2982136011123657], [0.22558467090129852], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d11ad7f59b73405823c7ec6588889da0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.025396505370736122], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33433759212493896], [0.12095697224140167], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_22b5ce6a82c6e75339c26a25096dc19f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41876405477523804], [0.0375184640288353], [0.09816955029964447], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.462017297744751], [0.11372362822294235], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_852d78fe1dcd3c0c7226d41997198209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_4f1b5ba1e644d5d58f04584b8625c3f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f1b5ba1e644d5d58f04584b8625c3f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f1b5ba1e644d5d58f04584b8625c3f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f1b5ba1e644d5d58f04584b8625c3f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a3c5b0244745a026e7e7884158846b55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3c5b0244745a026e7e7884158846b55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3c5b0244745a026e7e7884158846b55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3c5b0244745a026e7e7884158846b55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_e86b988eb9b5bc9403f66b6f3851e93b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5eedff16140f88cc9adb3c2ac8fd821e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e86b988eb9b5bc9403f66b6f3851e93b
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eedff16140f88cc9adb3c2ac8fd821e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e86b988eb9b5bc9403f66b6f3851e93b
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eedff16140f88cc9adb3c2ac8fd821e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e86b988eb9b5bc9403f66b6f3851e93b
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eedff16140f88cc9adb3c2ac8fd821e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e86b988eb9b5bc9403f66b6f3851e93b
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_739e60e31fa2c8fa42a4e52f21b4b8e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.32824042439460754], [0.11407893896102905], [0.04877207800745964], [0.24639740586280823], [0.4660728871822357], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.3819327652454376], [0.053689487278461456], [0.17333781719207764], [0.3545892834663391], [0.06132432818412781], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_f71f1010583078fcb38a45c04e3b5c19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.477491170167923], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.23870912194252014], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_b06885d9d25815edfec93f7fddfa9463(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33700039982795715], [0.13599033653736115], [0.16089512407779694], [0.4550243616104126], [0.2522077262401581], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.1544954627752304], [0.3554002642631531], [0.06358063966035843], [0.38996535539627075], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_f5dafb5e8f22392a16d986b72d3e6321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.2988268733024597], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.3334523141384125], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_b2e701693a98aed713742ae3e6897663(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e9510a4fa2158fea7ec5d9ce4035b6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2e701693a98aed713742ae3e6897663
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9510a4fa2158fea7ec5d9ce4035b6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2e701693a98aed713742ae3e6897663
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9510a4fa2158fea7ec5d9ce4035b6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2e701693a98aed713742ae3e6897663
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9510a4fa2158fea7ec5d9ce4035b6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2e701693a98aed713742ae3e6897663
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_6194070388692f65247186ed352b7255(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45263877511024475, 0.18129011988639832, 0.45884010195732117, 0.49139586091041565, 0.3470446765422821, 0.2975061535835266], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c7443136620000d0e0af1771b3b1da91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.4360129237174988, 0.43007832765579224, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.35412517189979553, 0.2729951739311218, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_eb3bb30548e30dadf3346799b17a0a06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.20852655172348022, 0.4761844575405121, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b0f743abbeab8d201a81f40913a31222(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.06498825550079346, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_7fd1160c0034a1972dde46a66a8cd66c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_419110f612bc8ff8bedf815f1e771316(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fd1160c0034a1972dde46a66a8cd66c
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_419110f612bc8ff8bedf815f1e771316(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fd1160c0034a1972dde46a66a8cd66c
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_419110f612bc8ff8bedf815f1e771316(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fd1160c0034a1972dde46a66a8cd66c
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_419110f612bc8ff8bedf815f1e771316(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fd1160c0034a1972dde46a66a8cd66c
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_537fcb6a11c9604175e63f01dc6a4d52(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f9c56015649bb26ad991a2e1695a6c75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_537fcb6a11c9604175e63f01dc6a4d52
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9c56015649bb26ad991a2e1695a6c75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_537fcb6a11c9604175e63f01dc6a4d52
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9c56015649bb26ad991a2e1695a6c75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_537fcb6a11c9604175e63f01dc6a4d52
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9c56015649bb26ad991a2e1695a6c75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_537fcb6a11c9604175e63f01dc6a4d52
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d8ca1a4ad5fa348e1ef05abb7c9c121a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_25dce77dbaaf5774ee70ffdc2e9068da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_358c68ea3c8a44a3a9115ea470bb5397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_d5692e84d0794c313440caab9ceb3a84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
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


    class TestPrimitiveOp_8c2876d1d2e7f93c7463a419d04cb73b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.2049286961555481], [0.05052289366722107], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.38781505823135376], [0.12123280763626099], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ba6a7857cbbc8fbb8bb0e69064386472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_10fbe1f320a8bdcb3bcbdb18a6246d05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06345830112695694], [0.37558823823928833], [0.1739027202129364], [0.027879230678081512], [0.49161165952682495], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.19266913831233978], [0.3696064352989197], [0.3602122664451599], [0.343851238489151], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d0f2aede681ba99e258a8bbec05543aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.13699816167354584], [0.30248284339904785], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.4184499979019165], [0.4230078458786011], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
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


    
    class PrimitiveOp_6375f5df6c26aacc02fd4912ddd976bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8afe5932fe11b99572e550699008164e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6375f5df6c26aacc02fd4912ddd976bf
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe5932fe11b99572e550699008164e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6375f5df6c26aacc02fd4912ddd976bf
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe5932fe11b99572e550699008164e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6375f5df6c26aacc02fd4912ddd976bf
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe5932fe11b99572e550699008164e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6375f5df6c26aacc02fd4912ddd976bf
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_4e992e2583c6c3f7cc1a9fba471e5b9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e6b9ed6e07322e553b491d14ff9a216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e992e2583c6c3f7cc1a9fba471e5b9a
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6b9ed6e07322e553b491d14ff9a216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e992e2583c6c3f7cc1a9fba471e5b9a
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6b9ed6e07322e553b491d14ff9a216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e992e2583c6c3f7cc1a9fba471e5b9a
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6b9ed6e07322e553b491d14ff9a216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e992e2583c6c3f7cc1a9fba471e5b9a
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5d5225a8dd33c281913dbe911996557(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5054f37c271bd1cd50e4b1d4789d87e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5d5225a8dd33c281913dbe911996557
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5054f37c271bd1cd50e4b1d4789d87e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5d5225a8dd33c281913dbe911996557
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5054f37c271bd1cd50e4b1d4789d87e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5d5225a8dd33c281913dbe911996557
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5054f37c271bd1cd50e4b1d4789d87e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5d5225a8dd33c281913dbe911996557
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c898860519d1675a47f8678743d36822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.08265111595392227], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.1922953873872757], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2a4cd6647876fa8f627d1198027d62c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.3689388930797577], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.03426346927881241], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_0f98ed811d438e6caaf82c3de5bd0d61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_4bd94269e5bc94f4db18f27c1b845481(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
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


    
    class PrimitiveOp_9ab7dfb1ae42ce8a5c2e4ba02cf2b1ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aedb76c14127c0ad8eb3cf4c7740b392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ab7dfb1ae42ce8a5c2e4ba02cf2b1ab
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aedb76c14127c0ad8eb3cf4c7740b392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ab7dfb1ae42ce8a5c2e4ba02cf2b1ab
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aedb76c14127c0ad8eb3cf4c7740b392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ab7dfb1ae42ce8a5c2e4ba02cf2b1ab
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aedb76c14127c0ad8eb3cf4c7740b392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ab7dfb1ae42ce8a5c2e4ba02cf2b1ab
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a07cd5dd2d2e7dfcfcefaefee1f654df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ac688ac2608622a49bc90f82839de1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a07cd5dd2d2e7dfcfcefaefee1f654df
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ac688ac2608622a49bc90f82839de1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a07cd5dd2d2e7dfcfcefaefee1f654df
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ac688ac2608622a49bc90f82839de1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a07cd5dd2d2e7dfcfcefaefee1f654df
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ac688ac2608622a49bc90f82839de1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a07cd5dd2d2e7dfcfcefaefee1f654df
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db9f8248f95f9e7491dfdcd56e903b07(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5e33f2a0841d647bdf84ac1a6b0cd45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db9f8248f95f9e7491dfdcd56e903b07
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5e33f2a0841d647bdf84ac1a6b0cd45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db9f8248f95f9e7491dfdcd56e903b07
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5e33f2a0841d647bdf84ac1a6b0cd45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db9f8248f95f9e7491dfdcd56e903b07
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5e33f2a0841d647bdf84ac1a6b0cd45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db9f8248f95f9e7491dfdcd56e903b07
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_db776e6042f403637b88884f8cda644e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.43085747957229614], [0.18588005006313324], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4131563901901245], [0.2982136011123657], [0.22558467090129852], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_e75cde4b3c3140dde738a1f797408c2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.025396505370736122], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33433759212493896], [0.12095697224140167], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_28d38bcd7a02e596ce37d069db14461b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41876405477523804], [0.0375184640288353], [0.09816955029964447], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.462017297744751], [0.11372362822294235], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_0b81d8c1508681c3bd015ce23866517d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    
    class PrimitiveOp_16c7ed96c35e9cdd4413d93b62dd067b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20e4bfd29f72cde403a46bba1c0f652b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16c7ed96c35e9cdd4413d93b62dd067b
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20e4bfd29f72cde403a46bba1c0f652b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16c7ed96c35e9cdd4413d93b62dd067b
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20e4bfd29f72cde403a46bba1c0f652b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16c7ed96c35e9cdd4413d93b62dd067b
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20e4bfd29f72cde403a46bba1c0f652b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16c7ed96c35e9cdd4413d93b62dd067b
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_08e125c44aa9fb272ad18e6a233afd9c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a662ad9acad112c87e16581558580dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08e125c44aa9fb272ad18e6a233afd9c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a662ad9acad112c87e16581558580dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08e125c44aa9fb272ad18e6a233afd9c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a662ad9acad112c87e16581558580dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08e125c44aa9fb272ad18e6a233afd9c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a662ad9acad112c87e16581558580dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08e125c44aa9fb272ad18e6a233afd9c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_eae83cc725b31429a302d0fdd5ecf614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eae83cc725b31429a302d0fdd5ecf614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eae83cc725b31429a302d0fdd5ecf614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eae83cc725b31429a302d0fdd5ecf614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_72d04221d1548db01486e7ff6b07a5bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.32824042439460754], [0.11407893896102905], [0.04877207800745964], [0.24639740586280823], [0.4660728871822357], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.3819327652454376], [0.053689487278461456], [0.17333781719207764], [0.3545892834663391], [0.06132432818412781], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_4c855bf6685f22eda2c144425f5fddaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.477491170167923], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.23870912194252014], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e55ede414881c50f6ccbab19d32a8cb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33700039982795715], [0.13599033653736115], [0.16089512407779694], [0.4550243616104126], [0.2522077262401581], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.1544954627752304], [0.3554002642631531], [0.06358063966035843], [0.38996535539627075], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_8c560ef05b27b9260b5dbada88d15250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.2988268733024597], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.3334523141384125], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e7fdf17275193b59e9f3b0617e3b9b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7fdf17275193b59e9f3b0617e3b9b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7fdf17275193b59e9f3b0617e3b9b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7fdf17275193b59e9f3b0617e3b9b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd2ce49d3a55bf74c0d514ad8b233d57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45263877511024475, 0.18129011988639832, 0.45884010195732117, 0.49139586091041565, 0.3470446765422821, 0.2975061535835266], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_92d92881f4f654d702cc5e0be767460e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.4360129237174988, 0.43007832765579224, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.35412517189979553, 0.2729951739311218, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f4e1f3884a85f6362f0299708b5d43d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.20852655172348022, 0.4761844575405121, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e1de4d669a201a2d1ae77cc53f200505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.06498825550079346, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8d9204e399323d4b17a012ef94da3f09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d9204e399323d4b17a012ef94da3f09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d9204e399323d4b17a012ef94da3f09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d9204e399323d4b17a012ef94da3f09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4c335aa1fe538561adac355f030afe4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c335aa1fe538561adac355f030afe4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c335aa1fe538561adac355f030afe4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c335aa1fe538561adac355f030afe4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8df184385a5249ac1ee8ec02577ed5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9c1dd07e9b15ccb9ff8b5fa8b4bda0ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_550d9dd31e474c8357aca02d99ccc334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_66d6b78962d9380c9c0d161020f5fa5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_935b1008220f71ab47d724080e22ddde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.2049286961555481], [0.05052289366722107], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.38781505823135376], [0.12123280763626099], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_285c6d8054a8e9e91a547e31e582b9d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_016a4507bad3bb00461b4284b2678d63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06345830112695694], [0.37558823823928833], [0.1739027202129364], [0.027879230678081512], [0.49161165952682495], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.19266913831233978], [0.3696064352989197], [0.3602122664451599], [0.343851238489151], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_1730a2e3607493133365161b31b2aa02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.13699816167354584], [0.30248284339904785], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.4184499979019165], [0.4230078458786011], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_3cf4a440642a9e1bd4cec53044534172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cf4a440642a9e1bd4cec53044534172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cf4a440642a9e1bd4cec53044534172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cf4a440642a9e1bd4cec53044534172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b09bf22604de97ae5edee1233162d555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b09bf22604de97ae5edee1233162d555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b09bf22604de97ae5edee1233162d555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b09bf22604de97ae5edee1233162d555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c87c53eebb6c10779ed2bd96cc8bafcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c87c53eebb6c10779ed2bd96cc8bafcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c87c53eebb6c10779ed2bd96cc8bafcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c87c53eebb6c10779ed2bd96cc8bafcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f404990f952b8401e6dfb6149e922ef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.08265111595392227], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.1922953873872757], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_d46be9040244eb19dcd1dc1b8971e428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.3689388930797577], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.03426346927881241], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9f7cfa08fb1aeb9fec67db5bcfc8946c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_060a64c847eef0c6a07910eb2660a67e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_0d8441e9d4d21ab140a2c0f2ad09620e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d8441e9d4d21ab140a2c0f2ad09620e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d8441e9d4d21ab140a2c0f2ad09620e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d8441e9d4d21ab140a2c0f2ad09620e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d814cae38f93c95100c48e5346bb416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d814cae38f93c95100c48e5346bb416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d814cae38f93c95100c48e5346bb416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d814cae38f93c95100c48e5346bb416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa116f4cdf252c8657e9739ee3bdad16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa116f4cdf252c8657e9739ee3bdad16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa116f4cdf252c8657e9739ee3bdad16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa116f4cdf252c8657e9739ee3bdad16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_66e8d21b92ff6c50d451fc82100bb1e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.43085747957229614], [0.18588005006313324], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4131563901901245], [0.2982136011123657], [0.22558467090129852], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d11ad7f59b73405823c7ec6588889da0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.025396505370736122], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33433759212493896], [0.12095697224140167], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_22b5ce6a82c6e75339c26a25096dc19f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41876405477523804], [0.0375184640288353], [0.09816955029964447], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.462017297744751], [0.11372362822294235], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_852d78fe1dcd3c0c7226d41997198209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_1a44b1352f67a5e38600f9981d5b6402(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a44b1352f67a5e38600f9981d5b6402(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a44b1352f67a5e38600f9981d5b6402(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a44b1352f67a5e38600f9981d5b6402(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d4ec1df27ddc4c7ae0f1fbb647bc77ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4ec1df27ddc4c7ae0f1fbb647bc77ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4ec1df27ddc4c7ae0f1fbb647bc77ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4ec1df27ddc4c7ae0f1fbb647bc77ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
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