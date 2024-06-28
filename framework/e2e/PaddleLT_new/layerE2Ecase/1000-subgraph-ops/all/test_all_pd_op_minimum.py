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


    class TestPrimitiveOp_e74b41f8f5c66d7c8984662303593d67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e74b41f8f5c66d7c8984662303593d67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e74b41f8f5c66d7c8984662303593d67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e74b41f8f5c66d7c8984662303593d67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7208f33aab00d545980e086927cc569e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.36908185482025146], [0.2961925268173218], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.06219051405787468], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4386983811855316], [0.046371277421712875], [0.0077659436501562595], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.047508757561445236], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_078f0b8851cff29d5856e5382d699e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.338043212890625], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.42142167687416077], [0.39432990550994873], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2c501f8a5175c8a923da056ec2dd0de0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18618053197860718], [0.007991598919034004], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.042280279099941254], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_7ced1ab4ac4141e41b9b2b1880cb798a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.2514757812023163], [0.4430234134197235], [0.21071745455265045], [0.002510953461751342], [0.4495491087436676], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.11843360215425491], [0.4964841306209564], [0.09898043423891068], [0.1466875672340393], [0.21377909183502197], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_30a36ff48836e253bc682afea26af02a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30a36ff48836e253bc682afea26af02a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30a36ff48836e253bc682afea26af02a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30a36ff48836e253bc682afea26af02a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a4a9ade6efc9f1bed7f35f8f3c0a904e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a78a0adc0145495417f38bf70125a6e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ea8d988a0e1e78256c9abf19d75a03b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.3307744562625885, 0.4149071276187897, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_66b3f16d6a57a72148935ecc8755b3f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.4658578038215637, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c74621a48ef473904be451ba32467675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c74621a48ef473904be451ba32467675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c74621a48ef473904be451ba32467675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c74621a48ef473904be451ba32467675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0cae788177dd6e88aca267b35da4de91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cae788177dd6e88aca267b35da4de91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cae788177dd6e88aca267b35da4de91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cae788177dd6e88aca267b35da4de91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2c851af680f75f2b704accbcb74071ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_089b7529d9d334f4bdf390e4b6d4e69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ca1999548c56c09d8edc134c99f6298b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_24ee920747f2453a8feac70463f9c8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_63d89991bf901ba70299627200f3546a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_6e4a405f3fdc13cc776c541a14970389(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.09585190564393997], [0.27821820974349976], [0.4517127275466919], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.2810487151145935], [0.3574141561985016], [0.4414989948272705], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_bd4cf1e93afc3406ef0ab32b812240d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13643445074558258], [0.08216515928506851], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.41118323802948], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2a86460a2c6dacfd8bf1f9df5389cf89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.42264029383659363], [0.03228634223341942], [0.35276615619659424], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.39316296577453613], [0.4681013226509094], [0.0868077278137207], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_ef6ce4644365f7968f5148f3f348a328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef6ce4644365f7968f5148f3f348a328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef6ce4644365f7968f5148f3f348a328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef6ce4644365f7968f5148f3f348a328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_5a560c890c4a9fc2cdf52a479240f356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a560c890c4a9fc2cdf52a479240f356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a560c890c4a9fc2cdf52a479240f356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a560c890c4a9fc2cdf52a479240f356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bedb79278a1c0c5f7298c68a79931bcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bedb79278a1c0c5f7298c68a79931bcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bedb79278a1c0c5f7298c68a79931bcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bedb79278a1c0c5f7298c68a79931bcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4c9c57ec7f0efb6d4012e2fd80b46c16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_52156daceee0795a83d07c82703d12e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.16028515994548798], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.26832517981529236], [0.4581025242805481], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_7e648d3cd603a96bd244cc0dd9dba4dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.33523043990135193], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.3625732958316803], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_a382c7155ee84c5bf7f4f02a8df485dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40533360838890076], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_b15fa1896bbd9d47c0a4acf7ac98f0ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b15fa1896bbd9d47c0a4acf7ac98f0ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b15fa1896bbd9d47c0a4acf7ac98f0ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b15fa1896bbd9d47c0a4acf7ac98f0ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_078b206d8adc5c7121c01e30696742f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_078b206d8adc5c7121c01e30696742f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_078b206d8adc5c7121c01e30696742f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_078b206d8adc5c7121c01e30696742f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3014ac0b44b3885d43e2befa63a9b3e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3014ac0b44b3885d43e2befa63a9b3e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3014ac0b44b3885d43e2befa63a9b3e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3014ac0b44b3885d43e2befa63a9b3e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_fbf2358f264863a691cd675157afca14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.021595124155282974], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09640353918075562], [0.25064805150032043], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_e3aa09e7ae32bd720aa77000bcbbb08e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.3512554168701172], [0.08861131221055984], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.22977420687675476], [0.061657778918743134], [0.2658151686191559], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d473b4fa46427fe855b76620e90261a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41484373807907104], [0.09475763142108917], [0.012042115442454815], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09736814349889755], [0.3636853098869324], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_029830b96181449ac864b99ac29b95d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6426f88ce3ae55bf126c4fd7295d43a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6426f88ce3ae55bf126c4fd7295d43a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6426f88ce3ae55bf126c4fd7295d43a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6426f88ce3ae55bf126c4fd7295d43a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_cefb9befbc32ea27043c17fb23d9923e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cefb9befbc32ea27043c17fb23d9923e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cefb9befbc32ea27043c17fb23d9923e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cefb9befbc32ea27043c17fb23d9923e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a843833b741c4f4b054e2a23ec64ca73
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_f9c9632d528ea35f85248b6ee2cea333(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22dd50e41cdef915552d8eb72605561d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9c9632d528ea35f85248b6ee2cea333
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22dd50e41cdef915552d8eb72605561d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9c9632d528ea35f85248b6ee2cea333
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22dd50e41cdef915552d8eb72605561d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9c9632d528ea35f85248b6ee2cea333
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22dd50e41cdef915552d8eb72605561d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9c9632d528ea35f85248b6ee2cea333
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_fd5da07e2ed073bffb96160c3574e6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.36908185482025146], [0.2961925268173218], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.06219051405787468], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4386983811855316], [0.046371277421712875], [0.0077659436501562595], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.047508757561445236], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_07e68fb3a46c21d1594314ef2e6a2c70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.338043212890625], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.42142167687416077], [0.39432990550994873], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_3474a4693e39db579c49198efdfb0b56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18618053197860718], [0.007991598919034004], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.042280279099941254], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_ed13113e33b27c89ddd3b97a880b8982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f74a103afccc4deed2a909c53aff8cd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.2514757812023163], [0.4430234134197235], [0.21071745455265045], [0.002510953461751342], [0.4495491087436676], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.11843360215425491], [0.4964841306209564], [0.09898043423891068], [0.1466875672340393], [0.21377909183502197], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_32ae7e1f9ae47200915e01ce5fc0b78d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5454, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_facc89ef5db03cb2cccdc0acf077d5c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32ae7e1f9ae47200915e01ce5fc0b78d
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_facc89ef5db03cb2cccdc0acf077d5c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32ae7e1f9ae47200915e01ce5fc0b78d
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_facc89ef5db03cb2cccdc0acf077d5c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32ae7e1f9ae47200915e01ce5fc0b78d
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_facc89ef5db03cb2cccdc0acf077d5c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32ae7e1f9ae47200915e01ce5fc0b78d
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0915e6a7ae8c817c5ea0f093f88c5d9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_89babd09cb611687c7def67aefc91062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_febc06be08ea2eceb29fbcfe64df02f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.3307744562625885, 0.4149071276187897, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b32ad7e70310302899c2139b3ba90f93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe950497b8328d0e0e1d3acde9543df0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.4658578038215637, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_0c33bc8ffa52e61900cf63bd8385c868(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1722, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1701c1e740f4eb20eda306e3883bec2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c33bc8ffa52e61900cf63bd8385c868
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1701c1e740f4eb20eda306e3883bec2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c33bc8ffa52e61900cf63bd8385c868
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1701c1e740f4eb20eda306e3883bec2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c33bc8ffa52e61900cf63bd8385c868
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1701c1e740f4eb20eda306e3883bec2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c33bc8ffa52e61900cf63bd8385c868
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_e6e3657caff8b5774a43d6f985bddb7a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1518, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23c7931560a62bbfb487428a20bd348b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e3657caff8b5774a43d6f985bddb7a
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23c7931560a62bbfb487428a20bd348b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e3657caff8b5774a43d6f985bddb7a
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23c7931560a62bbfb487428a20bd348b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e3657caff8b5774a43d6f985bddb7a
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23c7931560a62bbfb487428a20bd348b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e3657caff8b5774a43d6f985bddb7a
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_ad046d73eaf1b96b0c7db88fef6f99d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_74c2d14df58aa4ef9bdd36e2836fdaa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_148744e3388771702c2a72ef2725cba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_d73f5bd26dcb701b1956693f0f502101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f429859c88336af9c3bb54b5a031cd99
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
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


    class TestPrimitiveOp_0c9e86a53ea5e91ceb7d4827e4b76f67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_86d042cdd6dbfdb05be18de411b580a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.09585190564393997], [0.27821820974349976], [0.4517127275466919], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.2810487151145935], [0.3574141561985016], [0.4414989948272705], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2a12c8da6fac6228333167a1b9846bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13643445074558258], [0.08216515928506851], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.41118323802948], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_43697f24b6a6defd7e84fa861cf653db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c76d2a5dbbdfc403876257f49009587
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.42264029383659363], [0.03228634223341942], [0.35276615619659424], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.39316296577453613], [0.4681013226509094], [0.0868077278137207], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
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


    
    class PrimitiveOp_21b0b577a8278cea2fe3313c566af295(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2133, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd8f880e2df9b0da0a3e68552f2f2e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21b0b577a8278cea2fe3313c566af295
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd8f880e2df9b0da0a3e68552f2f2e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21b0b577a8278cea2fe3313c566af295
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd8f880e2df9b0da0a3e68552f2f2e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21b0b577a8278cea2fe3313c566af295
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd8f880e2df9b0da0a3e68552f2f2e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21b0b577a8278cea2fe3313c566af295
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_7ec190b9551e3ad9b411dd0249276855(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4631, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a7de7f92cce522504b690204c7a661c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ec190b9551e3ad9b411dd0249276855
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a7de7f92cce522504b690204c7a661c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ec190b9551e3ad9b411dd0249276855
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a7de7f92cce522504b690204c7a661c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ec190b9551e3ad9b411dd0249276855
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a7de7f92cce522504b690204c7a661c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ec190b9551e3ad9b411dd0249276855
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_846eb94eac8bdcc8ce2d3ca08158f9bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1039, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff2f1c341fd2b3b7a64ce9b0acc8d16f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_846eb94eac8bdcc8ce2d3ca08158f9bf
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff2f1c341fd2b3b7a64ce9b0acc8d16f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_846eb94eac8bdcc8ce2d3ca08158f9bf
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff2f1c341fd2b3b7a64ce9b0acc8d16f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_846eb94eac8bdcc8ce2d3ca08158f9bf
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff2f1c341fd2b3b7a64ce9b0acc8d16f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_846eb94eac8bdcc8ce2d3ca08158f9bf
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c93d1b4ff743cba5b85668fc64f103c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_7138fcff59d73204f8f4e416d8915406(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.16028515994548798], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.26832517981529236], [0.4581025242805481], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e6b9db8d2c888e3e5e14b70c323e47c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.33523043990135193], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.3625732958316803], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f36dd1561aed13a21ea15fedff284294(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ad12952d2b1e11faabb7b37275c96e
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40533360838890076], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
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


    
    class PrimitiveOp_13630bb95bb96bf1c644f92449065378(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2318, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_461ca96cb923b210329dde74027448f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13630bb95bb96bf1c644f92449065378
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_461ca96cb923b210329dde74027448f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13630bb95bb96bf1c644f92449065378
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_461ca96cb923b210329dde74027448f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13630bb95bb96bf1c644f92449065378
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_461ca96cb923b210329dde74027448f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13630bb95bb96bf1c644f92449065378
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e71aa5b6d69ece75d7157308bb7c2a49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2961, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1f370ec9c8879d10fc6ef718cdaefc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e71aa5b6d69ece75d7157308bb7c2a49
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1f370ec9c8879d10fc6ef718cdaefc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e71aa5b6d69ece75d7157308bb7c2a49
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1f370ec9c8879d10fc6ef718cdaefc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e71aa5b6d69ece75d7157308bb7c2a49
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1f370ec9c8879d10fc6ef718cdaefc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e71aa5b6d69ece75d7157308bb7c2a49
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3dd61aed58aadb7244b6f2f504f929e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3739, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_799a2381d0ccead0083b8781700453c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dd61aed58aadb7244b6f2f504f929e2
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_799a2381d0ccead0083b8781700453c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dd61aed58aadb7244b6f2f504f929e2
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_799a2381d0ccead0083b8781700453c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dd61aed58aadb7244b6f2f504f929e2
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_799a2381d0ccead0083b8781700453c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dd61aed58aadb7244b6f2f504f929e2
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2afa08e95daa129bf7c51bf36c7c4209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.021595124155282974], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09640353918075562], [0.25064805150032043], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_bebe764606fba45192e37a7109d4bb52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.3512554168701172], [0.08861131221055984], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.22977420687675476], [0.061657778918743134], [0.2658151686191559], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_28f61fa83671d88c064addd26b7f41e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41484373807907104], [0.09475763142108917], [0.012042115442454815], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09736814349889755], [0.3636853098869324], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_53caddccd80136fe4c47e5279e6055c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e3f130ccc6ade800c581ee85058ade
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    
    class PrimitiveOp_f29165e96e7267ad6cdcc50b54e8ab5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2013, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d24a14dba67fe4f774acaafa9820a28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f29165e96e7267ad6cdcc50b54e8ab5d
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d24a14dba67fe4f774acaafa9820a28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f29165e96e7267ad6cdcc50b54e8ab5d
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d24a14dba67fe4f774acaafa9820a28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f29165e96e7267ad6cdcc50b54e8ab5d
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d24a14dba67fe4f774acaafa9820a28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f29165e96e7267ad6cdcc50b54e8ab5d
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_4f60eb8ef93863753543850a338a908e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.minimum(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4177, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87d6bb52127585b873ffc3965b157a54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f60eb8ef93863753543850a338a908e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87d6bb52127585b873ffc3965b157a54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f60eb8ef93863753543850a338a908e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87d6bb52127585b873ffc3965b157a54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f60eb8ef93863753543850a338a908e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87d6bb52127585b873ffc3965b157a54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f60eb8ef93863753543850a338a908e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_94840ffc1be350088f9541a599cc7987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94840ffc1be350088f9541a599cc7987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94840ffc1be350088f9541a599cc7987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94840ffc1be350088f9541a599cc7987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7208f33aab00d545980e086927cc569e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.36908185482025146], [0.2961925268173218], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.06219051405787468], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4386983811855316], [0.046371277421712875], [0.0077659436501562595], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.047508757561445236], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_078f0b8851cff29d5856e5382d699e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.338043212890625], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.42142167687416077], [0.39432990550994873], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2c501f8a5175c8a923da056ec2dd0de0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18618053197860718], [0.007991598919034004], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.042280279099941254], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_7ced1ab4ac4141e41b9b2b1880cb798a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.2514757812023163], [0.4430234134197235], [0.21071745455265045], [0.002510953461751342], [0.4495491087436676], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.11843360215425491], [0.4964841306209564], [0.09898043423891068], [0.1466875672340393], [0.21377909183502197], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_01fd3ae850c32bf6b8c75066e6ae3f62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01fd3ae850c32bf6b8c75066e6ae3f62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01fd3ae850c32bf6b8c75066e6ae3f62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01fd3ae850c32bf6b8c75066e6ae3f62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4a9ade6efc9f1bed7f35f8f3c0a904e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a78a0adc0145495417f38bf70125a6e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ea8d988a0e1e78256c9abf19d75a03b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.3307744562625885, 0.4149071276187897, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_66b3f16d6a57a72148935ecc8755b3f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b6b312a9155366e9d48b3bf31711
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.4658578038215637, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_90bb3eddb35b2e8f264be67c9b7025e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90bb3eddb35b2e8f264be67c9b7025e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90bb3eddb35b2e8f264be67c9b7025e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90bb3eddb35b2e8f264be67c9b7025e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_87d0de80ce6fdd95d02d86427708680d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87d0de80ce6fdd95d02d86427708680d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87d0de80ce6fdd95d02d86427708680d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87d0de80ce6fdd95d02d86427708680d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2c851af680f75f2b704accbcb74071ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_089b7529d9d334f4bdf390e4b6d4e69f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ca1999548c56c09d8edc134c99f6298b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_24ee920747f2453a8feac70463f9c8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_63d89991bf901ba70299627200f3546a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_6e4a405f3fdc13cc776c541a14970389(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.09585190564393997], [0.27821820974349976], [0.4517127275466919], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.2810487151145935], [0.3574141561985016], [0.4414989948272705], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_bd4cf1e93afc3406ef0ab32b812240d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13643445074558258], [0.08216515928506851], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.41118323802948], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2a86460a2c6dacfd8bf1f9df5389cf89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.42264029383659363], [0.03228634223341942], [0.35276615619659424], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.39316296577453613], [0.4681013226509094], [0.0868077278137207], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_10e4c23c4b73d76ec9fb863e36ce0f25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10e4c23c4b73d76ec9fb863e36ce0f25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10e4c23c4b73d76ec9fb863e36ce0f25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10e4c23c4b73d76ec9fb863e36ce0f25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_9c82ee0f40b0aa3056a002e0f2652e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c82ee0f40b0aa3056a002e0f2652e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c82ee0f40b0aa3056a002e0f2652e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c82ee0f40b0aa3056a002e0f2652e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d30e8679d928c8617ab0431b6314168a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d30e8679d928c8617ab0431b6314168a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d30e8679d928c8617ab0431b6314168a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d30e8679d928c8617ab0431b6314168a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4c9c57ec7f0efb6d4012e2fd80b46c16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_52156daceee0795a83d07c82703d12e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.16028515994548798], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.26832517981529236], [0.4581025242805481], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_7e648d3cd603a96bd244cc0dd9dba4dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.33523043990135193], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.3625732958316803], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_a382c7155ee84c5bf7f4f02a8df485dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40533360838890076], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_e4df496a46b2e887d5be51f7f7c35503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4df496a46b2e887d5be51f7f7c35503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4df496a46b2e887d5be51f7f7c35503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4df496a46b2e887d5be51f7f7c35503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50d4abb008a3e3d801ee770264734be4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50d4abb008a3e3d801ee770264734be4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50d4abb008a3e3d801ee770264734be4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50d4abb008a3e3d801ee770264734be4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb65f47aa4617c27360b4514c01880ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb65f47aa4617c27360b4514c01880ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb65f47aa4617c27360b4514c01880ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb65f47aa4617c27360b4514c01880ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_fbf2358f264863a691cd675157afca14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.021595124155282974], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09640353918075562], [0.25064805150032043], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_e3aa09e7ae32bd720aa77000bcbbb08e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.3512554168701172], [0.08861131221055984], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.22977420687675476], [0.061657778918743134], [0.2658151686191559], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d473b4fa46427fe855b76620e90261a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41484373807907104], [0.09475763142108917], [0.012042115442454815], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09736814349889755], [0.3636853098869324], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_029830b96181449ac864b99ac29b95d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_99ab810e434d80b1b75e44b7b98b872d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99ab810e434d80b1b75e44b7b98b872d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99ab810e434d80b1b75e44b7b98b872d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99ab810e434d80b1b75e44b7b98b872d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_3628e48eea173c41f8c59cfa8227eb6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3628e48eea173c41f8c59cfa8227eb6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3628e48eea173c41f8c59cfa8227eb6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3628e48eea173c41f8c59cfa8227eb6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d39784cbb1345c66c89fb21626093f97
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
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