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
    class PrimitiveOp_d0761ba2835f7b5057e8a223f20a8253(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0, 1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5863bd7f7160c127d8a7f41d91f88bf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0761ba2835f7b5057e8a223f20a8253
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5863bd7f7160c127d8a7f41d91f88bf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0761ba2835f7b5057e8a223f20a8253
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d7e698ab95543cdc203e11e68d51b130(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6013f67c93c6c01b56ef400c71ad082(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7e698ab95543cdc203e11e68d51b130
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e082ca3402de7d4ac36edb18a90dac21(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7eae8f55a1134dcb59a58bade397d17e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e082ca3402de7d4ac36edb18a90dac21
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3e9cd33bd539da485ac129dd911bb77c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_16f39b4523655ee84269d9d9c2710704(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e9cd33bd539da485ac129dd911bb77c
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_80ce04430644b291d8789500e09f33f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccfe0afd99bf44f3abeac781f1356935(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80ce04430644b291d8789500e09f33f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c7ed04d08a69b622618157066b261198(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c82d199c02101869dd3529103ae575b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ed04d08a69b622618157066b261198
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c20e00b06c69049cb001c68eff76bdaa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_878765e0b5b8d35055faa817a5fee75b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c20e00b06c69049cb001c68eff76bdaa
        def get_inputs(self):
            return [
                paddle.uniform([4, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6883ebbb7ba17fef9bab5b000d6263ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_66eba2edea612d6875bd8166e2f52c84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6883ebbb7ba17fef9bab5b000d6263ce
        def get_inputs(self):
            return [
                paddle.to_tensor([0.26202839612960815, 0.3870741128921509, 0.4544948637485504, 0.17211998999118805], dtype='float32').reshape([4]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_deff3f0901180340f28b77272265fd79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 9, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_216f5d39da6d11813b831be90e037ff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_deff3f0901180340f28b77272265fd79
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_b8d216718f5a450b8c32979f7ba44734(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ed24432df0a996010b271f76b1b0d34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8d216718f5a450b8c32979f7ba44734
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_51262104195d12dfe1c3be78f6124b70(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e604d7161f4b41908182a66284be08b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51262104195d12dfe1c3be78f6124b70
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_e5771af9ff8ab4329bd78d726160662b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c414a036584af233dcc64dd0245efed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5771af9ff8ab4329bd78d726160662b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7e3607bf0faa62f7627eea39e2925d50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5d76491c7673d563c98740aee09bbdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e3607bf0faa62f7627eea39e2925d50
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_072efdeb3794620367dbf0104a51f389(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c236ab62a2006d7d52d972943fff38b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dcb37142a6aaa008e7a6418a4241ec63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ed04d08a69b622618157066b261198
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_73553e87faea3423db74308c60af8cda(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3fcea0e83e3303a7c8b617f8286a31a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73553e87faea3423db74308c60af8cda
        def get_inputs(self):
            return [
                paddle.uniform([21, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e604d7161f4b41908182a66284be08b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51262104195d12dfe1c3be78f6124b70
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1c414a036584af233dcc64dd0245efed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5771af9ff8ab4329bd78d726160662b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dcb37142a6aaa008e7a6418a4241ec63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ed04d08a69b622618157066b261198
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1fac487653c3344c18ed6f065cfb267d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c902b736080ddd7856687c39a718a1d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fac487653c3344c18ed6f065cfb267d
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_2118bae44bec70985c665db0e1c54c49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41e7905e89cf17e7a843fe1f606cc80a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2118bae44bec70985c665db0e1c54c49
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.48046234250068665, 0.22905594110488892], [0.2770739495754242, 0.3431546986103058], [0.24031347036361694, 0.48387277126312256], [0.4746449291706085, 0.4006763994693756], [0.10624674707651138, 0.05708790943026543], [0.49964481592178345, 0.2771390378475189]]], dtype='float32').reshape([1, 6, 2]),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a5060ca4f7e6ec2a85ca57ead85f0397(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 49, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee7cec234d9dc2f7811e38bde8fee386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5060ca4f7e6ec2a85ca57ead85f0397
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0dc603decea2354a00fba53893fc737(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2d793d46bcbf9e642e9b7c837f7998e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_86839e16372832921628c5526685dedb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56e57911b67a2e651cc1c5e8d7807f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86839e16372832921628c5526685dedb
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6aff7f1398b57660bbca6a34c02e9839(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b7bb0e3e99e192beec3fc18ae89bd31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6aff7f1398b57660bbca6a34c02e9839
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_649e36a95497576be99ffd439eab018d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05cbfd68c200e2b24e8b85cd819da18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_649e36a95497576be99ffd439eab018d
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d71730ec8eb5f5aabab0204dcb63cc91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_541cb19fcd651b61c4e67b815a9547de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d71730ec8eb5f5aabab0204dcb63cc91
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1097528263926506, 0.4775157868862152, 0.11485323309898376, 0.08862581104040146]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([-2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a03fff89ae841e00fb102e141c0001dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_326da642ff8e8ec9b349b72736b3cd19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a03fff89ae841e00fb102e141c0001dd
        def get_inputs(self):
            return [
                paddle.uniform([300, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9019949e252022421c9dc758f82355a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9019949e252022421c9dc758f82355a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a349e39b8cd164285e80dd74f203bedd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fac487653c3344c18ed6f065cfb267d
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_bb1a92e75bf677bf41d16bda0dec46a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_309a0f329fa2187d1bd9f19d20998463(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb1a92e75bf677bf41d16bda0dec46a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_10c88c9d78f11cfe7fe57047aac814ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0761ba2835f7b5057e8a223f20a8253
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_10c88c9d78f11cfe7fe57047aac814ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0761ba2835f7b5057e8a223f20a8253
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8c64b821d74483afb70f3503a1147d54(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 21], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_294e7d59f310a8d20c8313fc59251f9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c64b821d74483afb70f3503a1147d54
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e33f8a48ee0f88c0059d6834a0dc17d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c20e00b06c69049cb001c68eff76bdaa
        def get_inputs(self):
            return [
                paddle.uniform([3, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f28cfbd439faf9796c3c9858f121604e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6883ebbb7ba17fef9bab5b000d6263ce
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4758526086807251, 0.22843202948570251, 0.1572914719581604], dtype='float32').reshape([3]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_895751d5e0c7c483cdd16b15258c3db4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e3607bf0faa62f7627eea39e2925d50
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b23867ce4c70e503307751b2bb1ea1a6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 872], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_924feabea68590eed62e92f5eac2df79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b23867ce4c70e503307751b2bb1ea1a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3f448c77796847c02dc02047769a0a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_ebf00d6e962b5f84e089248e789c3484(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_463eb7537cf517060e22c02c03be5498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([1696], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b0df5f714f9237ab4dcc3bb43a9b648c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da0ca8ba912aaaa76772fbf6844a77c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1696, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_da0ca8ba912aaaa76772fbf6844a77c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1696, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_466d6cfb6acf01e2e45e7bb2a611d55d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0761ba2835f7b5057e8a223f20a8253
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_466d6cfb6acf01e2e45e7bb2a611d55d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0761ba2835f7b5057e8a223f20a8253
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5abbec3c264923e9a6c314ad7aed82fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 4, 49, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eb49d867a5bf796510d7ce39046916f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5abbec3c264923e9a6c314ad7aed82fd
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_edadf8a3cafc2a4225ec595e9394786b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ed04d08a69b622618157066b261198
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8b3fde427dc5f53a51ac559db1abe760(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a9b9b2b71d80627780fbcc26c3f0359(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3fde427dc5f53a51ac559db1abe760
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0c3534eda51b1453fefb4295ba18a370(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86839e16372832921628c5526685dedb
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b3a5a98f2785b0c07aff70bb7b27f54d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f8912cdbce1f436cd47be3406db428a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([5517], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d886041c0dc8bbec38169e1b0416477a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5f2b5580b1fbc2ae044fae47d191bd60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5517, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5f2b5580b1fbc2ae044fae47d191bd60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5517, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3cf688041e410d093fe20db62423cd2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3cf688041e410d093fe20db62423cd2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0c82d199c02101869dd3529103ae575b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ed04d08a69b622618157066b261198
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a8c060ab84becb8413f64f51eb107679(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 49, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41b956ab14dbec59a9715121779a0fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8c060ab84becb8413f64f51eb107679
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_62617dd6d1d067d2ecc854eaeab69f70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73553e87faea3423db74308c60af8cda
        def get_inputs(self):
            return [
                paddle.uniform([19, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_67ab446a9733f3c944547fe6ddb00756(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78cd825476da715bf8bab3f91bb9df50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67ab446a9733f3c944547fe6ddb00756
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a402b3278f5323b705e5552f23403242(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80ce04430644b291d8789500e09f33f7
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d3f448c77796847c02dc02047769a0a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e78dd1bb548e5039ee958f99d5f39189(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([1794], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b0df5f714f9237ab4dcc3bb43a9b648c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a59b21b3ab23260f25448271811f628b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1794, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a59b21b3ab23260f25448271811f628b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1794, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9c16fdfceebe4d3c8ab5a001a5dc1b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ed04d08a69b622618157066b261198
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_aa1d30f9a8dee86ce73ce4de4673982f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86839e16372832921628c5526685dedb
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_edadf8a3cafc2a4225ec595e9394786b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ed04d08a69b622618157066b261198
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6388f616cafc9e1e6bbacc3e2bdd1806(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[512], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0120550622709ca86cbbfd20f00f287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6388f616cafc9e1e6bbacc3e2bdd1806
        def get_inputs(self):
            return [
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3861b23e354a0951c6d74f391e6f50e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8aa5ffb432a44f7ec0c9dbc17cfc8de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3861b23e354a0951c6d74f391e6f50e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_39067dca5502e36a85f1d62df14bbf53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67f819896c71d9aa32cab60613a2eb90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39067dca5502e36a85f1d62df14bbf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_061d84aa7611e777b423e5c27e5be4b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 8, 49, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15394c1893581eccba97743146b0b684(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_061d84aa7611e777b423e5c27e5be4b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3dc5f394ba7c3d611ceb94c0383a1be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_29990316f784ae2aa9a8e15aef231018(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5cdcfd91c5511996d185b433e4aa938f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e3607bf0faa62f7627eea39e2925d50
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2338cf7a47ba2021e679d1188afa8260(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cdc0fccc384696ff2dfb938d0ce4c9fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([1504], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4af2f6f808282e8192d8ae086c134d01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_11d59afbe928c605e5f8045f96e7724f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1504, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_11d59afbe928c605e5f8045f96e7724f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1504, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dde2d24b058a7eb181065ca42f7d0d03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8d216718f5a450b8c32979f7ba44734
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_e5bd46efda03bbe11119803b04242786(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 16, 49, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f4fd76e06f7d07c70763b54c53f195b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5bd46efda03bbe11119803b04242786
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_92f5fabd6bf06bc8ec815948e4ba72cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86839e16372832921628c5526685dedb
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bb72f79fef7b826afe429b4a1873fd94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_678d78147c1ed33beaa89a116a81b8fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a349e39b8cd164285e80dd74f203bedd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fac487653c3344c18ed6f065cfb267d
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_309a0f329fa2187d1bd9f19d20998463(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb1a92e75bf677bf41d16bda0dec46a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_412845550fe11316b8cf3b669d4ed51b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95506f0af326552137bfce9295507b31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_412845550fe11316b8cf3b669d4ed51b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_04bfacc95b543a27539af797eabcd410(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67ab446a9733f3c944547fe6ddb00756
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9f4fd76e06f7d07c70763b54c53f195b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5bd46efda03bbe11119803b04242786
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a349e39b8cd164285e80dd74f203bedd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fac487653c3344c18ed6f065cfb267d
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_309a0f329fa2187d1bd9f19d20998463(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb1a92e75bf677bf41d16bda0dec46a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ae7050a3b1cd7b130b2dafa621d636cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e3607bf0faa62f7627eea39e2925d50
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4be16c09b5be75d9347a03c446b1368c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4be16c09b5be75d9347a03c446b1368c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d7db0848c978834a29c11f976d50998f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce8684dceb232c27aadce0aec718636a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7db0848c978834a29c11f976d50998f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_821039fdf57a04e7002c20e8fbf2a374(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8d216718f5a450b8c32979f7ba44734
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9c16fdfceebe4d3c8ab5a001a5dc1b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ed04d08a69b622618157066b261198
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_15394c1893581eccba97743146b0b684(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_061d84aa7611e777b423e5c27e5be4b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1ae31fb75aca8e3798420e542bbfc98b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_36555804f8ff2f45f296073a74fa6c7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([2039], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d6890c2a49bd56d1fd6b723c0811aec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_395b729ae62421217d270cb568d92842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2039, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_395b729ae62421217d270cb568d92842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2039, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6eb2644ce0ffbbbfdad55f54b46b1044(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47e6e2cc9836bbc82f3b7285ff50b08c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([4584], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_81c8305a6fa449dee64c05b4983c5680(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d9f393c615c406a2ed32628113f65479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4584, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d9f393c615c406a2ed32628113f65479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4584, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cdcfc248c22892c1be6240d6feee10e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c20e00b06c69049cb001c68eff76bdaa
        def get_inputs(self):
            return [
                paddle.uniform([6, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7564f062bb99200c542008bfb1fc948a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6883ebbb7ba17fef9bab5b000d6263ce
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2733139097690582, 0.4986962676048279, 0.030321704223752022, 0.3326023817062378, 0.11917427182197571, 0.408772349357605], dtype='float32').reshape([6]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6379ee09eaa2c0f1ea7bd1b172f9eca0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cee79040e7db4f5dad273b1b438be682(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9979a30f2e25cff060ea1d1c5c1440de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([1071], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c08962d4c23e151fb0ebea4fb9ffd7bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a6151ef6c2e9b46a7a0eece349031208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1071, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a6151ef6c2e9b46a7a0eece349031208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1071, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_aaabf4c036a61a4138aee752dfd2c16b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19689b8bb80f1958233fefab3d40f4a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aaabf4c036a61a4138aee752dfd2c16b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.07161086797714233, 0.04861365258693695, 0.19623209536075592, 0.01351057831197977], dtype='float32').reshape([4]),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_6aedfb10ba48ce43233ead3454da21e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_548c8e6654ed3729f96e3837a4f167cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6aedfb10ba48ce43233ead3454da21e9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07161086797714233, 0.04861365258693695, 0.19623209536075592, 0.01351057831197977]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_340b5a6edfdcb6062eb6ef2d20968356(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de3431a95a196e4905ed03d1d25dc7af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_340b5a6edfdcb6062eb6ef2d20968356
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8c7db28f378dbefae209d7619f090545(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b3fde427dc5f53a51ac559db1abe760
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_37e359d2c29aa1eb29b3ad8623bb8f2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 49, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_893ff26cbd0f0405de64166f91828bac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37e359d2c29aa1eb29b3ad8623bb8f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a349e39b8cd164285e80dd74f203bedd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fac487653c3344c18ed6f065cfb267d
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_309a0f329fa2187d1bd9f19d20998463(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb1a92e75bf677bf41d16bda0dec46a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_61e79727633960898c038151d294e383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aaabf4c036a61a4138aee752dfd2c16b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3680858016014099, 0.10794095695018768, 0.2429090440273285, 0.47353971004486084], dtype='float32').reshape([4]),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_068c928bde5b54fc74c24d1f1c894090(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6aedfb10ba48ce43233ead3454da21e9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3680858016014099, 0.10794095695018768, 0.2429090440273285, 0.47353971004486084]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_caa32f254751b010fead257e090e61fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39d3b483973b8b35d225f2bda34db81c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_caa32f254751b010fead257e090e61fc
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f942349540872d8a67df77f08ec5b528(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1248], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c746bd5426cacc33278dd2a5caa5de6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f942349540872d8a67df77f08ec5b528
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_76aec4851cb222dc1ed03018c3f50f41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80ce04430644b291d8789500e09f33f7
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b381d0b89610878fcc99ebbd401548ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67ab446a9733f3c944547fe6ddb00756
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fd7f20566171765a2c25066ac5d62d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0761ba2835f7b5057e8a223f20a8253
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fd7f20566171765a2c25066ac5d62d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0761ba2835f7b5057e8a223f20a8253
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4e1d8d6d28cd40941bc89357a2c10d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4e1d8d6d28cd40941bc89357a2c10d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3d6a69b9735b3818d3f14021ce98f916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d6a2e60f2f68ea545e86c75a12648530(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([2370], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bf75898295ab54ef9ad26eef4ab3480a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_db2fe69be657603d0e5dddfc1e365a3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2370, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_db2fe69be657603d0e5dddfc1e365a3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2370, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_03f580e93bbb524c0d3f8fbada66a6ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_03f580e93bbb524c0d3f8fbada66a6ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_df7eb752b1db0f2d8fd8923ccfaa477f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_56414f0c595958c51c7e47813988ebef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([2993], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_00670fff28d386ee0dee66d6d9d027c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2993ed895b15a4a9db6a7088524616e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2993, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2993ed895b15a4a9db6a7088524616e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2993, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e2907e236e1f62816d9912d6809a654a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d8ea12b2aea74d177ad4fe498d3ae59f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([3832], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bfa31eed51f515b634d7e81151977da2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d08b0e7bc47a90f44c41bf7e0aa6076e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3832, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d08b0e7bc47a90f44c41bf7e0aa6076e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3832, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_92d513315229374692a68bd97f40b253(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_92d513315229374692a68bd97f40b253(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f9a8b0f6e4a5b550e15fe89dbd9d4bde(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a875c8f4c255ea5003880d217aabb10f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9a8b0f6e4a5b550e15fe89dbd9d4bde
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f6c677026ebbabe4fffb1db203e58003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f6c677026ebbabe4fffb1db203e58003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74ec71bc579a6310d67441336b5ddc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_924feabea68590eed62e92f5eac2df79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b23867ce4c70e503307751b2bb1ea1a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4e377bb835402864478cd30cb0204e44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 49, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5274237ef0a914480809c3ac03ab7159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e377bb835402864478cd30cb0204e44
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3b5cb3d05c0e3aee6ecb24ee73e3bc85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80ce04430644b291d8789500e09f33f7
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c6325c08cc783125b17b0340773e34a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80ce04430644b291d8789500e09f33f7
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d0106965c8cfa9b7a9e917aa5fc105d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c20e00b06c69049cb001c68eff76bdaa
        def get_inputs(self):
            return [
                paddle.uniform([2, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c7481a994d57fb85e3f7cbc34f996b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6883ebbb7ba17fef9bab5b000d6263ce
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3661021292209625, 0.4057813286781311], dtype='float32').reshape([2]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_798513876b0fd744ff63ee0ff27b5d88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67ab446a9733f3c944547fe6ddb00756
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bca17977c2684f6913a8eed7b902dae7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 49, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fad61e157bf0fda05b216fe0ba064431(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bca17977c2684f6913a8eed7b902dae7
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_184ce990a2440d0e0857fb3bdbe1ff69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c4018709ba0fc4c29ad0def744be1a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_184ce990a2440d0e0857fb3bdbe1ff69
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_41b956ab14dbec59a9715121779a0fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8c060ab84becb8413f64f51eb107679
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_77b66b9df75373c01e3fc1b8f3b79876(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_46436268fc5af686ec04ac87efc072c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4885adf2b7da5dd17971675f46e50d6d
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ce8684dceb232c27aadce0aec718636a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7db0848c978834a29c11f976d50998f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5274237ef0a914480809c3ac03ab7159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e377bb835402864478cd30cb0204e44
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1ae31fb75aca8e3798420e542bbfc98b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4f5b62b6dc8384cac1b687bf7752435d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([1995], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d6890c2a49bd56d1fd6b723c0811aec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5c836dbd0437fcb36c7f0eb4a2678a91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5c836dbd0437fcb36c7f0eb4a2678a91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e604d7161f4b41908182a66284be08b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51262104195d12dfe1c3be78f6124b70
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1c414a036584af233dcc64dd0245efed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5771af9ff8ab4329bd78d726160662b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_22263194280dd661d1fdcca32880f7dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2, 9, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68b5856d3a6c732c77d40a1ce303c40d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22263194280dd661d1fdcca32880f7dd
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_893ff26cbd0f0405de64166f91828bac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37e359d2c29aa1eb29b3ad8623bb8f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_244827ce7898643fbbcdb78cdb8528d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73553e87faea3423db74308c60af8cda
        def get_inputs(self):
            return [
                paddle.uniform([150, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8b0cd30e04e4c658c51593ac19fd5048(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_573d433260e699e1ee02eddc7fcd0c29
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f482ed90469c820eeb96391da3274f67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf00d6e962b5f84e089248e789c3484
        def get_inputs(self):
            return [
                paddle.uniform([4181], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_77bf21ad16b01415fa7dcb9bd49b1f3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_072efdeb3794620367dbf0104a51f389
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_db0bff8963d8df5d92214a499170da6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4181, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_db0bff8963d8df5d92214a499170da6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f47d362ce3bac3750575bfb6dd034b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4181, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3a01eabbe442de2f62522d1c2b519f23(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 624], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbb63d4b187cdbba165df0b7add36f06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a01eabbe442de2f62522d1c2b519f23
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fad61e157bf0fda05b216fe0ba064431(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bca17977c2684f6913a8eed7b902dae7
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_8fa628a6528b4272d83cdb7031f88ec1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0, 1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7eef4a91e3fda7e2135aebfb16219559(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa628a6528b4272d83cdb7031f88ec1
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7eef4a91e3fda7e2135aebfb16219559(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa628a6528b4272d83cdb7031f88ec1
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4b89b1fcbd16e29812ad86fe47f96907(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dfc52376b3d510fe337fee44acaaad55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b89b1fcbd16e29812ad86fe47f96907
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1bb1623cd652fb2040f9a17b1ba18efa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd3c046c6f68de628bea69bd8d792a93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bb1623cd652fb2040f9a17b1ba18efa
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_66e455de3aaedc143b4894bb1a77384b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fef534b3e8eb96954402d5b28c00c7ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66e455de3aaedc143b4894bb1a77384b
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9d85d843c8b4c41f36e736bad34a8444(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d136cda43de59d43899da39aebcaa805(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d85d843c8b4c41f36e736bad34a8444
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e156a84843304ec3824fdfac7b0df4ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81358b37e2be301a4fae64c1f5e8206d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e156a84843304ec3824fdfac7b0df4ef
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1938c7486a3ba3007c63c85ab2fcfd42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2ab497c0a0d2624e4b1eeeefdb60dde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1938c7486a3ba3007c63c85ab2fcfd42
        def get_inputs(self):
            return [
                paddle.uniform([4, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_36f40ab81e330a1f313bbba0845b1cf9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55c87628867a26b444206de6e8a9899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36f40ab81e330a1f313bbba0845b1cf9
        def get_inputs(self):
            return [
                paddle.to_tensor([0.26202839612960815, 0.3870741128921509, 0.4544948637485504, 0.17211998999118805], dtype='float32').reshape([4]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_216f5d39da6d11813b831be90e037ff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_deff3f0901180340f28b77272265fd79
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a9e1ad2578d6988fc2b8376ea8fc6275(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e43aa741aa6e0630693ffbfe6bfbb096(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9e1ad2578d6988fc2b8376ea8fc6275
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c9ba52da61159e6075964964fd5ef524(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 500], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbf00884ccb7247204d663d477c76b8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9ba52da61159e6075964964fd5ef524
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3b343977f234156d212017ce3cc39d1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 500], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_251006bb840241558ed401f48e9ba56f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b343977f234156d212017ce3cc39d1b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_707741bdfbee5b1a06864d282e12c91b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0bee6035d619f1b14195e24981bd99aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_707741bdfbee5b1a06864d282e12c91b
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_70576218a90072df00bd0b20c306f4dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8732], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a20ce336d93f1d3c38e7056b2a1c057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70576218a90072df00bd0b20c306f4dd
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9223ba8483de344603ac0adae8e1b022(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4100ed7160444cb0e6cb81bc430ae4d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9223ba8483de344603ac0adae8e1b022
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_785a787cc274c060fe55b8e79d12eddb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[21, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e64f9fceac4709d5643ec7b1e067bc11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_785a787cc274c060fe55b8e79d12eddb
        def get_inputs(self):
            return [
                paddle.uniform([21, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dbf00884ccb7247204d663d477c76b8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9ba52da61159e6075964964fd5ef524
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_251006bb840241558ed401f48e9ba56f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b343977f234156d212017ce3cc39d1b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4100ed7160444cb0e6cb81bc430ae4d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9223ba8483de344603ac0adae8e1b022
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3e23ef00f10ab7aaf5fb54daf7b66139(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21824, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67c9c13ea98e53f8f9fdac226be41c22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e23ef00f10ab7aaf5fb54daf7b66139
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_aaea8fedd9cacec26d08f68d72ce2c54(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9fb0b5c16c7366653b9e599ec5a628bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aaea8fedd9cacec26d08f68d72ce2c54
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.48046234250068665, 0.22905594110488892], [0.2770739495754242, 0.3431546986103058], [0.24031347036361694, 0.48387277126312256], [0.4746449291706085, 0.4006763994693756], [0.10624674707651138, 0.05708790943026543], [0.49964481592178345, 0.2771390378475189]]], dtype='float32').reshape([1, 6, 2]),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ee7cec234d9dc2f7811e38bde8fee386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5060ca4f7e6ec2a85ca57ead85f0397
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_53fbc6fd4d5a8266e041fb0bf11ef689(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a4178dd70395492e39e9a460b516a89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53fbc6fd4d5a8266e041fb0bf11ef689
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e6cc0fb768a821ec79572a62b7a05fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53fbc6fd4d5a8266e041fb0bf11ef689
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a9eb6887c63bce00025f15a964f5fdd9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcdd09fddaa7c5c3ce3de046fa61760e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9eb6887c63bce00025f15a964f5fdd9
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_16067c70b8919c40c9eea333c8bcacd6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05c3da882b08dbd001f21f3bd90950e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16067c70b8919c40c9eea333c8bcacd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_05cbfd68c200e2b24e8b85cd819da18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_649e36a95497576be99ffd439eab018d
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_42abd81b8b4c02728a197a560a9db636(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3cb8a7daff2e008367442786d89a4eb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42abd81b8b4c02728a197a560a9db636
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1097528263926506, 0.4775157868862152, 0.11485323309898376, 0.08862581104040146]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([-2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_326da642ff8e8ec9b349b72736b3cd19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a03fff89ae841e00fb102e141c0001dd
        def get_inputs(self):
            return [
                paddle.uniform([300, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7e8cc53b075e5054db134d8482511e0d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 36, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9f779fce281cce637f2a13722deb755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e8cc53b075e5054db134d8482511e0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d9f779fce281cce637f2a13722deb755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e8cc53b075e5054db134d8482511e0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4cac9f9f1f2d37dec844370a5e372c0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[512, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_494b3dea2ade2da8e1b23419346b826d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cac9f9f1f2d37dec844370a5e372c0b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_68a307491a2890b2c4a68789333e22ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bae096cfea6474922f864b41f18f585c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68a307491a2890b2c4a68789333e22ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f38a30b9e22255fd18be40c578c4b73d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0, 1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f256c29e4f0db2b7b5a369df08c37b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f38a30b9e22255fd18be40c578c4b73d
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f256c29e4f0db2b7b5a369df08c37b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f38a30b9e22255fd18be40c578c4b73d
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e1cddf5183b3463fb9653fd15491818a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 21], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_478f923c5d61830dbf573a03aa089e4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1cddf5183b3463fb9653fd15491818a
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_fcbf71e8a6282e37d6dc4b4951230977(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e26f6e61ea21d0a6496b66c8847ca79d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcbf71e8a6282e37d6dc4b4951230977
        def get_inputs(self):
            return [
                paddle.uniform([3, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_80ba0bd6475ff485fc78aa215123a2a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05e4fdb699683cac5e423ae9da54cf0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80ba0bd6475ff485fc78aa215123a2a7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4758526086807251, 0.22843202948570251, 0.1572914719581604], dtype='float32').reshape([3]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_62b6571904a2fe4c26fa72b1f519db66(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d60fb148c815ef7cac862f60ecbe50e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62b6571904a2fe4c26fa72b1f519db66
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_708ad435b1a8fc06d850ba4c092ac32e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fce13e55a8cf39ad65bb4288015f8830(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_708ad435b1a8fc06d850ba4c092ac32e
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2a2308d0b2101e667f240170d52da0a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a8fcd8d1e8ac4b1b85c760862d425d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a2308d0b2101e667f240170d52da0a9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_97fb415c90594f756546d5ee9a6d566c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b0ba55788e328ea188a6bddb1e1405b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97fb415c90594f756546d5ee9a6d566c
        def get_inputs(self):
            return [
                paddle.uniform([1696], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7670f1867f0d103fe28d0cc901bca014(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de116e433779222f2db916eb896f9932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7670f1867f0d103fe28d0cc901bca014
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_01e062c60ee45d1777a5f726edb7a388(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d6aed4f8fb8fdf1c0e0dbe335fd87d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01e062c60ee45d1777a5f726edb7a388
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1696, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4d6aed4f8fb8fdf1c0e0dbe335fd87d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01e062c60ee45d1777a5f726edb7a388
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1696, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_5ef28727cb276b18fd0dcdf53677750a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0, 1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a75577f36014c2c4231a842951f3c8c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ef28727cb276b18fd0dcdf53677750a
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a75577f36014c2c4231a842951f3c8c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ef28727cb276b18fd0dcdf53677750a
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_eb49d867a5bf796510d7ce39046916f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5abbec3c264923e9a6c314ad7aed82fd
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_116d8a9fade8b4daf52992a45b45f6f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_97bd9a58d05d402ab033739228dcb9e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_116d8a9fade8b4daf52992a45b45f6f7
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_61161465cd0c6dc1f098760e709fe714(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 768, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4217d086ad438f0b7f5ebf01f4fdc981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61161465cd0c6dc1f098760e709fe714
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c99e754342a62f42c9e55e7fe16f55d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12734d7538e9c0da7cc703b8e07e2476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c99e754342a62f42c9e55e7fe16f55d6
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9e66b774e3e55e2aef0fbc0ab6e9110d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f2727be19242b463b9d66e0b4a43196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e66b774e3e55e2aef0fbc0ab6e9110d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7aaf40ef92b2079c80b7f58fe059e943(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99c49b911166b781323b2416ea30b6d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aaf40ef92b2079c80b7f58fe059e943
        def get_inputs(self):
            return [
                paddle.uniform([5517], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_0f3f6342b39e98d2e24940bc7019db72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51bfca63bc6b77b9785fe6790e91c0cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f3f6342b39e98d2e24940bc7019db72
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c482ca86b246be3b598b8b2b355fdaf3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4790afd22244c952a48393d68d141b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c482ca86b246be3b598b8b2b355fdaf3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5517, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f4790afd22244c952a48393d68d141b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c482ca86b246be3b598b8b2b355fdaf3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5517, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_eb12ee7a60ef6a65a44e2b08eaf24b8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_312362e448d097a00aa21fc04215cec2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb12ee7a60ef6a65a44e2b08eaf24b8f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_312362e448d097a00aa21fc04215cec2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb12ee7a60ef6a65a44e2b08eaf24b8f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_81358b37e2be301a4fae64c1f5e8206d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e156a84843304ec3824fdfac7b0df4ef
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_41b956ab14dbec59a9715121779a0fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8c060ab84becb8413f64f51eb107679
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f50f8fad79c5e142485749209ba69efa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[19, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b498bbd50c7276c12a061cbfc9e15478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f50f8fad79c5e142485749209ba69efa
        def get_inputs(self):
            return [
                paddle.uniform([19, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_98e818a68817e95aeb4260e834cd9042(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24d5924516ceef470a68991e716aad38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98e818a68817e95aeb4260e834cd9042
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_56df5a4543ed8e85dac4752ffcbfc455(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6df8d2342a253773e97da35b051d8d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56df5a4543ed8e85dac4752ffcbfc455
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0a8fcd8d1e8ac4b1b85c760862d425d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a2308d0b2101e667f240170d52da0a9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_2713792fb01cd946e89893e8d10d4e97(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5a634fcd2a026db6f8d95e61bf99920f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2713792fb01cd946e89893e8d10d4e97
        def get_inputs(self):
            return [
                paddle.uniform([1794], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_de116e433779222f2db916eb896f9932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7670f1867f0d103fe28d0cc901bca014
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c6ad7f11f26daef73cf2f4662d7893ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f0df19daa6601f60d11819cd3c6a160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6ad7f11f26daef73cf2f4662d7893ea
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1794, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9f0df19daa6601f60d11819cd3c6a160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6ad7f11f26daef73cf2f4662d7893ea
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1794, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_64fc8442602e04cfa855b4e0cb508211(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_977d6ec2a927a2e62d2a873706571a1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64fc8442602e04cfa855b4e0cb508211
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_92c1bb6878b7e8c5044173d0cc8dcf5e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ddd357e9329fa8559ff06f95111994cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92c1bb6878b7e8c5044173d0cc8dcf5e
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_97bd9a58d05d402ab033739228dcb9e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_116d8a9fade8b4daf52992a45b45f6f7
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c0120550622709ca86cbbfd20f00f287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6388f616cafc9e1e6bbacc3e2bdd1806
        def get_inputs(self):
            return [
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c8aa5ffb432a44f7ec0c9dbc17cfc8de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3861b23e354a0951c6d74f391e6f50e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_67f819896c71d9aa32cab60613a2eb90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39067dca5502e36a85f1d62df14bbf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_15394c1893581eccba97743146b0b684(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_061d84aa7611e777b423e5c27e5be4b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c4c97de8034e969733c99bf3efa45565(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ea51678950650cc5d6537d5b2d883fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4c97de8034e969733c99bf3efa45565
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4e628e6f28839fd3f5d2b7dcb44176f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4c97de8034e969733c99bf3efa45565
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9cef472278292306a5c27bfebedb1612(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c86fd8fbd010838e97a4694c9793a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cef472278292306a5c27bfebedb1612
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d80b16427bd62bde0894d98981d6cdf1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d3d8939649c12205580ce7f8e53afc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d80b16427bd62bde0894d98981d6cdf1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a8b07ac9943028f8254c3dd5bac84f26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89c8ba43681ed06d9f0db021ce0e6915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8b07ac9943028f8254c3dd5bac84f26
        def get_inputs(self):
            return [
                paddle.uniform([1504], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_8d1ac5b095559a565b2a92096eb971aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7d9990a5ad6ef205ebcdab034cbce38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d1ac5b095559a565b2a92096eb971aa
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_90fd6f6e4da5efe14987d6234d90aae6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1de7b1a75a75258d129e2542c2566bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90fd6f6e4da5efe14987d6234d90aae6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1504, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1de7b1a75a75258d129e2542c2566bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90fd6f6e4da5efe14987d6234d90aae6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1504, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_1c6d97f0d49f1dd652a76794098b4967(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c8ee260e82074b3a53b02cabc2f89f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c6d97f0d49f1dd652a76794098b4967
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9f4fd76e06f7d07c70763b54c53f195b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5bd46efda03bbe11119803b04242786
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_073b5a64fece2e63836a709f0957d842(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d476bbf5d2f6e32b4994389457e339c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_073b5a64fece2e63836a709f0957d842
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1927a27307439fc5df5d0290ccb6eec0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb824b1c9a273ba5428eb24c5bf83668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1927a27307439fc5df5d0290ccb6eec0
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_343b604055b3b2eafbdff77d6272000f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1927a27307439fc5df5d0290ccb6eec0
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_494b3dea2ade2da8e1b23419346b826d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cac9f9f1f2d37dec844370a5e372c0b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bae096cfea6474922f864b41f18f585c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68a307491a2890b2c4a68789333e22ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_95506f0af326552137bfce9295507b31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_412845550fe11316b8cf3b669d4ed51b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_730ff1540797c1eeb809d88268c97581(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_976b0b454144eb44fb48ef3f68beb4ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_730ff1540797c1eeb809d88268c97581
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9f4fd76e06f7d07c70763b54c53f195b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5bd46efda03bbe11119803b04242786
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_494b3dea2ade2da8e1b23419346b826d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cac9f9f1f2d37dec844370a5e372c0b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bae096cfea6474922f864b41f18f585c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68a307491a2890b2c4a68789333e22ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4c150bd2d6884811a6a16125cabae366(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_66d06ed17faadfc0e62fdb6058abb241(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c150bd2d6884811a6a16125cabae366
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b57c993307258e606f789fab415a2baf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02c057c59fb9c49ae4bc38e672dc9139(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b57c993307258e606f789fab415a2baf
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_02c057c59fb9c49ae4bc38e672dc9139(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b57c993307258e606f789fab415a2baf
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_779dfa7719a6be81fa7e5a497b02a623(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_189abec98ab3d18eef16e7392673c36b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_779dfa7719a6be81fa7e5a497b02a623
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_cd2aebdfb8e7b388c27a459e2c3b91e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ccba1171540c6a3eb35b41d4f0766c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd2aebdfb8e7b388c27a459e2c3b91e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_977d6ec2a927a2e62d2a873706571a1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64fc8442602e04cfa855b4e0cb508211
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_15394c1893581eccba97743146b0b684(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_061d84aa7611e777b423e5c27e5be4b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_5cfe7c98f00412ac23d66936a68cbe3f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d712747b251001dbe21aad83385a403a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfe7c98f00412ac23d66936a68cbe3f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_e26cba9c4482fe8cdf616c435e98606b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bafdeaef0f02e2f9a0f87795f6133323(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e26cba9c4482fe8cdf616c435e98606b
        def get_inputs(self):
            return [
                paddle.uniform([2039], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_adfa3852b5eb81defb8fb4ad2a4c46bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b22f497b62eb6a4776eb9800508266e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adfa3852b5eb81defb8fb4ad2a4c46bf
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c6678c4db99fd8ed1290e7b7e6732d43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02b00304275d319ebb25ea44d8fc5aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6678c4db99fd8ed1290e7b7e6732d43
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2039, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_02b00304275d319ebb25ea44d8fc5aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6678c4db99fd8ed1290e7b7e6732d43
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2039, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_6f0d0dbf79d00d9a9a09819800fe05ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f610dc622191ffc76f3e1406e9a533e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f0d0dbf79d00d9a9a09819800fe05ad
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_fc33f30cc407dffd83ab84a064e04f11(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56343d925aa4023d6065dcab6744a61a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc33f30cc407dffd83ab84a064e04f11
        def get_inputs(self):
            return [
                paddle.uniform([4584], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_df0270c2519e82b2f1a9c3eec35ea3db(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4219c02e0f2c626b3a922dcd02fe8dc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df0270c2519e82b2f1a9c3eec35ea3db
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_1fa5c613d1bb1f9d99d0db6d4373c5cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c70376494178e73f6887e8e3752fe53d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fa5c613d1bb1f9d99d0db6d4373c5cf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4584, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c70376494178e73f6887e8e3752fe53d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fa5c613d1bb1f9d99d0db6d4373c5cf
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4584, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_cdd82cd8ac6b1805bd433edd1d1132e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dacffe4555f7d70560e28ae4843cd9e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdd82cd8ac6b1805bd433edd1d1132e7
        def get_inputs(self):
            return [
                paddle.uniform([6, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8393d4e85685585606eae265a2f95db0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0da2e96a3ed07d2cc05acfb9159a389b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8393d4e85685585606eae265a2f95db0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2733139097690582, 0.4986962676048279, 0.030321704223752022, 0.3326023817062378, 0.11917427182197571, 0.408772349357605], dtype='float32').reshape([6]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3a1e79a9453c0cceafe99b19acebd0f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2434], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77c44c95357378d0e31feb25a848991b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a1e79a9453c0cceafe99b19acebd0f4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_ace4b555c5f973ca9c7bb0554669fb07(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18979dd3a04383445a0d95a946905722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ace4b555c5f973ca9c7bb0554669fb07
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3ff340a9645ef5ec60f6ac59c3acf2f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0112d614a66babbbfebee3986abe1d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ff340a9645ef5ec60f6ac59c3acf2f6
        def get_inputs(self):
            return [
                paddle.uniform([1071], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_bff5c129cd4baea2708d64691bd49a6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5236bc4160de5c51b877bdc14f656ee7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bff5c129cd4baea2708d64691bd49a6b
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_dde333ebc8e08ffd0d5c41770dc516a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ae9948a6af7a8178103d4899c102c51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dde333ebc8e08ffd0d5c41770dc516a2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1071, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4ae9948a6af7a8178103d4899c102c51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dde333ebc8e08ffd0d5c41770dc516a2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1071, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d3dff6b36cc0376f8384b051cc68044a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff3ace37318603c4f030e72e2615f68d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3dff6b36cc0376f8384b051cc68044a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.07161086797714233, 0.04861365258693695, 0.19623209536075592, 0.01351057831197977], dtype='float32').reshape([4]),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_e65b6ca2206f5073d598d2ea129b3a40(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07501aae77b5d6897eff39babaeaedc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e65b6ca2206f5073d598d2ea129b3a40
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07161086797714233, 0.04861365258693695, 0.19623209536075592, 0.01351057831197977]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_142c3505febffdb5e34aa0bcba1504cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b6454099860b84b9ddca590046b5392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_142c3505febffdb5e34aa0bcba1504cc
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c3aa1e996ab3451bfa85c289877d979a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 768, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b32704c6e2546e728169a9f414d44314(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3aa1e996ab3451bfa85c289877d979a
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_893ff26cbd0f0405de64166f91828bac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37e359d2c29aa1eb29b3ad8623bb8f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_494b3dea2ade2da8e1b23419346b826d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cac9f9f1f2d37dec844370a5e372c0b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bae096cfea6474922f864b41f18f585c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68a307491a2890b2c4a68789333e22ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9bc25993b1e61cab1a35d84980850db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3dff6b36cc0376f8384b051cc68044a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3680858016014099, 0.10794095695018768, 0.2429090440273285, 0.47353971004486084], dtype='float32').reshape([4]),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_341433f9bf814b7f18c5230c9fdb3381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e65b6ca2206f5073d598d2ea129b3a40
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3680858016014099, 0.10794095695018768, 0.2429090440273285, 0.47353971004486084]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4cb49834d75f4efb5cc50f85b818bca4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_189c0c183a53f8c6012c7c0ac8e40425(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cb49834d75f4efb5cc50f85b818bca4
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9c77e1cb9cf14e7937bd02dbc9ab126a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8eea3714fa6e500f813ef242b8f095c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c77e1cb9cf14e7937bd02dbc9ab126a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4518aa279e2c8ae507a19bae4e2e0525(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0647870ea5f91f1c838a8e46ca220f5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4518aa279e2c8ae507a19bae4e2e0525
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bd3089df84cb3dc497e9b078e8448e98(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_786232221aef04e6932ce087900786ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3089df84cb3dc497e9b078e8448e98
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_94c5770e203e337c4fda6f891cbc1ca5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0, 1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15, 25], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_311b0377d8a6b3d394e5cfabc7107e91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c5770e203e337c4fda6f891cbc1ca5
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_311b0377d8a6b3d394e5cfabc7107e91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c5770e203e337c4fda6f891cbc1ca5
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_467219acc09f0c5a02c7f1b2f06b33e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 18, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8109192b37ca971fb01a7e75fc539ec2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_467219acc09f0c5a02c7f1b2f06b33e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8109192b37ca971fb01a7e75fc539ec2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_467219acc09f0c5a02c7f1b2f06b33e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_cdd64f54e534a2ad32238f5b7fb54b6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_174fc4ee228bb71757dacb3fa260b0e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdd64f54e534a2ad32238f5b7fb54b6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_43ae8aeb602f5bfca7cb6c0facf05f46(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8fb59bd17e7c61525c7639a6f2a781ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43ae8aeb602f5bfca7cb6c0facf05f46
        def get_inputs(self):
            return [
                paddle.uniform([2370], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_14f59880103b4ec7eee9683dcd9994bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6eacd2a0f3212fd9ee9a9fd90af0440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14f59880103b4ec7eee9683dcd9994bf
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_95a303976248ac32d9ce7e697e06fb4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d1a4cec9073a49f92bfd6dfe8375bb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95a303976248ac32d9ce7e697e06fb4c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2370, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2d1a4cec9073a49f92bfd6dfe8375bb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95a303976248ac32d9ce7e697e06fb4c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2370, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_877313f7163ed0c7f350b6c485e39124(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a99ecf800f4dc169bf9a878c4d11d15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_877313f7163ed0c7f350b6c485e39124
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8a99ecf800f4dc169bf9a878c4d11d15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_877313f7163ed0c7f350b6c485e39124
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_5c39a73ccfebe3f7c6c69375fd91bea4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7a10559e8edcbaf0219781bbe00b2a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c39a73ccfebe3f7c6c69375fd91bea4
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_8df620a83b37af9ead8341532dc104e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1bb866d027c49cbb7d2fab842c9666c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8df620a83b37af9ead8341532dc104e6
        def get_inputs(self):
            return [
                paddle.uniform([2993], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a3e71638ba6278ee35cd5c6c8d9e8976(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_453ffad23e944397150f24e946015d6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3e71638ba6278ee35cd5c6c8d9e8976
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_665208bb700285ba5e0ab06c04b0bd6a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e46260549f884b76194ce9b34bd83f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665208bb700285ba5e0ab06c04b0bd6a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2993, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e46260549f884b76194ce9b34bd83f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665208bb700285ba5e0ab06c04b0bd6a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2993, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9379d6adff113583350d0283ed3f9543(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3abb1540ab6adc0ac54f381905472d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9379d6adff113583350d0283ed3f9543
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4c1f7d59cb3cbea721d6327280563bbb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6adf4b467e767af05acc6b6b1d117ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c1f7d59cb3cbea721d6327280563bbb
        def get_inputs(self):
            return [
                paddle.uniform([3832], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_095818f021901a446338a44ee1006d7b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ebfd039ceb45d578bebdcd42f9b705d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_095818f021901a446338a44ee1006d7b
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_fcc215b63c4963fbde105e9f9d89731a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ae891adb1d5b2420ffbe58692ddb3e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcc215b63c4963fbde105e9f9d89731a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3832, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7ae891adb1d5b2420ffbe58692ddb3e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcc215b63c4963fbde105e9f9d89731a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3832, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_b47f0cb46c8be9cb0d74757f28dcf7e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8afab0c9e24060daf82ec9f26b8850f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47f0cb46c8be9cb0d74757f28dcf7e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8afab0c9e24060daf82ec9f26b8850f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b47f0cb46c8be9cb0d74757f28dcf7e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_351e0545724b46fdfc840dfe78ac75a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0cd9abc43ae5cc303f188235a0ad413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_351e0545724b46fdfc840dfe78ac75a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3fde256d658baf2dcb12335f5fd36ad6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47a050e4efc4de1af8620a2439c6e64f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fde256d658baf2dcb12335f5fd36ad6
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_47a050e4efc4de1af8620a2439c6e64f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fde256d658baf2dcb12335f5fd36ad6
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fce13e55a8cf39ad65bb4288015f8830(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_708ad435b1a8fc06d850ba4c092ac32e
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5274237ef0a914480809c3ac03ab7159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e377bb835402864478cd30cb0204e44
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9e2816e3cab0f36f02ecbbb5eb3267ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c12fb9ff1d1ac70184a7c0622416417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2816e3cab0f36f02ecbbb5eb3267ee
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_728d9e7257f8590a97280e78fb29dc3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcd4c9e63c4b78cafe67fc54987b5bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_728d9e7257f8590a97280e78fb29dc3d
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d1c63d855d70e23d56f81851db77d5ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1bc8d0e5aa3a1a1e8c2b94a65e9f337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1c63d855d70e23d56f81851db77d5ae
        def get_inputs(self):
            return [
                paddle.uniform([2, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bddcddd47b664daccaea561a696af999(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9529aac10ca383b079b459ff9935599a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bddcddd47b664daccaea561a696af999
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3661021292209625, 0.4057813286781311], dtype='float32').reshape([2]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3ec30ffcb7dbb72534c0fb2c0b0f03fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5dc89609f67624083e75e10c557ae86d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec30ffcb7dbb72534c0fb2c0b0f03fa
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fad61e157bf0fda05b216fe0ba064431(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bca17977c2684f6913a8eed7b902dae7
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_34ad5848b62e43c1d6d0067b37863ffb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b75d4591eac69327270ea402337c161e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34ad5848b62e43c1d6d0067b37863ffb
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_41b956ab14dbec59a9715121779a0fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8c060ab84becb8413f64f51eb107679
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_2eb19f3fbdc79931ff91b07198880059(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1966b953d652a263c1c30e3c201ff1aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2eb19f3fbdc79931ff91b07198880059
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fa1185d5b6acdf0632cdbd84aea31de8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2eb19f3fbdc79931ff91b07198880059
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_189abec98ab3d18eef16e7392673c36b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_779dfa7719a6be81fa7e5a497b02a623
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5274237ef0a914480809c3ac03ab7159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e377bb835402864478cd30cb0204e44
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d712747b251001dbe21aad83385a403a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfe7c98f00412ac23d66936a68cbe3f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_5d7f994241702e1b10b5f1524336e544(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f93eb2a240f6d731c6e48cc9ccf70bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d7f994241702e1b10b5f1524336e544
        def get_inputs(self):
            return [
                paddle.uniform([1995], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4b22f497b62eb6a4776eb9800508266e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adfa3852b5eb81defb8fb4ad2a4c46bf
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a48735301b5777c9efc23f9d1e695c1f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aef539ad577f1f99d3eeb72dba87e02c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a48735301b5777c9efc23f9d1e695c1f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_aef539ad577f1f99d3eeb72dba87e02c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a48735301b5777c9efc23f9d1e695c1f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dbf00884ccb7247204d663d477c76b8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9ba52da61159e6075964964fd5ef524
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_251006bb840241558ed401f48e9ba56f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b343977f234156d212017ce3cc39d1b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_68b5856d3a6c732c77d40a1ce303c40d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22263194280dd661d1fdcca32880f7dd
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_893ff26cbd0f0405de64166f91828bac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37e359d2c29aa1eb29b3ad8623bb8f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a9e23ea28f1114b9558725060473a7ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ce32e461d0abc4c44d9e917e04cd3d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9e23ea28f1114b9558725060473a7ab
        def get_inputs(self):
            return [
                paddle.uniform([150, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7ca19571e1af73bbb380d07735d36350(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a467b2aa56238fd6d151bf11813727e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ca19571e1af73bbb380d07735d36350
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_98276f971e8260a6b91933d98e0bfe9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f80599dfa78b3f2e7bf956093795842b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98276f971e8260a6b91933d98e0bfe9e
        def get_inputs(self):
            return [
                paddle.uniform([4181], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_dcc26493d2bbdce665bffaa887c72237(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b54eb2d3ead32b30c26b5ade97568d74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcc26493d2bbdce665bffaa887c72237
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_ce9fe610fee3e21cd3eccc02a4ae8ac6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 4], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_656aeafb56f34caf474ddc45ac1b7454(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce9fe610fee3e21cd3eccc02a4ae8ac6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4181, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_656aeafb56f34caf474ddc45ac1b7454(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce9fe610fee3e21cd3eccc02a4ae8ac6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4181, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_52b1b6086747b67bf982ba983e70074d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3dc3cec7e08eb3b6aa9371bf20155d81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52b1b6086747b67bf982ba983e70074d
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fad61e157bf0fda05b216fe0ba064431(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bca17977c2684f6913a8eed7b902dae7
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_304206a18eeae7e55091d68c06e89f94(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0, 1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c75e90e86d60a8d016d1bb2806ad1b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304206a18eeae7e55091d68c06e89f94
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6c75e90e86d60a8d016d1bb2806ad1b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304206a18eeae7e55091d68c06e89f94
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ac6e56436d7f0d516f1109db72fb932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8d9f8c8d68f746726d860570ed3f89c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a612b34786687b5607f0e2aca763d39a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_acd83dc4c2c970375228d2465488b64a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_70ac0c3d7db897e94e5b3c682bdabd06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_25e1ae6d1374ee30ce60abdf4a30246c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([4, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c0cc5755cf11d58f06c5f554f5b36893(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d860f1b0d5e8d943d0b4496f819b19ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0cc5755cf11d58f06c5f554f5b36893
        def get_inputs(self):
            return [
                paddle.to_tensor([0.26202839612960815, 0.3870741128921509, 0.4544948637485504, 0.17211998999118805], dtype='float32').reshape([4]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d2fe2e827ba8a256c31840872cf8bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7d37541940ebb3c5c8748dfc392a689e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77a5133bb6f7ce7359749d35317b498a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d37541940ebb3c5c8748dfc392a689e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9e492fa94724f0764643518ad1c96156(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18bb005af5252c83e51a56f5a75e80da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e492fa94724f0764643518ad1c96156
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_ba926a91d9f1584dc53a4d8bcaa3a034(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c06fac42538cffb0df2a4a78fc7602e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba926a91d9f1584dc53a4d8bcaa3a034
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dead47baa3d89cc2843cd3653585b6e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_affc6270166f52d6650b848f4184b51a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_231b813d9b9c7f65fd947200c58be4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_76de591e2b1d23446631a4f9c23a91e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_505b14d1307db0e4948a92c86712d244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76de591e2b1d23446631a4f9c23a91e6
        def get_inputs(self):
            return [
                paddle.uniform([21, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_18bb005af5252c83e51a56f5a75e80da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e492fa94724f0764643518ad1c96156
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c06fac42538cffb0df2a4a78fc7602e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba926a91d9f1584dc53a4d8bcaa3a034
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_231b813d9b9c7f65fd947200c58be4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_eaefeb2aafdffbf7cfe2962441dff5f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a697574fd2af885c1b6e8141e639631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaefeb2aafdffbf7cfe2962441dff5f5
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4bb8da2940232800583cab2386aeb278(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a610212803f4996e48c82a82138490f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bb8da2940232800583cab2386aeb278
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.48046234250068665, 0.22905594110488892], [0.2770739495754242, 0.3431546986103058], [0.24031347036361694, 0.48387277126312256], [0.4746449291706085, 0.4006763994693756], [0.10624674707651138, 0.05708790943026543], [0.49964481592178345, 0.2771390378475189]]], dtype='float32').reshape([1, 6, 2]),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_83b9f6ffc3671e6ddebf7c888f1e55b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4fa8106d0e525f071b0ff8200f615928(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aefac4b9589045f382c05c59f63a2ae1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1390a34b0c22cc3fc4d6c3a5fbfc87e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e56dc373aa5b778922fb47e96807a78d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7d76e4488ddd46f1fb1637cc47a2e130(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c5d9a553dafa30eb4f42951b3861cae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d76e4488ddd46f1fb1637cc47a2e130
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_29c3810673be9817217e9062a82370bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76de591e2b1d23446631a4f9c23a91e6
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3cfa49698725f7e399fb1279aa759607(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_771dd179626e25113f3f69d4e752ceef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cfa49698725f7e399fb1279aa759607
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1097528263926506, 0.4775157868862152, 0.11485323309898376, 0.08862581104040146]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([-2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6d27f0729c2b7b7939fb6f68f9fe2f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76de591e2b1d23446631a4f9c23a91e6
        def get_inputs(self):
            return [
                paddle.uniform([300, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c35917ff182319f2e7be5c341f7dd2af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c35917ff182319f2e7be5c341f7dd2af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_234a614b1dfc76325715003c39b2b8ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaefeb2aafdffbf7cfe2962441dff5f5
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_2f5cf8c7b56f83e264af9754524a6e1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ceb2d18adf2a2fa669d2b67f01261cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f5cf8c7b56f83e264af9754524a6e1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4c744301571ac54bbb6adc76b80bc58e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304206a18eeae7e55091d68c06e89f94
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4c744301571ac54bbb6adc76b80bc58e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304206a18eeae7e55091d68c06e89f94
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_85f04adc8d0164b55964a87e717c949e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d76e4488ddd46f1fb1637cc47a2e130
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_76d18530eca236c66639069272d2bb10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([3, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d1af0c2deb6ece364348c3a684adabee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0cc5755cf11d58f06c5f554f5b36893
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4758526086807251, 0.22843202948570251, 0.1572914719581604], dtype='float32').reshape([3]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5bab65e93b0890ec36c5aaaf77023d93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_429ef3c1bf29367647922238928f10da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_85747862aa94ea0a33250ebfe5604427(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89290026db95cdeb0968cb246b8176a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9e506ff39b2074fcae42c5172667140e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ddeaa01e5b5493bf784bb5834874b7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([1696], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f4cfd604b5fc6fc46c702ec1498575f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eaa587e6cac352665632911f8475590f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1696, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_eaa587e6cac352665632911f8475590f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1696, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cde5f28d28e241f0ce7c7ac95ac3df53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304206a18eeae7e55091d68c06e89f94
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cde5f28d28e241f0ce7c7ac95ac3df53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304206a18eeae7e55091d68c06e89f94
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b8ebbd8f8500e1e85f2c366b52f82db4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7ac91ed075c4ad51c170606a0a30dd81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c8824f41e8b1fe27b37dc001eb8bbf18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bb8da2940232800583cab2386aeb278
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9120fcbdde9316541055e8f7b89e1fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b20c37c3ce50a211bdf9ad35aad9a615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_df8cb6c07c56c0aa2568e732fe7ce4d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([5517], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_de3beaea901bbc10bdcc8f8115fec332(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_eb2a5eb60aa116eeac1c7bcaf66aab95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5517, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_eb2a5eb60aa116eeac1c7bcaf66aab95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5517, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ca16711d867f04ebe074b95bd4fd29a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ca16711d867f04ebe074b95bd4fd29a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_70ac0c3d7db897e94e5b3c682bdabd06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_13a67c64d5217aded4ac989150c3e2ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_84091c2602ddf2bdc5abeb5b56e3f95c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76de591e2b1d23446631a4f9c23a91e6
        def get_inputs(self):
            return [
                paddle.uniform([19, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5bf313f259f72007e543407c75cca0ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b4fa7bc36ad435d82de44ff7202687db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_89290026db95cdeb0968cb246b8176a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f7d411f18e9664953c98955a6e79b275(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([1794], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f4cfd604b5fc6fc46c702ec1498575f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b3fbbe1fac8a8c506016f986b180547f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1794, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b3fbbe1fac8a8c506016f986b180547f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1794, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b8ce3602e49815ff2bcf42164ca20599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9120a7b026cdad62aa511369a395cd6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7ac91ed075c4ad51c170606a0a30dd81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_35b48a04612d2ea85bb519c865f26d34(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b0fcf573cabb73626cc6774a9cba3a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35b48a04612d2ea85bb519c865f26d34
        def get_inputs(self):
            return [
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7d9311a7432999517072c87dc2c15f3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28d15d9335435b157a3805d3003cf168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d9311a7432999517072c87dc2c15f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_79c79d57be33ef078a331a3da16139ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b96fbb72a17cdfec4e71be77a56ec93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79c79d57be33ef078a331a3da16139ef
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7c0ba3095e2e5b634e67c029f7182d69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_28c3219f1250f1aeeb52324272ad1bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0d592355a8a1e4739e6793d0a1c946f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_522c0aebd2dc7252f70fb0efabce67bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6f3daa82476e3eabcc332834f3152b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d3d0f347c0cf27044aeb53815175dd10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([1504], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_49d2d02303db828095a94924931066d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_67f226a7ca75a5ff2abdc0dc61e98021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1504, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_67f226a7ca75a5ff2abdc0dc61e98021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1504, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_40545e7e614c6a70a485707d6e10d195(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d37541940ebb3c5c8748dfc392a689e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_94d32af4dfc43396e554200b69b1186c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d7e864c1ccbea54a2de20e23611ed796(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_eb23696cfc0949895895748f67d09ba8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_87ef686eb2bd16ba361aa0e96a89527f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_234a614b1dfc76325715003c39b2b8ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaefeb2aafdffbf7cfe2962441dff5f5
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ceb2d18adf2a2fa669d2b67f01261cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f5cf8c7b56f83e264af9754524a6e1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f56cd5b86372807392ff9567893eb227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f5cf8c7b56f83e264af9754524a6e1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_983e815d7b9d0dc9b7c3b67a058adfc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_94d32af4dfc43396e554200b69b1186c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_234a614b1dfc76325715003c39b2b8ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaefeb2aafdffbf7cfe2962441dff5f5
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ceb2d18adf2a2fa669d2b67f01261cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f5cf8c7b56f83e264af9754524a6e1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e2a4972b2d16553aaca34ee82c040d37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ec7fb6dd3346e54f58f8df01feabfdf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ec7fb6dd3346e54f58f8df01feabfdf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1f8a1b2a6b13cca3c6c7207f695f1fcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_efc65711d16706fcbda1a9af5fe92ed9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d37541940ebb3c5c8748dfc392a689e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b8ce3602e49815ff2bcf42164ca20599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7c0ba3095e2e5b634e67c029f7182d69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_df5e8f38d1925f4f0b3ea43ee38e0a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_249354b80c1ecdfe92b47d30e35e0a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([2039], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1e02ce2bc6e6da5042964822989ca37a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_36a90df68c9b45140884657852236ac0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2039, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_36a90df68c9b45140884657852236ac0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2039, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_02023593f023bd417b247e9a0129c3c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ea1c634d7e4291c707983d9720628107(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([4584], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cb64839cd6bb00c7ca1ebc7076acde42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ac15fc1b3c61563e68171e3e2cd8be45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4584, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ac15fc1b3c61563e68171e3e2cd8be45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4584, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e868d11562058285a0fd8f237f047f32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([6, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a4c7ed37067507ad3032ea5507c98c98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0cc5755cf11d58f06c5f554f5b36893
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2733139097690582, 0.4986962676048279, 0.030321704223752022, 0.3326023817062378, 0.11917427182197571, 0.408772349357605], dtype='float32').reshape([6]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_124ac434f43c2c3e3f20b4af6b34811d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_33091979ca10bafdacf61f2d2b4cf07f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_edecb20b5548a0d26751490a541a377f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([1071], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f3b7d9478c20a47892b980098bca515a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b0431424f6e786067f297aadc488af0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1071, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b0431424f6e786067f297aadc488af0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1071, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9e9b38da76da156a6cee0be5fc95ecb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35b48a04612d2ea85bb519c865f26d34
        def get_inputs(self):
            return [
                paddle.to_tensor([0.07161086797714233, 0.04861365258693695, 0.19623209536075592, 0.01351057831197977], dtype='float32').reshape([4]),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d8ee20efaaa86a3cd58896f005c17a81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df84b5a549233f409f3e0ed72ceccff2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8ee20efaaa86a3cd58896f005c17a81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07161086797714233, 0.04861365258693695, 0.19623209536075592, 0.01351057831197977]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6d493dc7e4c09ab4b187a64987bf3740(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cfa49698725f7e399fb1279aa759607
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_15c9df597a59503b477f7564ed52e2c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bb8da2940232800583cab2386aeb278
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_889a0e81173b55736835121b88f55c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_234a614b1dfc76325715003c39b2b8ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaefeb2aafdffbf7cfe2962441dff5f5
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ceb2d18adf2a2fa669d2b67f01261cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f5cf8c7b56f83e264af9754524a6e1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_aa2e6de7338ad9b0aeab589526f51251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35b48a04612d2ea85bb519c865f26d34
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3680858016014099, 0.10794095695018768, 0.2429090440273285, 0.47353971004486084], dtype='float32').reshape([4]),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4a6a7675c3e1cfb5c45c02f3a814f240(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8ee20efaaa86a3cd58896f005c17a81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3680858016014099, 0.10794095695018768, 0.2429090440273285, 0.47353971004486084]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c0ad89ec9fc29004c365e87731a6c286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cfa49698725f7e399fb1279aa759607
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_90b964bf393db77cb28d41af5873c58a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d1a9455e52ac82bec58d31eff7d5d175(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7421847a7ed51f702c23cde797fb9e52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8c750807ae0a4b33792e1e00e82a1ddd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304206a18eeae7e55091d68c06e89f94
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8c750807ae0a4b33792e1e00e82a1ddd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304206a18eeae7e55091d68c06e89f94
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cf219f6e7cc5ed8e32654bc880bfa8d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cf219f6e7cc5ed8e32654bc880bfa8d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_91d016e7ccf724a49d5b27c7f34576c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c07339de4600b74fade8188dd68aa7eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([2370], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f506162af737dba5851e1a82bf131561(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_313309ee885cd90b79fbcc8c39649dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2370, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_313309ee885cd90b79fbcc8c39649dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2370, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2abfb33a7079e2243f4251d78e0e1462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2abfb33a7079e2243f4251d78e0e1462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0db86ddb0c358117f6330f12d26a720e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8f7fe25b3c3b343cef6a07c34fa81c5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([2993], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_26d214415020bf774e8b03f6ed4e5947(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c5f034fca11f7f0d616a3627a8d0536d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2993, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c5f034fca11f7f0d616a3627a8d0536d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2993, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8cd43f71bf628bd0d0add982198877bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_af45fd0fad6ad5eeb8072fd513329edc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([3832], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ab7010e1079064c29027d91a15b26919(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5309a17bfb9c083108b5c8bf71365b33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3832, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5309a17bfb9c083108b5c8bf71365b33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3832, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_79b817a873ba1c16d6db51142df3ba78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_79b817a873ba1c16d6db51142df3ba78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_26ff80cfce87c3e9324ca5b98001526b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d1d3f866aa003429b93e5672c8bfb13f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d1d3f866aa003429b93e5672c8bfb13f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbc51898c623d48a6633cdbce8e29be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_429ef3c1bf29367647922238928f10da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_30e1a346c34e9b443eef7c6b1600c709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cce1d85683081318c8d3f4adbab8339d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_10e60522856297e226a29e06d0b568c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d7e63c88465ec334d11d1ef1fd5e7f1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([2, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ed9bd3180990b05a7848c100df337f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0cc5755cf11d58f06c5f554f5b36893
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3661021292209625, 0.4057813286781311], dtype='float32').reshape([2]),
                paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_75a94b93f7d28140d24afa101648a1a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_544495ae6d5fcfa0c0b6d62ef73e8385(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_174067c8f8412e6a6125d512322723c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_13a67c64d5217aded4ac989150c3e2ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_04fabb133e94682190af1c0a1eb09e89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0fa04f26e50f0da7752b8ee7978e7c01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fa8106d0e525f071b0ff8200f615928
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1f8a1b2a6b13cca3c6c7207f695f1fcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_30e1a346c34e9b443eef7c6b1600c709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_df5e8f38d1925f4f0b3ea43ee38e0a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8bdede47acbd4eba6d19ad99407eef26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([1995], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1e02ce2bc6e6da5042964822989ca37a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bb0a45391d64f79702c058c168dfc1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bb0a45391d64f79702c058c168dfc1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1995, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_18bb005af5252c83e51a56f5a75e80da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e492fa94724f0764643518ad1c96156
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c06fac42538cffb0df2a4a78fc7602e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba926a91d9f1584dc53a4d8bcaa3a034
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8d8fbc76d1cc4f03fd503e5b82431e8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_889a0e81173b55736835121b88f55c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6c4de1e51c9412f9d10052332e1d5ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76de591e2b1d23446631a4f9c23a91e6
        def get_inputs(self):
            return [
                paddle.uniform([150, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_71ae24b4390573e342fff45f57b14688(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85747862aa94ea0a33250ebfe5604427
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400], dtype='int32'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3eeaecce89e43f8412f57311134bfca8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e506ff39b2074fcae42c5172667140e
        def get_inputs(self):
            return [
                paddle.uniform([4181], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_eca2d9f23bae22a22188e802630b68cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5654b31089ff1fa6366f6e1e9becda26
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f97e86906e43ba64a818c3755fbe520b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4181, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f97e86906e43ba64a818c3755fbe520b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2ab75417892b38b34420e4c6306fd3c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4181, 4], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f0867e1f69b352202800d63d958e847f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24e6dbda9e91070a8f9bfe9f2652c17d
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_544495ae6d5fcfa0c0b6d62ef73e8385(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_101aaa8bf7c49df6e1469b23b085dc59
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()