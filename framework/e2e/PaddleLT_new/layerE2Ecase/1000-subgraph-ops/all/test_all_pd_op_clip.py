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
    class PrimitiveOp_543857bd7183b6604600e8a26931c7ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8de7d74f14d9d8179eddd7c12d276884(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_543857bd7183b6604600e8a26931c7ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_6a25c29009e4da322c33fe2b4e7de789(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1126d1e4621bdacbb2705aa58762afb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a25c29009e4da322c33fe2b4e7de789
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_29e30c9b5c51add39602437706c63456(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc9b105f3f50d5f6d77c0ba15570171c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29e30c9b5c51add39602437706c63456
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_65cf8ac4b84ced67bdd12b0ef59125f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44b962586990a1eb1d48afd60e9a8a48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65cf8ac4b84ced67bdd12b0ef59125f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63f43f71ad92c634a5701fcaf6dbaeb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2972797751426697]], [[0.225433811545372]], [[0.3709498941898346]], [[0.04370572045445442]], [[0.3836890459060669]], [[0.4882890582084656]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
                paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_410e1b6e3db17d89ff7c182ddb283648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_410e1b6e3db17d89ff7c182ddb283648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_410e1b6e3db17d89ff7c182ddb283648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_410e1b6e3db17d89ff7c182ddb283648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_eeefda8af6e801b4d34ff2e5527106d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57358cad4ca1f9eee6a0b10a06ec70e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eeefda8af6e801b4d34ff2e5527106d3
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bb3536f13b604eac348764ef39d0819e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bb3536f13b604eac348764ef39d0819e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bb3536f13b604eac348764ef39d0819e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bb3536f13b604eac348764ef39d0819e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_8746a36ea4872d1857d74196ffb3486e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9debd760f46a00000971cc7bd08645b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8746a36ea4872d1857d74196ffb3486e
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2eaa730f987c73c089ae637a686846af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2eaa730f987c73c089ae637a686846af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2eaa730f987c73c089ae637a686846af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2eaa730f987c73c089ae637a686846af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b76e23a46dd0c8c50917290cb09b67b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b76e23a46dd0c8c50917290cb09b67b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b76e23a46dd0c8c50917290cb09b67b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b76e23a46dd0c8c50917290cb09b67b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_0855d9517f596411ee96ce242752de7f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6aeba9c722f46ff5d2718d49a73c06b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0855d9517f596411ee96ce242752de7f
        def get_inputs(self):
            return [
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d9f0afda133d2f573b785e6aea7c332f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eeefda8af6e801b4d34ff2e5527106d3
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d9f0afda133d2f573b785e6aea7c332f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eeefda8af6e801b4d34ff2e5527106d3
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a354bbb0501760b7aba153219a6e32c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a354bbb0501760b7aba153219a6e32c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a354bbb0501760b7aba153219a6e32c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a354bbb0501760b7aba153219a6e32c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_916f6b738fc966d066e14ecd4fc28cbc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c1b6294ea016698965e0b748fe8c4cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_916f6b738fc966d066e14ecd4fc28cbc
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b56ea66ac904c8ef548f70845cac0dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b56ea66ac904c8ef548f70845cac0dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b56ea66ac904c8ef548f70845cac0dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b56ea66ac904c8ef548f70845cac0dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_410e1b6e3db17d89ff7c182ddb283648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_410e1b6e3db17d89ff7c182ddb283648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_410e1b6e3db17d89ff7c182ddb283648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_410e1b6e3db17d89ff7c182ddb283648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a619828f1bc3fb84a9ae84d3f27adc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a619828f1bc3fb84a9ae84d3f27adc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a619828f1bc3fb84a9ae84d3f27adc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a619828f1bc3fb84a9ae84d3f27adc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b89701f6fdb1c41899c7dad74ebf4a50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8746a36ea4872d1857d74196ffb3486e
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_565dc5930f2d7b63c368d32c2bfaf4fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 872], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ea7c815ace0a212c3bba631cfcfc850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_565dc5930f2d7b63c368d32c2bfaf4fd
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f9cc5d75917da0e6362fbef20645b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f9cc5d75917da0e6362fbef20645b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f9cc5d75917da0e6362fbef20645b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f9cc5d75917da0e6362fbef20645b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_933786cda0c47ced7699fcbd6a093166(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_926df81a8017505b4ebb94ed498575e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_926df81a8017505b4ebb94ed498575e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_416136500d1870c1520de6e5e2a3d6d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a53ea7967b223b5efae657e25cf1bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a53ea7967b223b5efae657e25cf1bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a53ea7967b223b5efae657e25cf1bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a53ea7967b223b5efae657e25cf1bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d856dd1d5a18b5967d4771b610b4f2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d856dd1d5a18b5967d4771b610b4f2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d856dd1d5a18b5967d4771b610b4f2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d856dd1d5a18b5967d4771b610b4f2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_25158fafbe89f462fa6b723c19d4905e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eeefda8af6e801b4d34ff2e5527106d3
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d2ba605c11a24f4f442c35fd88fb08a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d2ba605c11a24f4f442c35fd88fb08a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d2ba605c11a24f4f442c35fd88fb08a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d2ba605c11a24f4f442c35fd88fb08a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a72b38d18a5a5b3662c28740e84a153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a72b38d18a5a5b3662c28740e84a153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a72b38d18a5a5b3662c28740e84a153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a72b38d18a5a5b3662c28740e84a153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_aff20f532df99d3a72c1357525d57ce7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_911793fa8c9bf69ee6da8a71c82708f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.23468855023384094], [0.06051406264305115], [-0.3219420611858368], [-0.12678393721580505], [-0.3362758755683899], [-0.3670476973056793], [0.18165214359760284], [-0.16228656470775604], [-0.4335936903953552]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a3f05860c9e192b2a9d3ac2f22a1dedb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.09839127957820892], [-0.01789139211177826], [-0.09474319219589233], [-0.1871345341205597], [-0.11744290590286255], [-0.1367679387331009], [0.16015516221523285], [-0.03370022773742676], [-0.19704505801200867]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4f0663e74245e58ac9c4632f7202cab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_916f6b738fc966d066e14ecd4fc28cbc
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_69c0d901a61b3c6713357471d5983e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06528989225625992]], [[0.41746431589126587]], [[0.12354917079210281]], [[0.3680232763290405]], [[0.05339458957314491]], [[0.011256362311542034]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
                paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_60002401de2031d223fb8fc26b2e472f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_60002401de2031d223fb8fc26b2e472f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b0d6d425ba21d9c50e3bd586d72e3377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_57358cad4ca1f9eee6a0b10a06ec70e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eeefda8af6e801b4d34ff2e5527106d3
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_54dcf65c23b3909946fe79dfb6d13014(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c855def2af7a26be330da1be2462a0fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54dcf65c23b3909946fe79dfb6d13014
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8baec095f14448ac53dc9342709971ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65cf8ac4b84ced67bdd12b0ef59125f8
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d15ce339bf325913567542d66b91d0b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d15ce339bf325913567542d66b91d0b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ddddf2431358b2ee973e15d83c231dd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-2.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4cd93c7c7e2ea74585f4b264344125c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eeefda8af6e801b4d34ff2e5527106d3
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_1b3191181fd254c3d5901c1b63f226ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca29f04eb4e40ca2a697a8a74a9c563b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b3191181fd254c3d5901c1b63f226ee
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_96ef9cada2cb48db3c1aafbaf4474f61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_916f6b738fc966d066e14ecd4fc28cbc
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_25158fafbe89f462fa6b723c19d4905e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eeefda8af6e801b4d34ff2e5527106d3
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_818bb6a6c5b0950b87756ae6e72ddc86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8746a36ea4872d1857d74196ffb3486e
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b76e23a46dd0c8c50917290cb09b67b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b76e23a46dd0c8c50917290cb09b67b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b76e23a46dd0c8c50917290cb09b67b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b76e23a46dd0c8c50917290cb09b67b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d856dd1d5a18b5967d4771b610b4f2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d856dd1d5a18b5967d4771b610b4f2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d856dd1d5a18b5967d4771b610b4f2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d856dd1d5a18b5967d4771b610b4f2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0b634c47fb23d5d2ca35ecf464d8fbae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0b634c47fb23d5d2ca35ecf464d8fbae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e4f1708f6079dd289b8c6bf95f121a9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b56ea66ac904c8ef548f70845cac0dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b56ea66ac904c8ef548f70845cac0dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b56ea66ac904c8ef548f70845cac0dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b56ea66ac904c8ef548f70845cac0dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9efdd707b933ce4ccf069fb495e0ea09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_916f6b738fc966d066e14ecd4fc28cbc
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bb3536f13b604eac348764ef39d0819e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bb3536f13b604eac348764ef39d0819e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bb3536f13b604eac348764ef39d0819e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bb3536f13b604eac348764ef39d0819e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_feedff0529198edc77819ba26d67d4d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54dcf65c23b3909946fe79dfb6d13014
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33cb519d1b1c4caa1716223994571591(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.039678797125816345]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ef2856bfc75cae5102ee0b558ba14c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06958729028701782]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0fc811f2389541a88bb4317b32f3258a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.10821263492107391], [-0.0546928346157074], [-0.3427676260471344], [-0.15528357028961182], [-0.4410887658596039], [-0.2718930244445801]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b65c5bee1297b1461f332114f7ae27a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.3555653691291809], [-0.17415457963943481], [-0.27010005712509155], [-0.3420833945274353], [-0.057611167430877686], [-0.19337552785873413]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_835b3d3c5a9c2ada80f0d771d7b5fb5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8746a36ea4872d1857d74196ffb3486e
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_4537a3ffbcc184e704762911cb636912(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb207c5788d5ae7b1ec422ab89994381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4537a3ffbcc184e704762911cb636912
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4cd93c7c7e2ea74585f4b264344125c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eeefda8af6e801b4d34ff2e5527106d3
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a53ea7967b223b5efae657e25cf1bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a53ea7967b223b5efae657e25cf1bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a53ea7967b223b5efae657e25cf1bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a53ea7967b223b5efae657e25cf1bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2eaa730f987c73c089ae637a686846af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2eaa730f987c73c089ae637a686846af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2eaa730f987c73c089ae637a686846af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2eaa730f987c73c089ae637a686846af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8e24a82d7bc576d62143cc69dc605a4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8e24a82d7bc576d62143cc69dc605a4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a459aa51b3d89a02b754928562bf8dd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6cb925013fe3393389b1e7e961eaaf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6cb925013fe3393389b1e7e961eaaf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6cb925013fe3393389b1e7e961eaaf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6cb925013fe3393389b1e7e961eaaf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b25439551f7afaf830554e9ef445b31f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b25439551f7afaf830554e9ef445b31f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fae66b519e77da336a1bd8b4dadd560f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b2732fd1b88fc2430069d3210955d702(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0855d9517f596411ee96ce242752de7f
        def get_inputs(self):
            return [
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_95b8ab37e664765aa4b08a3f3ee536b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_95b8ab37e664765aa4b08a3f3ee536b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_364186effa0a845bc68646554ff899ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a354bbb0501760b7aba153219a6e32c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a354bbb0501760b7aba153219a6e32c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a354bbb0501760b7aba153219a6e32c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a354bbb0501760b7aba153219a6e32c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d2ba605c11a24f4f442c35fd88fb08a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d2ba605c11a24f4f442c35fd88fb08a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d2ba605c11a24f4f442c35fd88fb08a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d2ba605c11a24f4f442c35fd88fb08a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_064cdac146510cafc415cf490e51177c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2703114151954651], [-0.11170564591884613], [0.10895724594593048], [-0.24553707242012024], [-0.022169344127178192]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_02ece3f6d23bf8f2f258e6696eead6aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.38871267437934875], [0.03571050614118576], [-0.26595792174339294], [-0.2316424697637558], [-0.02788636088371277]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_4b53466a95f2655b9f4868784355a519(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1248], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4585e3153f49167789202dab42e01bb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b53466a95f2655b9f4868784355a519
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f9cc5d75917da0e6362fbef20645b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f9cc5d75917da0e6362fbef20645b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f9cc5d75917da0e6362fbef20645b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f9cc5d75917da0e6362fbef20645b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34279c5817495e697543d38ab9f05b5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65cf8ac4b84ced67bdd12b0ef59125f8
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a72b38d18a5a5b3662c28740e84a153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a72b38d18a5a5b3662c28740e84a153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a72b38d18a5a5b3662c28740e84a153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a72b38d18a5a5b3662c28740e84a153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64afe0993d4569e1e310584d97fe565d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64afe0993d4569e1e310584d97fe565d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64afe0993d4569e1e310584d97fe565d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64afe0993d4569e1e310584d97fe565d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0e1e499647f7aacfb8d225b6f03fc318(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54dcf65c23b3909946fe79dfb6d13014
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dd31ddc2e2d38a2b201155a7341fea89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dd31ddc2e2d38a2b201155a7341fea89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a2299e4312d65d72d1acd0d569b36ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d965732111d151e016fd4ae5f6406e96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d965732111d151e016fd4ae5f6406e96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6b7ac9020b355f9aabc49ac912ce7d08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5b1db631e0772a22ce307435d006621b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5b1db631e0772a22ce307435d006621b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b806653e1303e42fda1fa25312aa56a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_898ba26105564895a37086ecd3653f81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ef01e1f16b6a14408d57c3e7ca55310(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_898ba26105564895a37086ecd3653f81
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6cb925013fe3393389b1e7e961eaaf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6cb925013fe3393389b1e7e961eaaf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6cb925013fe3393389b1e7e961eaaf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6cb925013fe3393389b1e7e961eaaf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6ea7c815ace0a212c3bba631cfcfc850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_565dc5930f2d7b63c368d32c2bfaf4fd
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c8083fb817e06277ebfcd1e49a67b45a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65cf8ac4b84ced67bdd12b0ef59125f8
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b018ffe2f1914e3d63d3072e49a25573(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65cf8ac4b84ced67bdd12b0ef59125f8
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ad23e3f70a9ad060d152d325951baeaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54dcf65c23b3909946fe79dfb6d13014
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_0c890cb5d4bb8ce41b8711e83151fa8b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0f0b871c9a02eddf1172a286026fa5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c890cb5d4bb8ce41b8711e83151fa8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb207c5788d5ae7b1ec422ab89994381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4537a3ffbcc184e704762911cb636912
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_99450b399e5d5503e3c75321a6d67df9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.3109183609485626], [-0.16380369663238525], [0.07215642184019089], [-0.3890814781188965]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6a4e00b45739c5b853d3ba013569521a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07386109232902527], [-0.10055411607027054], [-0.031159162521362305], [-0.15769541263580322]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1523c6cf8768aa56e611038c6670934e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1523c6cf8768aa56e611038c6670934e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a459aa51b3d89a02b754928562bf8dd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64afe0993d4569e1e310584d97fe565d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64afe0993d4569e1e310584d97fe565d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64afe0993d4569e1e310584d97fe565d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64afe0993d4569e1e310584d97fe565d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a619828f1bc3fb84a9ae84d3f27adc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a619828f1bc3fb84a9ae84d3f27adc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a619828f1bc3fb84a9ae84d3f27adc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a619828f1bc3fb84a9ae84d3f27adc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c6650e368b36f2b5be9882add7073cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c6650e368b36f2b5be9882add7073cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c6650e368b36f2b5be9882add7073cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c6650e368b36f2b5be9882add7073cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0b5fe81c30cea827df227cfca5c815a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0b5fe81c30cea827df227cfca5c815a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e671ebebf42d0005af2fbc3a814e97af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9f18b8cf3ebdd60213989535f989b035(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 624], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1848fc79e340fefdc50cf07cffb8ddea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f18b8cf3ebdd60213989535f989b035
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c6650e368b36f2b5be9882add7073cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c6650e368b36f2b5be9882add7073cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c6650e368b36f2b5be9882add7073cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c6650e368b36f2b5be9882add7073cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d7d6bd88a4926fc1ed8ac61c7df6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_189635e505451f1f2864f6e8a7f5ca95(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0560f50d85985340adc13315d8c442ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_189635e505451f1f2864f6e8a7f5ca95
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c630dca3a2153832455e70c3535673f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d4b7f5fe446ef8738d3c93639d90b84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c630dca3a2153832455e70c3535673f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_6d983e4fd3bf6c459053045e16b8eff1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f841d82644ba61fcec5801961addf907(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d983e4fd3bf6c459053045e16b8eff1
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_7cacf0ed223e6cc54b889d3eda55dae6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a2b8d0cd8124f74de86e0e5c37e3b9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7cacf0ed223e6cc54b889d3eda55dae6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ec11d0ef27f194d7f6fccac2b0161bc5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f50ee295f0d4cd1300404dd6dbd758c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec11d0ef27f194d7f6fccac2b0161bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2972797751426697]], [[0.225433811545372]], [[0.3709498941898346]], [[0.04370572045445442]], [[0.3836890459060669]], [[0.4882890582084656]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
                paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_362c9d1619c744d5875ee5a2aab49a73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_79f8b4c4fd54cfcbb364e6052a8eee8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_362c9d1619c744d5875ee5a2aab49a73
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79f8b4c4fd54cfcbb364e6052a8eee8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_362c9d1619c744d5875ee5a2aab49a73
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79f8b4c4fd54cfcbb364e6052a8eee8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_362c9d1619c744d5875ee5a2aab49a73
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79f8b4c4fd54cfcbb364e6052a8eee8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_362c9d1619c744d5875ee5a2aab49a73
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_3dfabf392cb3ed3f907a95ac344c62ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_353f9c952ddac27b4947a591a5c0cfc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dfabf392cb3ed3f907a95ac344c62ec
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_642c07c4708d7ae71ae31cce55710eef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ba93568d84378877193abe6f0a6b2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_642c07c4708d7ae71ae31cce55710eef
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0ba93568d84378877193abe6f0a6b2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_642c07c4708d7ae71ae31cce55710eef
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0ba93568d84378877193abe6f0a6b2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_642c07c4708d7ae71ae31cce55710eef
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0ba93568d84378877193abe6f0a6b2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_642c07c4708d7ae71ae31cce55710eef
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_e721395ad417a8442bee7a1b8370b977(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0abb448fc7ad3435bcb3650769073439(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e721395ad417a8442bee7a1b8370b977
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_033ed8c2f41b957833c25a5d8a4a01e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_263503f673cdeae501e08c33ee4e5ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_033ed8c2f41b957833c25a5d8a4a01e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_263503f673cdeae501e08c33ee4e5ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_033ed8c2f41b957833c25a5d8a4a01e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_263503f673cdeae501e08c33ee4e5ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_033ed8c2f41b957833c25a5d8a4a01e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_263503f673cdeae501e08c33ee4e5ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_033ed8c2f41b957833c25a5d8a4a01e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_04d4a77665e0b2130daa0d004b9d4c3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33f4d804565cf49e73d042d729e33623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04d4a77665e0b2130daa0d004b9d4c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33f4d804565cf49e73d042d729e33623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04d4a77665e0b2130daa0d004b9d4c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33f4d804565cf49e73d042d729e33623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04d4a77665e0b2130daa0d004b9d4c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33f4d804565cf49e73d042d729e33623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04d4a77665e0b2130daa0d004b9d4c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b6aeba9c722f46ff5d2718d49a73c06b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0855d9517f596411ee96ce242752de7f
        def get_inputs(self):
            return [
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9f575ba0b9255603357d349c7167366f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc5a70e7c70f5c4d7754ecc3db4c940d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f575ba0b9255603357d349c7167366f
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cc5a70e7c70f5c4d7754ecc3db4c940d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f575ba0b9255603357d349c7167366f
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_4f180b237266917bbb34e0099a28fa81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb756b30411a910006f1f13da42f5954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f180b237266917bbb34e0099a28fa81
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb756b30411a910006f1f13da42f5954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f180b237266917bbb34e0099a28fa81
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb756b30411a910006f1f13da42f5954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f180b237266917bbb34e0099a28fa81
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb756b30411a910006f1f13da42f5954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f180b237266917bbb34e0099a28fa81
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ab9f476e382f8f94b6a3e1bde39d1b26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b81adadf6a418023d15bb10fc60b4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab9f476e382f8f94b6a3e1bde39d1b26
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c6f5268056f084e026eb8099721d5227(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8fa9078740b1519d77a50d9a8d4da95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6f5268056f084e026eb8099721d5227
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8fa9078740b1519d77a50d9a8d4da95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6f5268056f084e026eb8099721d5227
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8fa9078740b1519d77a50d9a8d4da95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6f5268056f084e026eb8099721d5227
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8fa9078740b1519d77a50d9a8d4da95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6f5268056f084e026eb8099721d5227
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79f8b4c4fd54cfcbb364e6052a8eee8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_362c9d1619c744d5875ee5a2aab49a73
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79f8b4c4fd54cfcbb364e6052a8eee8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_362c9d1619c744d5875ee5a2aab49a73
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79f8b4c4fd54cfcbb364e6052a8eee8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_362c9d1619c744d5875ee5a2aab49a73
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79f8b4c4fd54cfcbb364e6052a8eee8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_362c9d1619c744d5875ee5a2aab49a73
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_6bfe4798a52d11c5eb16af68dc819308(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64ecd357ed8ec282e79a9de5f49dd5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bfe4798a52d11c5eb16af68dc819308
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64ecd357ed8ec282e79a9de5f49dd5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bfe4798a52d11c5eb16af68dc819308
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64ecd357ed8ec282e79a9de5f49dd5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bfe4798a52d11c5eb16af68dc819308
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64ecd357ed8ec282e79a9de5f49dd5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bfe4798a52d11c5eb16af68dc819308
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_65013ccbc34672d7d819fef93a662ba2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae573703dec5ff5b47b419e30a18871f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65013ccbc34672d7d819fef93a662ba2
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9ced8a7708713dc21102de38af6e704c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c80d157db6953a4284a6637725d2605e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ced8a7708713dc21102de38af6e704c
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_2f34a1f410be5941a5f6626d7d7c7032(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_957baa1c35b7b1bff4cbc109ddf284b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f34a1f410be5941a5f6626d7d7c7032
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_957baa1c35b7b1bff4cbc109ddf284b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f34a1f410be5941a5f6626d7d7c7032
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_957baa1c35b7b1bff4cbc109ddf284b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f34a1f410be5941a5f6626d7d7c7032
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_957baa1c35b7b1bff4cbc109ddf284b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f34a1f410be5941a5f6626d7d7c7032
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_5c9694955b1efda00a132c66ef5db442(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d81ed20bc51b46cb0b0a3244e33c4af8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c9694955b1efda00a132c66ef5db442
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d81ed20bc51b46cb0b0a3244e33c4af8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c9694955b1efda00a132c66ef5db442
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_cd4ab8206b92a870b6c42d307251826a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_137f4e0ffa641663c57d5c4e788f17d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4ab8206b92a870b6c42d307251826a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_400adce09020233edfdfe39ea59f4904(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_854fda4f87437f41c08c63a261910698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_400adce09020233edfdfe39ea59f4904
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_854fda4f87437f41c08c63a261910698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_400adce09020233edfdfe39ea59f4904
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_854fda4f87437f41c08c63a261910698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_400adce09020233edfdfe39ea59f4904
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_854fda4f87437f41c08c63a261910698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_400adce09020233edfdfe39ea59f4904
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_8dc537a9b8ced42a60dd4350322aa8e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40b6ed42e8f18d84c5b02a45c6419b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dc537a9b8ced42a60dd4350322aa8e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_40b6ed42e8f18d84c5b02a45c6419b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dc537a9b8ced42a60dd4350322aa8e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_40b6ed42e8f18d84c5b02a45c6419b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dc537a9b8ced42a60dd4350322aa8e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_40b6ed42e8f18d84c5b02a45c6419b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dc537a9b8ced42a60dd4350322aa8e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_a0916aa3d8ff13363bd57a0552b15e71(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6dd5a424456b4033ba2a288e822fbce0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0916aa3d8ff13363bd57a0552b15e71
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_3bbf2f451211da9ff4511c756004e5ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc352eb25dd623124c509b7ae1b5f3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bbf2f451211da9ff4511c756004e5ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc352eb25dd623124c509b7ae1b5f3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bbf2f451211da9ff4511c756004e5ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc352eb25dd623124c509b7ae1b5f3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bbf2f451211da9ff4511c756004e5ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc352eb25dd623124c509b7ae1b5f3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bbf2f451211da9ff4511c756004e5ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_31cb3937cf1c7cab7694fbf6f871e5d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20719c23af13771ac516f668ebc7fa8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31cb3937cf1c7cab7694fbf6f871e5d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20719c23af13771ac516f668ebc7fa8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31cb3937cf1c7cab7694fbf6f871e5d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20719c23af13771ac516f668ebc7fa8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31cb3937cf1c7cab7694fbf6f871e5d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20719c23af13771ac516f668ebc7fa8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31cb3937cf1c7cab7694fbf6f871e5d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_bc70d1953fd54f720572c07dfe428482(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c88c169d4473d511fd4a735459fdc909(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc70d1953fd54f720572c07dfe428482
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.23468855023384094], [0.06051406264305115], [-0.3219420611858368], [-0.12678393721580505], [-0.3362758755683899], [-0.3670476973056793], [0.18165214359760284], [-0.16228656470775604], [-0.4335936903953552]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_604275befebd34fc46ec622e220a2c63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc70d1953fd54f720572c07dfe428482
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.09839127957820892], [-0.01789139211177826], [-0.09474319219589233], [-0.1871345341205597], [-0.11744290590286255], [-0.1367679387331009], [0.16015516221523285], [-0.03370022773742676], [-0.19704505801200867]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_dd4d0c83b1362134d77a94df83d40bde(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8e2bb9eedc9679cee32d5de6aaa116b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd4d0c83b1362134d77a94df83d40bde
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7fe18c32c9da313f7e288e28f98c5507(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec11d0ef27f194d7f6fccac2b0161bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06528989225625992]], [[0.41746431589126587]], [[0.12354917079210281]], [[0.3680232763290405]], [[0.05339458957314491]], [[0.011256362311542034]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
                paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_62b0d746c899f612d4366e164081c995(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_afcdce8f66ced9afae045edf660758df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62b0d746c899f612d4366e164081c995
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_afcdce8f66ced9afae045edf660758df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62b0d746c899f612d4366e164081c995
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_1bbd66c8db52cacf6b162fb4b7f7ba9c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48bf934a8b28ab0cc2bf0a920b5ba512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bbd66c8db52cacf6b162fb4b7f7ba9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_353f9c952ddac27b4947a591a5c0cfc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dfabf392cb3ed3f907a95ac344c62ec
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_00d04b17bbb76bd2ff90203bd7dda261(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cfa6e928d19074dba240f1a279d6a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00d04b17bbb76bd2ff90203bd7dda261
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_910c8119e08db3c04e1352bb5d554db8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88258b086544e5d23d74870e11dfc43f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_910c8119e08db3c04e1352bb5d554db8
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_4873668cd4b30757f28429aa7076ad05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7c6549a7a80a48328aad179900d2008(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4873668cd4b30757f28429aa7076ad05
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b7c6549a7a80a48328aad179900d2008(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4873668cd4b30757f28429aa7076ad05
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9d8b722913f17e85f402982b16207fc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4ab8206b92a870b6c42d307251826a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-2.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_b75216b406b900fc3a755523962c4ebd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e05433b5ad1b316df17e624789252f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b75216b406b900fc3a755523962c4ebd
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_fd30820c5e546091eb5a6f5875b36de2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 100, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a33a0963659d3b15985d957a75dc3061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd30820c5e546091eb5a6f5875b36de2
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_18323576432ffa2e134de4fc82958d7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4102ea14754e82970806ad9a011af371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18323576432ffa2e134de4fc82958d7e
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6dd5a424456b4033ba2a288e822fbce0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0916aa3d8ff13363bd57a0552b15e71
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_65c214e78a8b102c662daab0074adbdc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_35fdbde2ec52584222b66be8466050b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65c214e78a8b102c662daab0074adbdc
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33f4d804565cf49e73d042d729e33623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04d4a77665e0b2130daa0d004b9d4c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33f4d804565cf49e73d042d729e33623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04d4a77665e0b2130daa0d004b9d4c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33f4d804565cf49e73d042d729e33623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04d4a77665e0b2130daa0d004b9d4c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33f4d804565cf49e73d042d729e33623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04d4a77665e0b2130daa0d004b9d4c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_40b6ed42e8f18d84c5b02a45c6419b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dc537a9b8ced42a60dd4350322aa8e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_40b6ed42e8f18d84c5b02a45c6419b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dc537a9b8ced42a60dd4350322aa8e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_40b6ed42e8f18d84c5b02a45c6419b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dc537a9b8ced42a60dd4350322aa8e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_40b6ed42e8f18d84c5b02a45c6419b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dc537a9b8ced42a60dd4350322aa8e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_d684a19cece663556d681fda4a93de11(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22ae27c77b717fdb5ceef09e9ae1f901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d684a19cece663556d681fda4a93de11
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_22ae27c77b717fdb5ceef09e9ae1f901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d684a19cece663556d681fda4a93de11
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_3086da6b12b648607647d45c2deddfbe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9284a37fa7069653c8a2aa1c8a9fe0b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3086da6b12b648607647d45c2deddfbe
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8fa9078740b1519d77a50d9a8d4da95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6f5268056f084e026eb8099721d5227
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8fa9078740b1519d77a50d9a8d4da95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6f5268056f084e026eb8099721d5227
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8fa9078740b1519d77a50d9a8d4da95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6f5268056f084e026eb8099721d5227
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8fa9078740b1519d77a50d9a8d4da95f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6f5268056f084e026eb8099721d5227
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ebb39c54d1a47dfb1bd362d9e2a5b107(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ab067f2c00b0bd9885a29d0e50a228f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebb39c54d1a47dfb1bd362d9e2a5b107
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0ba93568d84378877193abe6f0a6b2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_642c07c4708d7ae71ae31cce55710eef
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0ba93568d84378877193abe6f0a6b2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_642c07c4708d7ae71ae31cce55710eef
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0ba93568d84378877193abe6f0a6b2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_642c07c4708d7ae71ae31cce55710eef
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0ba93568d84378877193abe6f0a6b2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_642c07c4708d7ae71ae31cce55710eef
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_628b7f9c2d0998a886ea54debb99af26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c47fb5004b77ba72f1ada705ed161b28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_628b7f9c2d0998a886ea54debb99af26
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_205a28d92f3d8bb1792ebd57e033b1c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb95a13cf7cdeef98fb39b1b3aaf62e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_205a28d92f3d8bb1792ebd57e033b1c7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.039678797125816345]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_203daaf5165692ff449d8c204ddfa114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_205a28d92f3d8bb1792ebd57e033b1c7
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06958729028701782]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_375005ede75b8e0c990f817702f21e7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5f620f0e3a7b2e406de211f8ce73f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_375005ede75b8e0c990f817702f21e7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.10821263492107391], [-0.0546928346157074], [-0.3427676260471344], [-0.15528357028961182], [-0.4410887658596039], [-0.2718930244445801]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_480362e119802e47cc364e20e16b758e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_375005ede75b8e0c990f817702f21e7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.3555653691291809], [-0.17415457963943481], [-0.27010005712509155], [-0.3420833945274353], [-0.057611167430877686], [-0.19337552785873413]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9e5a181064f9a1f75bb50eb3398a701c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68eff0c90ccef8e5761b86824983a272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e5a181064f9a1f75bb50eb3398a701c
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_8f6cde93a6026a82f651d55d586f5ce0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6a9973b537472183956a273f01531e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f6cde93a6026a82f651d55d586f5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4e05433b5ad1b316df17e624789252f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b75216b406b900fc3a755523962c4ebd
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_854fda4f87437f41c08c63a261910698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_400adce09020233edfdfe39ea59f4904
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_854fda4f87437f41c08c63a261910698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_400adce09020233edfdfe39ea59f4904
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_854fda4f87437f41c08c63a261910698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_400adce09020233edfdfe39ea59f4904
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_854fda4f87437f41c08c63a261910698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_400adce09020233edfdfe39ea59f4904
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_263503f673cdeae501e08c33ee4e5ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_033ed8c2f41b957833c25a5d8a4a01e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_263503f673cdeae501e08c33ee4e5ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_033ed8c2f41b957833c25a5d8a4a01e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_263503f673cdeae501e08c33ee4e5ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_033ed8c2f41b957833c25a5d8a4a01e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_263503f673cdeae501e08c33ee4e5ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_033ed8c2f41b957833c25a5d8a4a01e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_66881e286fd4249f7815eb652d43e432(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30809a1aa664ffa1e0f0866d8074089f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66881e286fd4249f7815eb652d43e432
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_30809a1aa664ffa1e0f0866d8074089f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66881e286fd4249f7815eb652d43e432
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c87e8696512fb49177d80fe5fd045047(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d6f7278015e05e4ce628ce1916505ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c87e8696512fb49177d80fe5fd045047
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ebec10c5675919cfa826272524d7a1b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f697e82d684e5cbed0bfce45dc14558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebec10c5675919cfa826272524d7a1b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8f697e82d684e5cbed0bfce45dc14558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebec10c5675919cfa826272524d7a1b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8f697e82d684e5cbed0bfce45dc14558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebec10c5675919cfa826272524d7a1b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8f697e82d684e5cbed0bfce45dc14558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebec10c5675919cfa826272524d7a1b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_31a8948413717b95edf3984baca9f3a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eab426a76dd168a7dafbfb5e112f083f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a8948413717b95edf3984baca9f3a1
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_eab426a76dd168a7dafbfb5e112f083f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a8948413717b95edf3984baca9f3a1
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ca3ab647cd942fab27e471df761ac682(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46720b2e209cb4c05044228507dd1dee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca3ab647cd942fab27e471df761ac682
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b2732fd1b88fc2430069d3210955d702(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0855d9517f596411ee96ce242752de7f
        def get_inputs(self):
            return [
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_564cf0d728956a4e6987b03d5063445e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_172ee6b7ccbb9d4011cc44d4c7ebbda1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_564cf0d728956a4e6987b03d5063445e
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_172ee6b7ccbb9d4011cc44d4c7ebbda1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_564cf0d728956a4e6987b03d5063445e
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_a27764e81e8978ec30cb8411e15a9a35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_479abf59fa4306cd09b6c18a69673b77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a27764e81e8978ec30cb8411e15a9a35
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb756b30411a910006f1f13da42f5954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f180b237266917bbb34e0099a28fa81
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb756b30411a910006f1f13da42f5954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f180b237266917bbb34e0099a28fa81
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb756b30411a910006f1f13da42f5954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f180b237266917bbb34e0099a28fa81
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb756b30411a910006f1f13da42f5954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f180b237266917bbb34e0099a28fa81
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc352eb25dd623124c509b7ae1b5f3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bbf2f451211da9ff4511c756004e5ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc352eb25dd623124c509b7ae1b5f3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bbf2f451211da9ff4511c756004e5ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc352eb25dd623124c509b7ae1b5f3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bbf2f451211da9ff4511c756004e5ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc352eb25dd623124c509b7ae1b5f3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bbf2f451211da9ff4511c756004e5ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_69917a69b2f2222fdad481b424ca7c19(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4dbe39f08112632be4b6426dad7bc2c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69917a69b2f2222fdad481b424ca7c19
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2703114151954651], [-0.11170564591884613], [0.10895724594593048], [-0.24553707242012024], [-0.022169344127178192]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_973d0f2eed104d5868b226d711995210(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69917a69b2f2222fdad481b424ca7c19
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.38871267437934875], [0.03571050614118576], [-0.26595792174339294], [-0.2316424697637558], [-0.02788636088371277]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_1fb44567998c58b85fd5206fcd4db32e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_13b30f2a006e1886fb0fc76352b1b447(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fb44567998c58b85fd5206fcd4db32e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_957baa1c35b7b1bff4cbc109ddf284b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f34a1f410be5941a5f6626d7d7c7032
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_957baa1c35b7b1bff4cbc109ddf284b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f34a1f410be5941a5f6626d7d7c7032
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_957baa1c35b7b1bff4cbc109ddf284b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f34a1f410be5941a5f6626d7d7c7032
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_957baa1c35b7b1bff4cbc109ddf284b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f34a1f410be5941a5f6626d7d7c7032
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_43b7819bc46456f3d3af503a2d448b3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34af27acbd6b035daca55e1ce7ddc3ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43b7819bc46456f3d3af503a2d448b3e
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20719c23af13771ac516f668ebc7fa8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31cb3937cf1c7cab7694fbf6f871e5d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20719c23af13771ac516f668ebc7fa8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31cb3937cf1c7cab7694fbf6f871e5d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20719c23af13771ac516f668ebc7fa8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31cb3937cf1c7cab7694fbf6f871e5d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20719c23af13771ac516f668ebc7fa8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31cb3937cf1c7cab7694fbf6f871e5d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_e1770956995dd0257bd06ee507e21716(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34ad8137c4e3b7b2499de2225a63ee39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1770956995dd0257bd06ee507e21716
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34ad8137c4e3b7b2499de2225a63ee39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1770956995dd0257bd06ee507e21716
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34ad8137c4e3b7b2499de2225a63ee39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1770956995dd0257bd06ee507e21716
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34ad8137c4e3b7b2499de2225a63ee39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1770956995dd0257bd06ee507e21716
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_b242ff47d8728b6957ba89ea18aa16a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_196ed6b6ac0e4d914585b69f47af70c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b242ff47d8728b6957ba89ea18aa16a7
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ba0fd000a29bd52b30414749d2448438(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_945f07e6133c28bcf902e04703b81dd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba0fd000a29bd52b30414749d2448438
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_945f07e6133c28bcf902e04703b81dd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba0fd000a29bd52b30414749d2448438
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_efc58ae23d862aa8082c16fc4de733f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c172955a3c007d245e7db71bc527d4c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_efc58ae23d862aa8082c16fc4de733f1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_32fc00a8a0b4ff23bd9f264fe51f6758(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4baf4e99eef6f3515b0b03144b9ba2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fc00a8a0b4ff23bd9f264fe51f6758
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d4baf4e99eef6f3515b0b03144b9ba2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fc00a8a0b4ff23bd9f264fe51f6758
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_b430a46546dbd565a0efc8accc4c6df6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3fdafefd70bbcfe9674005dc842843bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b430a46546dbd565a0efc8accc4c6df6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_d74a252eb5119e4509e37e0d180d0824(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c206c39bbfaeb9ce8d65c74db5733c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74a252eb5119e4509e37e0d180d0824
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c206c39bbfaeb9ce8d65c74db5733c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74a252eb5119e4509e37e0d180d0824
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_1fa64535baf35c29e489ce2a04cc1122(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7e9b7af0417e52b5d8961d086302aa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fa64535baf35c29e489ce2a04cc1122
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_448d44a265f1baa8d8847317a7bdbb6a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_383575d196034fcf13287126375dbe55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_448d44a265f1baa8d8847317a7bdbb6a
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8f697e82d684e5cbed0bfce45dc14558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebec10c5675919cfa826272524d7a1b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8f697e82d684e5cbed0bfce45dc14558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebec10c5675919cfa826272524d7a1b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8f697e82d684e5cbed0bfce45dc14558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebec10c5675919cfa826272524d7a1b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8f697e82d684e5cbed0bfce45dc14558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebec10c5675919cfa826272524d7a1b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c80d157db6953a4284a6637725d2605e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ced8a7708713dc21102de38af6e704c
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_41edb4f918afb94f5b77a56328d206ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f87dc057f23b1d1cc09d0cb18604d61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41edb4f918afb94f5b77a56328d206ef
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_1a9048c683f45df24adedbe210043ada(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f3bf18da10cd5829e15d48ad2afe724(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a9048c683f45df24adedbe210043ada
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_b3986e3d993c93b1757c1ac8aa483ac1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9cd62d83e358f1d7b674760e70990e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3986e3d993c93b1757c1ac8aa483ac1
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_36b5ca14bc74fce120d6270062b51596(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47ba3918f3bf5d36c5fe20113083d22b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36b5ca14bc74fce120d6270062b51596
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a6a9973b537472183956a273f01531e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f6cde93a6026a82f651d55d586f5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_921bdee860c15b9588904f1233ef87bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a09b026363ea82636082c16b1523a518(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_921bdee860c15b9588904f1233ef87bd
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.3109183609485626], [-0.16380369663238525], [0.07215642184019089], [-0.3890814781188965]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_63b934880e1e2c43eea2e5a9bcf91404(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_921bdee860c15b9588904f1233ef87bd
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07386109232902527], [-0.10055411607027054], [-0.031159162521362305], [-0.15769541263580322]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_fd84cf1e765a0967da6298974312a8b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a1e4c91f463e1a9a326f82ef23a7c02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd84cf1e765a0967da6298974312a8b5
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2a1e4c91f463e1a9a326f82ef23a7c02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd84cf1e765a0967da6298974312a8b5
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4d6f7278015e05e4ce628ce1916505ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c87e8696512fb49177d80fe5fd045047
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34ad8137c4e3b7b2499de2225a63ee39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1770956995dd0257bd06ee507e21716
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34ad8137c4e3b7b2499de2225a63ee39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1770956995dd0257bd06ee507e21716
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34ad8137c4e3b7b2499de2225a63ee39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1770956995dd0257bd06ee507e21716
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34ad8137c4e3b7b2499de2225a63ee39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1770956995dd0257bd06ee507e21716
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64ecd357ed8ec282e79a9de5f49dd5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bfe4798a52d11c5eb16af68dc819308
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64ecd357ed8ec282e79a9de5f49dd5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bfe4798a52d11c5eb16af68dc819308
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64ecd357ed8ec282e79a9de5f49dd5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bfe4798a52d11c5eb16af68dc819308
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_64ecd357ed8ec282e79a9de5f49dd5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bfe4798a52d11c5eb16af68dc819308
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_78675164e6a77452ba9b63b1b76b2f4f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7f3e953c923d3c475f2124a078bd2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78675164e6a77452ba9b63b1b76b2f4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7f3e953c923d3c475f2124a078bd2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78675164e6a77452ba9b63b1b76b2f4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7f3e953c923d3c475f2124a078bd2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78675164e6a77452ba9b63b1b76b2f4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7f3e953c923d3c475f2124a078bd2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78675164e6a77452ba9b63b1b76b2f4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ec36f4eaca7053a0e46923595eda09b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1700cc58f112f1fca10b2fe691ed8d61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec36f4eaca7053a0e46923595eda09b6
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1700cc58f112f1fca10b2fe691ed8d61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec36f4eaca7053a0e46923595eda09b6
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_56656f38b8d84af95236a290860c4c7f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88e248e3aeaedd89b4c4bf344d4e8586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56656f38b8d84af95236a290860c4c7f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_03074506d772015af78c55f92848cdff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9cfea47b754c34e16265333439b5de6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03074506d772015af78c55f92848cdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7f3e953c923d3c475f2124a078bd2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78675164e6a77452ba9b63b1b76b2f4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7f3e953c923d3c475f2124a078bd2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78675164e6a77452ba9b63b1b76b2f4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7f3e953c923d3c475f2124a078bd2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78675164e6a77452ba9b63b1b76b2f4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7f3e953c923d3c475f2124a078bd2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78675164e6a77452ba9b63b1b76b2f4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_fb6a90e105232210ba782b9b0feb8743(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8aa61e2380ca5692ad36b081352e854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_036ca3b8168539629af8702bdd5d0d10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33bb5fc6f30590c27e5ea86364f99f38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bd280ad7bc2ee4ce52107d21ad781ee9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d29ab87a292c901b99511d4d469396ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2972797751426697]], [[0.225433811545372]], [[0.3709498941898346]], [[0.04370572045445442]], [[0.3836890459060669]], [[0.4882890582084656]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
                paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c95d07104521e4ac5799ef435df5cf79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3f0eae01988436eb6d5ff89a3777c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e3f0eae01988436eb6d5ff89a3777c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e3f0eae01988436eb6d5ff89a3777c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e3f0eae01988436eb6d5ff89a3777c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c17400e933f9cb3e1c9419330659a813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44a2f7f85f6fddf345b9f204b57658e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44a2f7f85f6fddf345b9f204b57658e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44a2f7f85f6fddf345b9f204b57658e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44a2f7f85f6fddf345b9f204b57658e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9f05a2c098db7291054d288df25c35b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6d5b92b441ed4dad2b279dfe68162c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6d5b92b441ed4dad2b279dfe68162c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6d5b92b441ed4dad2b279dfe68162c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6d5b92b441ed4dad2b279dfe68162c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_93bf394ae21de6e8b35c567bef05338f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_93bf394ae21de6e8b35c567bef05338f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_93bf394ae21de6e8b35c567bef05338f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_93bf394ae21de6e8b35c567bef05338f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_71ebc1ff4edccf1a233a6e879c6a46a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d78007a506be08b12f7ce8d1380bfb4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ebc1ff4edccf1a233a6e879c6a46a7
        def get_inputs(self):
            return [
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5c90aa3d113784d21adbeff11fe596f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5c90aa3d113784d21adbeff11fe596f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62289c9c87356aa0b48d7df2acd76272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62289c9c87356aa0b48d7df2acd76272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62289c9c87356aa0b48d7df2acd76272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62289c9c87356aa0b48d7df2acd76272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5a89eea3c0427dbdb29e19b28c14adb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27df2d3d7cb5042665ac2ab932c3a8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27df2d3d7cb5042665ac2ab932c3a8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27df2d3d7cb5042665ac2ab932c3a8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27df2d3d7cb5042665ac2ab932c3a8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e3f0eae01988436eb6d5ff89a3777c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e3f0eae01988436eb6d5ff89a3777c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e3f0eae01988436eb6d5ff89a3777c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e3f0eae01988436eb6d5ff89a3777c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20095c532b879eda8c4422d807d5562f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20095c532b879eda8c4422d807d5562f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20095c532b879eda8c4422d807d5562f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20095c532b879eda8c4422d807d5562f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3bf70e18b90ff3d371ac42ab3bb9f79d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_97d70485f0070ef7e355588cfbb62831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_988f06e1170793fe765cb3a4474c6932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_988f06e1170793fe765cb3a4474c6932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_988f06e1170793fe765cb3a4474c6932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_988f06e1170793fe765cb3a4474c6932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_00eae0f17c8a227f162b80395f5a30c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_00eae0f17c8a227f162b80395f5a30c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b81807a4842da1820119c1f3715369b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_310ce4e1d1f0870d8479929273bb0e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_310ce4e1d1f0870d8479929273bb0e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_310ce4e1d1f0870d8479929273bb0e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_310ce4e1d1f0870d8479929273bb0e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67bfff9b1d3da33c35a1d97ff26e2b17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67bfff9b1d3da33c35a1d97ff26e2b17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67bfff9b1d3da33c35a1d97ff26e2b17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67bfff9b1d3da33c35a1d97ff26e2b17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6b29a8b1e05dd0adfa03c59f6acd6918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_77fe614118ca53586cdfa0e877e4126d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_77fe614118ca53586cdfa0e877e4126d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_77fe614118ca53586cdfa0e877e4126d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_77fe614118ca53586cdfa0e877e4126d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0f0fda8ab1d77605b428149bbb551948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0f0fda8ab1d77605b428149bbb551948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0f0fda8ab1d77605b428149bbb551948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0f0fda8ab1d77605b428149bbb551948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b0de489a61637e7e6fa8b66f33c7e09b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.23468855023384094], [0.06051406264305115], [-0.3219420611858368], [-0.12678393721580505], [-0.3362758755683899], [-0.3670476973056793], [0.18165214359760284], [-0.16228656470775604], [-0.4335936903953552]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bdab56d7db107c4559e61cc6741db13c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.09839127957820892], [-0.01789139211177826], [-0.09474319219589233], [-0.1871345341205597], [-0.11744290590286255], [-0.1367679387331009], [0.16015516221523285], [-0.03370022773742676], [-0.19704505801200867]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_91fa067ac7dbac03f1989289b1ff0430(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c5820c7d07856971141163eb935c67fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06528989225625992]], [[0.41746431589126587]], [[0.12354917079210281]], [[0.3680232763290405]], [[0.05339458957314491]], [[0.011256362311542034]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
                paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_70987171f31fa641018d4f7d1ea9a127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_70987171f31fa641018d4f7d1ea9a127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ae0d25ec20495f730ab480c70564b26a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c17400e933f9cb3e1c9419330659a813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d0cc45180cc09a46425b53b8dae44420(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_47e644239a2984b8ad4abf03feb3611a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d1c4b260e88798e3e9c8197aa68c1bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5d1c4b260e88798e3e9c8197aa68c1bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d9eff4aaad16f69f33731471fee42992(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-2.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_65aa59208e59d9851de1d623b658af99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_47131c91116d586ce24c183a6d27965d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf673ecca2d5b0bbbcf9363bcd4605f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47131c91116d586ce24c183a6d27965d
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_465959b000850e5f80770252ad8ad60a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6b29a8b1e05dd0adfa03c59f6acd6918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_952407e24c37ae210e16a86e17d0055d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_93bf394ae21de6e8b35c567bef05338f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_93bf394ae21de6e8b35c567bef05338f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_93bf394ae21de6e8b35c567bef05338f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_93bf394ae21de6e8b35c567bef05338f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67bfff9b1d3da33c35a1d97ff26e2b17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67bfff9b1d3da33c35a1d97ff26e2b17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67bfff9b1d3da33c35a1d97ff26e2b17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_67bfff9b1d3da33c35a1d97ff26e2b17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af0f1ca318f2746256c04f0bd788da47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af0f1ca318f2746256c04f0bd788da47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_60867811bbbeeaffbd1b6ed42993ec2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27df2d3d7cb5042665ac2ab932c3a8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27df2d3d7cb5042665ac2ab932c3a8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27df2d3d7cb5042665ac2ab932c3a8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_27df2d3d7cb5042665ac2ab932c3a8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_79349bb3a6b087e525eec1d4d2f10dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44a2f7f85f6fddf345b9f204b57658e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44a2f7f85f6fddf345b9f204b57658e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44a2f7f85f6fddf345b9f204b57658e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44a2f7f85f6fddf345b9f204b57658e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a275848200279b4153a3cd91e29de662(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_21391ac6d65fb778fe566af17fcfb0d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.039678797125816345]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a396c51043aad8596f3f7ead7c2c17cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06958729028701782]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1ed98ad6d0abd9d5f902012d3196d00c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.10821263492107391], [-0.0546928346157074], [-0.3427676260471344], [-0.15528357028961182], [-0.4410887658596039], [-0.2718930244445801]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b4c0959202ef361f86abdeee3abae86e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.3555653691291809], [-0.17415457963943481], [-0.27010005712509155], [-0.3420833945274353], [-0.057611167430877686], [-0.19337552785873413]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6d19bf3b09718ad7842e600c4462d4d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc09db02f643a9b3e2b5ce7dbda4ceef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_65aa59208e59d9851de1d623b658af99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_310ce4e1d1f0870d8479929273bb0e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_310ce4e1d1f0870d8479929273bb0e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_310ce4e1d1f0870d8479929273bb0e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_310ce4e1d1f0870d8479929273bb0e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6d5b92b441ed4dad2b279dfe68162c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6d5b92b441ed4dad2b279dfe68162c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6d5b92b441ed4dad2b279dfe68162c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6d5b92b441ed4dad2b279dfe68162c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62766efabd7268d26d6c8335131cfc07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62766efabd7268d26d6c8335131cfc07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2c8cb67bf5e76bec05e78470c90868d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_396e76d2840af3904741a7576457d31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_396e76d2840af3904741a7576457d31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_396e76d2840af3904741a7576457d31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_396e76d2840af3904741a7576457d31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c1a2547a26039338b427c058c97830bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c1a2547a26039338b427c058c97830bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_61995f17f207c03a6bd4afd78d1189d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44a1567fe136577af2c7d2fb225a7831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ebc1ff4edccf1a233a6e879c6a46a7
        def get_inputs(self):
            return [
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e53838623d4fabc1d1ce489855c24332(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e53838623d4fabc1d1ce489855c24332(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4e2fc965e193eee95b6f36b6704c176a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62289c9c87356aa0b48d7df2acd76272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62289c9c87356aa0b48d7df2acd76272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62289c9c87356aa0b48d7df2acd76272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62289c9c87356aa0b48d7df2acd76272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_77fe614118ca53586cdfa0e877e4126d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_77fe614118ca53586cdfa0e877e4126d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_77fe614118ca53586cdfa0e877e4126d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_77fe614118ca53586cdfa0e877e4126d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5a34a8f6b6e207634dd2e1b34c0ab2d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2703114151954651], [-0.11170564591884613], [0.10895724594593048], [-0.24553707242012024], [-0.022169344127178192]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6b6d852a48e4022335de7eab0eea4a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.38871267437934875], [0.03571050614118576], [-0.26595792174339294], [-0.2316424697637558], [-0.02788636088371277]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_48930789831f4dc79ea9d7e25c63061b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_988f06e1170793fe765cb3a4474c6932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_988f06e1170793fe765cb3a4474c6932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_988f06e1170793fe765cb3a4474c6932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_988f06e1170793fe765cb3a4474c6932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_11f1317b9bad22b9a2418765c1dd0207(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0f0fda8ab1d77605b428149bbb551948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0f0fda8ab1d77605b428149bbb551948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0f0fda8ab1d77605b428149bbb551948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0f0fda8ab1d77605b428149bbb551948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_814659dc2fb357c627ff6e9e645b7e4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_814659dc2fb357c627ff6e9e645b7e4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_814659dc2fb357c627ff6e9e645b7e4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_814659dc2fb357c627ff6e9e645b7e4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_06fa012df22e8a2a429a848a04da0ee6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6e9074e5cdf25fab08b7548fa19ee884(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6e9074e5cdf25fab08b7548fa19ee884(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ef817448115063b1859c4a54272e7da1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7d2704a291f1566e1c7d9711f2220544(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7d2704a291f1566e1c7d9711f2220544(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a58e692fc01642b3d7170a3556463900(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ebd355ccd0dedcb1b8fd97f9375662a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ebd355ccd0dedcb1b8fd97f9375662a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dea4b0416b0c0c8a3afa69feadbb3350(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_310a37c4d7b8aead296ab35c4e7114f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_396e76d2840af3904741a7576457d31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_396e76d2840af3904741a7576457d31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_396e76d2840af3904741a7576457d31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_396e76d2840af3904741a7576457d31b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_97d70485f0070ef7e355588cfbb62831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2194657acf66e56693c3a8e8ad4dce1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fe4233ef9c4cfcce6f426794d4ff2729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5544e6a0f2ea8b67008bd4d844307566(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6e2a1de27c3ed43b22c5cb0493a3938e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc09db02f643a9b3e2b5ce7dbda4ceef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dbd1628d5a12307f1c073bc08275a564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.3109183609485626], [-0.16380369663238525], [0.07215642184019089], [-0.3890814781188965]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b342976798e18e7fe5274041bb110af7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07386109232902527], [-0.10055411607027054], [-0.031159162521362305], [-0.15769541263580322]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fa417bea06f905c65bf511a97ad87571(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fa417bea06f905c65bf511a97ad87571(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2c8cb67bf5e76bec05e78470c90868d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_814659dc2fb357c627ff6e9e645b7e4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_814659dc2fb357c627ff6e9e645b7e4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_814659dc2fb357c627ff6e9e645b7e4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_814659dc2fb357c627ff6e9e645b7e4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20095c532b879eda8c4422d807d5562f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20095c532b879eda8c4422d807d5562f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20095c532b879eda8c4422d807d5562f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20095c532b879eda8c4422d807d5562f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8212411c8cd8f9563acd15f47bfb3b82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8212411c8cd8f9563acd15f47bfb3b82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8212411c8cd8f9563acd15f47bfb3b82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8212411c8cd8f9563acd15f47bfb3b82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_becb2af28f56e9655de21d91f134bd44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_becb2af28f56e9655de21d91f134bd44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ff01845e1100841d1e398886d910b7d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([15.989999771118164], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_13fe1112cf442d0e628e5e2c5fde0da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8212411c8cd8f9563acd15f47bfb3b82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8212411c8cd8f9563acd15f47bfb3b82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8212411c8cd8f9563acd15f47bfb3b82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8212411c8cd8f9563acd15f47bfb3b82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95d07104521e4ac5799ef435df5cf79
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()