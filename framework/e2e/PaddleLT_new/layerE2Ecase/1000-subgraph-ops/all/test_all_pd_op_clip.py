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


    class TestPrimitiveOp_5471f6884df2321627f580d82e4eb716(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.21430465579032898]], [[0.27133625745773315]], [[0.3204556703567505]], [[0.42410537600517273]], [[0.06479871273040771]], [[0.26669731736183167]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_d3b6994ac6f0d3f7d3192d4fbcf4c412(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d3b6994ac6f0d3f7d3192d4fbcf4c412(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d86d285a9877c484d06cd2beeaaac229(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.011502057313919067], [-0.28756415843963623], [-0.11075639724731445], [-0.33136725425720215], [0.09730316698551178], [-0.16637375950813293], [-0.27731984853744507], [-0.28065603971481323], [-0.2125084400177002]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_17e36f3734cce8225119434ebf165ad3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.12105238437652588], [-0.01972833275794983], [-0.03437850996851921], [0.137126624584198], [0.10457628965377808], [-0.2155960500240326], [-0.09550082683563232], [-0.3064834773540497], [-0.13248848915100098]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_198a02062b5c562c005cd215b37aecce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c6cff8e4aa7847e13f35a49655731ce
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06651467829942703]], [[0.02692185342311859]], [[0.13614360988140106]], [[0.13175329566001892]], [[0.35920295119285583]], [[0.41328200697898865]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
                paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33f3203e6bb7143164f8508ee804db6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_33f3203e6bb7143164f8508ee804db6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f168fe56b39e1b2d384a4d1eac14f19e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f168fe56b39e1b2d384a4d1eac14f19e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0556bd5f44c8093850fbeb76a07dd4f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0556bd5f44c8093850fbeb76a07dd4f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_9b5c2c4556da968c5c9731c4c4c9a92a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.12121646851301193]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_912b2c9cbb32cba7d9b0077b93f404b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03958737850189209]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d0a0a53e2309acadc7ab0aba69858b4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.34106171131134033], [-0.2862451374530792], [-0.1872786283493042], [-0.004125654697418213], [0.33751580119132996], [-0.4016847312450409]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f3e2f13d0af980cf415a691c73bb968c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.4495939314365387], [0.016538560390472412], [-0.06710958480834961], [-0.2517905831336975], [0.042601823806762695], [-0.3151167929172516]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_44885ad87ea37c6ec9e4dc0365994a46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44885ad87ea37c6ec9e4dc0365994a46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_1c9394084b3ed1d8108fca2d95dead84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1c9394084b3ed1d8108fca2d95dead84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4c9c21bfc6fb2ec91a0a5429bd2b4edd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4c9c21bfc6fb2ec91a0a5429bd2b4edd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_303039872ce6393cbd88847de762f087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.028618380427360535], [-0.15894195437431335], [-0.18241317570209503], [-0.12147417664527893], [-0.06606245785951614]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5f9e45feeb87d147920789b3cff3aafd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.1614772230386734], [0.049629244953393936], [-0.06883442401885986], [-0.30222707986831665], [-0.13241800665855408]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_65ef9ef2603f9a8ef2715ed3140aa3f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_65ef9ef2603f9a8ef2715ed3140aa3f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8c1897a347d8792968972a48aa4ed9b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8c1897a347d8792968972a48aa4ed9b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_ded1936d013bcc5a10023633a734f002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ded1936d013bcc5a10023633a734f002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4bee698d5108d9ebd7a5113c7aff0c5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2559038996696472], [0.08239400386810303], [-0.058451734483242035], [-0.37216609716415405]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a6f898bc613a5fbc2e243c8e932e67b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20f532df99d3a72c1357525d57ce7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.15236486494541168], [0.27904555201530457], [-0.17173093557357788], [-0.29265373945236206]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b10ad1a9a832f907aa70475f8b8164dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b10ad1a9a832f907aa70475f8b8164dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_edb273fb2bcb98e01e9bea7b9725946c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_edb273fb2bcb98e01e9bea7b9725946c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933786cda0c47ced7699fcbd6a093166
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_6f5a067d29a0b58f04938991cd471388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec11d0ef27f194d7f6fccac2b0161bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.21430465579032898]], [[0.27133625745773315]], [[0.3204556703567505]], [[0.42410537600517273]], [[0.06479871273040771]], [[0.26669731736183167]]], dtype='float32').reshape([6, 1, 1]),
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


    
    class PrimitiveOp_a28f7f6fb506c35b3bea6bdec2274e82(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e35c5595cb6a5d739ed6292b0813cbe5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28f7f6fb506c35b3bea6bdec2274e82
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e35c5595cb6a5d739ed6292b0813cbe5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28f7f6fb506c35b3bea6bdec2274e82
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c1a42a8d9b67ff2370532d87dc45c8af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc70d1953fd54f720572c07dfe428482
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.011502057313919067], [-0.28756415843963623], [-0.11075639724731445], [-0.33136725425720215], [0.09730316698551178], [-0.16637375950813293], [-0.27731984853744507], [-0.28065603971481323], [-0.2125084400177002]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4aca5dd9eb71235d215d2438ae69b166(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc70d1953fd54f720572c07dfe428482
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.12105238437652588], [-0.01972833275794983], [-0.03437850996851921], [0.137126624584198], [0.10457628965377808], [-0.2155960500240326], [-0.09550082683563232], [-0.3064834773540497], [-0.13248848915100098]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_4bc7653bd220607159448e2fbccdd5ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec11d0ef27f194d7f6fccac2b0161bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06651467829942703]], [[0.02692185342311859]], [[0.13614360988140106]], [[0.13175329566001892]], [[0.35920295119285583]], [[0.41328200697898865]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
                paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_16d84fb8cc98040256481678913c2252(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fa9f88c5173f7366a4bc2c567dc0cd6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16d84fb8cc98040256481678913c2252
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fa9f88c5173f7366a4bc2c567dc0cd6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16d84fb8cc98040256481678913c2252
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_de959682e1c214a399b2eafe82fbfac9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9626f502abfaffebcc2317282f635eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de959682e1c214a399b2eafe82fbfac9
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9626f502abfaffebcc2317282f635eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de959682e1c214a399b2eafe82fbfac9
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_c8a902f09560e0a99a3e3c999761c847(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f15206c9a77bb83380758d128c205da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8a902f09560e0a99a3e3c999761c847
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4f15206c9a77bb83380758d128c205da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8a902f09560e0a99a3e3c999761c847
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_21932de198b840a615f8877d6baeba92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_205a28d92f3d8bb1792ebd57e033b1c7
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.12121646851301193]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b96326e6dcde50cc841b1857d50f4d1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_205a28d92f3d8bb1792ebd57e033b1c7
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03958737850189209]], dtype='float32').reshape([1, 1]),
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


    class TestPrimitiveOp_53b0ad7d7d2b74f4a4149c5f1150101f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_375005ede75b8e0c990f817702f21e7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.34106171131134033], [-0.2862451374530792], [-0.1872786283493042], [-0.004125654697418213], [0.33751580119132996], [-0.4016847312450409]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6b394837c4a9c950175acec74eb91586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_375005ede75b8e0c990f817702f21e7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.4495939314365387], [0.016538560390472412], [-0.06710958480834961], [-0.2517905831336975], [0.042601823806762695], [-0.3151167929172516]], dtype='float32').reshape([6, 1]),
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


    
    class PrimitiveOp_75f117d5995bda1d61771b71aea592b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b9f783e3e395a6dd5b22fdff0163a8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75f117d5995bda1d61771b71aea592b3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4b9f783e3e395a6dd5b22fdff0163a8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75f117d5995bda1d61771b71aea592b3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_99a0c59a2c6f34fdac3e6f5fcd3dd997(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e66657b9b6f8de096969e71482893645(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99a0c59a2c6f34fdac3e6f5fcd3dd997
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e66657b9b6f8de096969e71482893645(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99a0c59a2c6f34fdac3e6f5fcd3dd997
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_9034d2b2737621c295422a61fb7f2147(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce59f20fd9a06fc278d229eed7c338c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9034d2b2737621c295422a61fb7f2147
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce59f20fd9a06fc278d229eed7c338c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9034d2b2737621c295422a61fb7f2147
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_54a05d555dc1d1b0d9fa3a60a8903b9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69917a69b2f2222fdad481b424ca7c19
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.028618380427360535], [-0.15894195437431335], [-0.18241317570209503], [-0.12147417664527893], [-0.06606245785951614]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_efdf794657e169b12f29e283d9ce0326(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69917a69b2f2222fdad481b424ca7c19
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.1614772230386734], [0.049629244953393936], [-0.06883442401885986], [-0.30222707986831665], [-0.13241800665855408]], dtype='float32').reshape([5, 1]),
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


    
    class PrimitiveOp_d3b00ad84cfacbe0c7bd57afcad8eeac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_753084b19f2bef73c179830cbb16451b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3b00ad84cfacbe0c7bd57afcad8eeac
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_753084b19f2bef73c179830cbb16451b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3b00ad84cfacbe0c7bd57afcad8eeac
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_503a2354048636791202d8fce407c33b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57a1ea8e735ed5fcdb2e0a20da5366e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_503a2354048636791202d8fce407c33b
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_57a1ea8e735ed5fcdb2e0a20da5366e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_503a2354048636791202d8fce407c33b
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_11195a8d6c3fb1995a9704558f4b156b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94de001909dbdbab04808e7b63363072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11195a8d6c3fb1995a9704558f4b156b
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_94de001909dbdbab04808e7b63363072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11195a8d6c3fb1995a9704558f4b156b
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_3e7e129ec09839e0b5004476b1d2e81b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_921bdee860c15b9588904f1233ef87bd
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2559038996696472], [0.08239400386810303], [-0.058451734483242035], [-0.37216609716415405]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2e4bd21a2eac6aaa66979bc37a38a3d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_921bdee860c15b9588904f1233ef87bd
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.15236486494541168], [0.27904555201530457], [-0.17173093557357788], [-0.29265373945236206]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_820867c5f6f0494238fb7497789253df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e803e5a38a107e3fc2e5a08d78d04d7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_820867c5f6f0494238fb7497789253df
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e803e5a38a107e3fc2e5a08d78d04d7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_820867c5f6f0494238fb7497789253df
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_b7122631e7dd13c84f8fe0ceb9df2230(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2286be702af705ee7db2ce1b27ce57a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7122631e7dd13c84f8fe0ceb9df2230
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2286be702af705ee7db2ce1b27ce57a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7122631e7dd13c84f8fe0ceb9df2230
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8438e24c8194116c4f21133c7d8c0e96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.21430465579032898]], [[0.27133625745773315]], [[0.3204556703567505]], [[0.42410537600517273]], [[0.06479871273040771]], [[0.26669731736183167]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_83593de8571fb3d5d40ffdc5fa8dfd42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_83593de8571fb3d5d40ffdc5fa8dfd42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0d3f8307ec22dddadde4c06b218d1e53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.011502057313919067], [-0.28756415843963623], [-0.11075639724731445], [-0.33136725425720215], [0.09730316698551178], [-0.16637375950813293], [-0.27731984853744507], [-0.28065603971481323], [-0.2125084400177002]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_abcc118198e69e9f9f5de2e9d685ae85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.12105238437652588], [-0.01972833275794983], [-0.03437850996851921], [0.137126624584198], [0.10457628965377808], [-0.2155960500240326], [-0.09550082683563232], [-0.3064834773540497], [-0.13248848915100098]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_f62b5c33771c142a9cf113d0e60f12fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bc494097d95b2cce45da9ff1cd601d7
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06651467829942703]], [[0.02692185342311859]], [[0.13614360988140106]], [[0.13175329566001892]], [[0.35920295119285583]], [[0.41328200697898865]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([-10000000000.0], dtype='float32').reshape([1]),
                paddle.to_tensor([4.135169982910156], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3ebe6966114b0560f43ee53035daf559(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3ebe6966114b0560f43ee53035daf559(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4dff24c68307f924c1ef6c7da1591928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4dff24c68307f924c1ef6c7da1591928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e5d9f8e13c3a8ad9de7b1b9ebd81e372(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e5d9f8e13c3a8ad9de7b1b9ebd81e372(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c7685e3df7c9f615e9efdb371a61192c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.12121646851301193]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_444eaa98e6fa5d98373cb2142e813f1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03958737850189209]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_52fbecada114b80ce2a60f6768bc445b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.34106171131134033], [-0.2862451374530792], [-0.1872786283493042], [-0.004125654697418213], [0.33751580119132996], [-0.4016847312450409]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dcd28fda6ae817b092204ac8f5014d2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.4495939314365387], [0.016538560390472412], [-0.06710958480834961], [-0.2517905831336975], [0.042601823806762695], [-0.3151167929172516]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_fca475090948cd1f624417506c622d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fca475090948cd1f624417506c622d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7e35f6385b6038a9f8fff93c2bf18cac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7e35f6385b6038a9f8fff93c2bf18cac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_020ccc5940fd3a4aa80eaff257008cc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_020ccc5940fd3a4aa80eaff257008cc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_066b87081b2f47a9286d37e4dd6b58bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.028618380427360535], [-0.15894195437431335], [-0.18241317570209503], [-0.12147417664527893], [-0.06606245785951614]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_395cbf599c876126a66e3e569d90eff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.1614772230386734], [0.049629244953393936], [-0.06883442401885986], [-0.30222707986831665], [-0.13241800665855408]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_58c3febd1b4692788648498b672f221e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_58c3febd1b4692788648498b672f221e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f96329b1aa07e0414f7bc93d171a1e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f96329b1aa07e0414f7bc93d171a1e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_10aebfa420e28c6b4d5f277487c53703(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_10aebfa420e28c6b4d5f277487c53703(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_85eb4002b1d614a6eeb8403208485f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2559038996696472], [0.08239400386810303], [-0.058451734483242035], [-0.37216609716415405]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_452dcf6e0fd1b468bc1280054adb4d69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.15236486494541168], [0.27904555201530457], [-0.17173093557357788], [-0.29265373945236206]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_29fbbc082f70d58729a4b22ba7cbe072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_29fbbc082f70d58729a4b22ba7cbe072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d8515033b52b5c168cd4af095f3e81f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([3.402820018375656e+38], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d8515033b52b5c168cd4af095f3e81f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb6a90e105232210ba782b9b0feb8743
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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