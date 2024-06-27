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
    class PrimitiveOp_b62ed7256944f66c36559fb75d33d959(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6de090b84e3c1ff632117dca16330f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cfe4530baba53aa45ded85e9515fb6b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a65e7ea27043e37481e1f708ad6a36f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_437523504476bb6f889e4599ced19784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a65e7ea27043e37481e1f708ad6a36f
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2852273db094a7119b9fd3d83cd0c314(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6365d45a592cd40665ab4846299a1ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40f562450e1aa492159f263d18988141(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de339f615ef61a0bcc42bc3aff25e671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c83329cd76083e1296073b5769a56d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a12ea67daa0cc5b5e490673d3e9034e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b511add50a3243443bf9c274aede8eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a12ea67daa0cc5b5e490673d3e9034e
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bac1c7ce46775fed3bb51a4b20d6dbd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73a399bd8410eba3b727819d9c5037fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8e64b96ceb7eba29b217842d16f32844(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6cc4560b6a10fce7d50d85ec0152e018(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e64b96ceb7eba29b217842d16f32844
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.095820665359497]], [[1.5141104459762573]], [[2.101196527481079]], [[2.188419818878174]], [[1.9872820377349854]], [[1.8394256830215454]], [[2.481407880783081]], [[2.1891562938690186]], [[1.9146358966827393]], [[1.3316240310668945]], [[2.3351967334747314]], [[1.9384446144104004]], [[1.7108885049819946]], [[2.8359644412994385]], [[2.057889223098755]], [[1.8670446872711182]], [[1.7827861309051514]], [[1.9267771244049072]], [[1.685118556022644]], [[2.0457208156585693]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_c95b3cea30a98c91228744b412e36897(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7beeea5a462cc0b87f4bbb52319f0c83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95b3cea30a98c91228744b412e36897
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_427f0a8b6d830555478927add3e9a006(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46cca03e4edefb5469994489cb80bc9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6cab69d11393bea70dbc5abf124f0f92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d6e5f8f6a4db59e509d9b7a50464b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cab69d11393bea70dbc5abf124f0f92
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a28cb470eca802733f0f400123ea4496(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49e6093d5cd72cb6695d95c5cb00fd30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5b09fa642e994cc3eca5d16a379f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_437523504476bb6f889e4599ced19784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a65e7ea27043e37481e1f708ad6a36f
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_687ccf3469de89ca29301dd35bd05ae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d6e5f8f6a4db59e509d9b7a50464b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cab69d11393bea70dbc5abf124f0f92
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_498d6274de6a46ee271670890d474644(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1152, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cad707f7505eddbb42d7d2933819cda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_498d6274de6a46ee271670890d474644
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3274df6f998666e89b7949358a6aa71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeecea24164e2eb6d84893bc67700a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5a367744042a90748d4fdf3df5857e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_890391dfab6e5a5cda3d558a1bb71a95(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f4b884d4e8bb5624694278356919209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_890391dfab6e5a5cda3d558a1bb71a95
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df0f67a8fecfb9654d5939b11278f358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_810e53d69c92b4b7658269ea1e80363f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10ea6bb83c5b73cf1b44edad78dacb15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_edb49d6e9febb7cebc9143b7edd1a924(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 15, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11287238138bc6a74a1f10e592ba7be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edb49d6e9febb7cebc9143b7edd1a924
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61c1404788a1c1fce4eef7140313cf04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85205f365883c7ff523844ae31929564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45fe4978feb9d75afecfb0b82eb213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f73bbcc884f549fc12b9563b6ec0a915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bbdcd03af4b98d7c620f8b7ae456f51f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9217d36fec24c4d933b09449d8619a40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbdcd03af4b98d7c620f8b7ae456f51f
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3bc8cdd1b620b3feab91694db6a29b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de339f615ef61a0bcc42bc3aff25e671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c164536ea3245eb106fd6122435a0dd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_437523504476bb6f889e4599ced19784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a65e7ea27043e37481e1f708ad6a36f
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70a25b856c0494fc56ec788be1ac01c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09716f7107eab049bbfe0bf6ce928221(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22025dcb78461db17043b6bc9b1529d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edb49d6e9febb7cebc9143b7edd1a924
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_687ccf3469de89ca29301dd35bd05ae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6053b1deb0ffbf555d57c487c9ca3267(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edb49d6e9febb7cebc9143b7edd1a924
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85205f365883c7ff523844ae31929564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64278eca55e69527bbda270a61b1105b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f6644134c7b60134389e6f8d678db5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_36b1601ddfc0c718c6440177e7f732a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74f2d8d2d9577c521ba5a58cda6786c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36b1601ddfc0c718c6440177e7f732a0
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fadf36bbfbbc7566c4e687bc1d71576f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a46d94339d8963d9a389fb24e9fe159(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a0b76fddb0842d5d5a620b98496b0a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a46d94339d8963d9a389fb24e9fe159
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_998a1b5f0c6072ea83226275c4ac7ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac4370fd63155604db8cad2468081856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fadf36bbfbbc7566c4e687bc1d71576f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3872141db037b7f858322a83d843d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c83329cd76083e1296073b5769a56d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_546fd49144316d5b40d967ddeafda4f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f73bbcc884f549fc12b9563b6ec0a915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79f94ee0faa6665041caef732943576f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1455b2a6b58e54f11a551541def71922(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c065300c53d293ea681751308f582a8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b43f412cbb322feaa0b729812581a97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cd84b15855d87c22d34d00cb0ebfbad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_99975f4791841f1995f0d8cd25fcdd77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9236c048530106e0d227aa272d6e6f3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99975f4791841f1995f0d8cd25fcdd77
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1da7173724fbb3e6442a5dd42e711adf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03374b769cb1971c801edb96c867a197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_834e3f765e823947350a275bd2b75ebe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d6e5f8f6a4db59e509d9b7a50464b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cab69d11393bea70dbc5abf124f0f92
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a66c939668115beb0b51c376ca412ce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5b09fa642e994cc3eca5d16a379f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c392189901a2fc55a315f3b45168737(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c665dada14c77dfdcf3cd3c9f1d24e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99975f4791841f1995f0d8cd25fcdd77
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a92ba801e6e07b46774a677182fcbd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a65e7ea27043e37481e1f708ad6a36f
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b76fddb0842d5d5a620b98496b0a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a46d94339d8963d9a389fb24e9fe159
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_687ccf3469de89ca29301dd35bd05ae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61c1404788a1c1fce4eef7140313cf04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0200d024cf693e703456e98fe9c3281f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_498d6274de6a46ee271670890d474644
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fece0e67a19b4b7edfd0a02cadd32b1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e427a2094323c3896353dfd1e4ab6e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fece0e67a19b4b7edfd0a02cadd32b1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5fe2179814000c9833c10aa520ac2a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1da7173724fbb3e6442a5dd42e711adf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6b39613c963f9b225fc928e240bd275(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a46d94339d8963d9a389fb24e9fe159
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11113a2386ac8ba1c088c204385a6124(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50e3989eccfba6abb1106b4ffa1d650e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e70f0047ff382770bde3fe410255440c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dfc8a4548b84346bcb6fc8f04db36085(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70f0047ff382770bde3fe410255440c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23546d3144135ce1ffce40c37861a475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_662a2f14c683759af991fa9a1c1eff5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de090b84e3c1ff632117dca16330f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a92ba801e6e07b46774a677182fcbd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a65e7ea27043e37481e1f708ad6a36f
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50e3989eccfba6abb1106b4ffa1d650e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c665dada14c77dfdcf3cd3c9f1d24e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99975f4791841f1995f0d8cd25fcdd77
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de090b84e3c1ff632117dca16330f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_418ad4d4b0d670be69750158811178b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb41e2a6f47712c37fae088c278495b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7861049391d58516b267d39c0689d43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cab69d11393bea70dbc5abf124f0f92
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50e3989eccfba6abb1106b4ffa1d650e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56e6c2455ff7e2e576a799163ff23746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ce2eaac6779604649dad587a2dc731a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b97b014f65db4c86725b34b1398ed54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ab2f23f3e05722dd70438a8cf7fedac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f03cd8eb30bcfdfc25ef159e5504de6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddaee5f6990559b7328dcfd011613518(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10ea6bb83c5b73cf1b44edad78dacb15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8ca637b61f0f75203c683ab70d43d7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fece0e67a19b4b7edfd0a02cadd32b1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9236c048530106e0d227aa272d6e6f3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99975f4791841f1995f0d8cd25fcdd77
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9eab1e6144318171d02d43663c7de730(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73945b436208495b9f83a4589f9a9cf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9eab1e6144318171d02d43663c7de730
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e90b5b1cc2063f075d05530c18d17db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edb49d6e9febb7cebc9143b7edd1a924
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb41e2a6f47712c37fae088c278495b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccd17fb8846d0c08fc01e9b8b42cc082(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85205f365883c7ff523844ae31929564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeecea24164e2eb6d84893bc67700a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_702546cfd6811b848adf01c594801858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5b09fa642e994cc3eca5d16a379f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45fe4978feb9d75afecfb0b82eb213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45fe4978feb9d75afecfb0b82eb213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5f691c0fe0e408a45ef1a874f215dbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac4370fd63155604db8cad2468081856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cf55a1c1d55bc7b25dc449fcf191b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c7011002a85def28098a2137c7829d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de090b84e3c1ff632117dca16330f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9236c048530106e0d227aa272d6e6f3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99975f4791841f1995f0d8cd25fcdd77
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f6644134c7b60134389e6f8d678db5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f4b884d4e8bb5624694278356919209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_890391dfab6e5a5cda3d558a1bb71a95
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0200d024cf693e703456e98fe9c3281f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_498d6274de6a46ee271670890d474644
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61c1404788a1c1fce4eef7140313cf04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c665dada14c77dfdcf3cd3c9f1d24e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99975f4791841f1995f0d8cd25fcdd77
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c9e77bb7e79f363cabf529b884eeade(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9e4d775cafef35bc730869e65ef6755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_361ebcfea011dc6240d5eff3a07a296c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a14c9344d6c7c9366694c96a7ed6273(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a2fc01c95434974e1187d86468277d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeecea24164e2eb6d84893bc67700a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f10761fb3b54c4af357954f3830f9cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_616b588b31a74622782cf4fb10c187ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a14c9344d6c7c9366694c96a7ed6273(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_437523504476bb6f889e4599ced19784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a65e7ea27043e37481e1f708ad6a36f
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d6e5f8f6a4db59e509d9b7a50464b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cab69d11393bea70dbc5abf124f0f92
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_13d9a358f9b0f5c75d2c561067525cde(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a76a8a9c7154eaabe336bfa7311fecd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13d9a358f9b0f5c75d2c561067525cde
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1ba1060e785b6f24b3b3e7aad1af5d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6e5e783971ad77780563ad78ed63f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_727576b7a125f3026b259ffd07334722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5af1882fde41b69010ad50b1779bb74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a76a8a9c7154eaabe336bfa7311fecd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13d9a358f9b0f5c75d2c561067525cde
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70ac88f891cfba655acbc7c15d8890a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11113a2386ac8ba1c088c204385a6124(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7861049391d58516b267d39c0689d43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cab69d11393bea70dbc5abf124f0f92
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de339f615ef61a0bcc42bc3aff25e671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a470e826c3547a8bfc41f1f5ba38443d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e6a1579f486de5f926b06cb5b6dc669d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a470e826c3547a8bfc41f1f5ba38443d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.813080072402954]], [[1.8140665292739868]], [[2.9743213653564453]], [[2.884744882583618]], [[2.6331980228424072]], [[1.6588718891143799]], [[1.8394668102264404]], [[2.028409957885742]], [[3.1729986667633057]], [[3.3984503746032715]], [[1.8893815279006958]], [[1.4538657665252686]], [[2.6711843013763428]], [[2.650304079055786]], [[2.484210968017578]], [[1.9773461818695068]], [[3.4528326988220215]], [[2.093771457672119]], [[1.9031968116760254]], [[1.5846881866455078]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_7a66c222ea178545c9719279bd5c6a04(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d99f54666f47278d525fe2efa591a3ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a66c222ea178545c9719279bd5c6a04
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_47fa4b8a07855cf86aacc9ff1c7fd76c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_385d06927a3c3e11dce66eb8a2d8b0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47fa4b8a07855cf86aacc9ff1c7fd76c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9f4a45eec7f78aef61e2297bd1bfc0b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c980cd0d41167c63afd11b33e7df930a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f4a45eec7f78aef61e2297bd1bfc0b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43ab29a2b9c6bdbb489b5330b8619f12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_396ee7caba7e3d406dd4b50e6c1f9df7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36b1601ddfc0c718c6440177e7f732a0
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8589b2e59b582654f560edde2899c957(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a92ba801e6e07b46774a677182fcbd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a65e7ea27043e37481e1f708ad6a36f
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10ea6bb83c5b73cf1b44edad78dacb15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2eaf2d4a8693345cbe1de2cf7fc622b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edb49d6e9febb7cebc9143b7edd1a924
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf45c85d89c14df63f071882f1b66a5e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a0bedcfcd888ed15cc0daf8c7e49139a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf45c85d89c14df63f071882f1b66a5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a48f5547787c08c11c214425c3b1b922(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4aefe029b60987329735abc358b2de94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a48f5547787c08c11c214425c3b1b922
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64278eca55e69527bbda270a61b1105b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73109bcb2bc851e9cb5f3da739499e2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cf428e9f649fde90ab0cb05c9ffea9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9cca61f1a0b32d57ef75d02a5d5c6618(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_273b0d4ea6226e7bd043cfda654e699a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cca61f1a0b32d57ef75d02a5d5c6618
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_998a1b5f0c6072ea83226275c4ac7ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45fe4978feb9d75afecfb0b82eb213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0ad463e9b9f1fa13da5944409a9485c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e64b96ceb7eba29b217842d16f32844
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.512618064880371]], [[2.1421754360198975]], [[2.6252074241638184]], [[2.2057788372039795]], [[2.862679958343506]], [[2.317392110824585]], [[2.359853744506836]], [[2.177365779876709]], [[1.6993900537490845]], [[1.720333456993103]], [[2.50675892829895]], [[2.556948661804199]], [[3.0345265865325928]], [[2.38387131690979]], [[1.7345192432403564]], [[1.7992361783981323]], [[1.979421615600586]], [[1.703852891921997]], [[1.2495094537734985]], [[2.8299062252044678]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_7beeea5a462cc0b87f4bbb52319f0c83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c95b3cea30a98c91228744b412e36897
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_337f8e532382d767836728528c669ffe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_471e5f68784153c9cadf0f2034b0ef22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_337f8e532382d767836728528c669ffe
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_396ee7caba7e3d406dd4b50e6c1f9df7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36b1601ddfc0c718c6440177e7f732a0
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49e6093d5cd72cb6695d95c5cb00fd30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a92ba801e6e07b46774a677182fcbd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a65e7ea27043e37481e1f708ad6a36f
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9236c048530106e0d227aa272d6e6f3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99975f4791841f1995f0d8cd25fcdd77
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11113a2386ac8ba1c088c204385a6124(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5b09fa642e994cc3eca5d16a379f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e082d6fd07c5c9145ab7c12c74e17c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d37d3541bed191ae23a73148a6347c3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_237f8baeb51c9363af27487d3569d9c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6b39613c963f9b225fc928e240bd275(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a46d94339d8963d9a389fb24e9fe159
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d534ed696556d5c12ad0b4e8bf6345f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_690ac17139d0779c9fbf11d7bc1d617b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42d417795aa8818d3565bd4fdbb75d4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_690ac17139d0779c9fbf11d7bc1d617b
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa1e96f3e4d65ab5ab87e65805bfcb95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_998a1b5f0c6072ea83226275c4ac7ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2460f7b91adb193db472023d6b6358c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85f5096e5bbd57211f06c3ac76455441(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cb470eca802733f0f400123ea4496
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_333c55d838ec80df780f045c3806297e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a921225834ed452e92cdbdccf105245a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_333c55d838ec80df780f045c3806297e
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5a367744042a90748d4fdf3df5857e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8ca637b61f0f75203c683ab70d43d7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fece0e67a19b4b7edfd0a02cadd32b1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74f2d8d2d9577c521ba5a58cda6786c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36b1601ddfc0c718c6440177e7f732a0
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_04891aef0c496353d16387902156ca9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c1e295b1fceeefe749d4d7538c35acd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04891aef0c496353d16387902156ca9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e427a2094323c3896353dfd1e4ab6e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fece0e67a19b4b7edfd0a02cadd32b1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_892817e00fefcbdaf83111fff3e1845e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0418cfabfd44735f6f5b0549585c3e49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_892817e00fefcbdaf83111fff3e1845e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7861049391d58516b267d39c0689d43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cab69d11393bea70dbc5abf124f0f92
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7861049391d58516b267d39c0689d43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cab69d11393bea70dbc5abf124f0f92
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea48907cff4abf212186add4dcc81a2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2316825700902d2158d8976fb5b87dae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea48907cff4abf212186add4dcc81a2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9d1fe783e160274f60862dc475259a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_237e7db3f9f1fc5fcee2742dad99b984(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9d1fe783e160274f60862dc475259a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba0262beb5b840fae7f580d2196aa271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64865bc785a75b3528aa4ed0c294de5d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f10761fb3b54c4af357954f3830f9cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c665dada14c77dfdcf3cd3c9f1d24e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99975f4791841f1995f0d8cd25fcdd77
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c83329cd76083e1296073b5769a56d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c8ec0731474ea2c1b2c44e06d315359(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a028b93a2b505be42336fbfeaabc8496(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_594e1427b2997d0b60f6f1baefbd3b10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a028b93a2b505be42336fbfeaabc8496
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cad707f7505eddbb42d7d2933819cda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_498d6274de6a46ee271670890d474644
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d24bd2eb9040722877185b91f6a65f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a48f5547787c08c11c214425c3b1b922
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2403129c039c1ba00a715a5defe3e80e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b83cba77c2b1f68db6b03e2a31d49669(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2403129c039c1ba00a715a5defe3e80e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9435344e2e10197099f88e2e7175a808(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 24, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d207b2e135299e13744a5fc27cea6af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9435344e2e10197099f88e2e7175a808
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_65bcf5f4139e308198fa6f6f443f1429(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 672, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa1f8772b4172c0ae4103748df519a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65bcf5f4139e308198fa6f6f443f1429
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2d54d995b4b6fc4ee90b3862e2a6cf5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 84, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e516cbd29d831ede5f0a7653d4675d9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2d54d995b4b6fc4ee90b3862e2a6cf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a02422f1260e653ca48bb6133234dc4d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_177d594b7537d55f7c054339b2bcc888(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a02422f1260e653ca48bb6133234dc4d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c903542a28346d517bca9b1c4af361e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06d219ff5b26989cc5fee365d75a0edc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c903542a28346d517bca9b1c4af361e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef61c702e62547f887ea9c7d76b7ef12(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1cc6b632841991f6945766f53b4c87f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef61c702e62547f887ea9c7d76b7ef12
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d6fd4e538450eca7fdaf97ee13da1151(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 48, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_645be0768017910a1fe08384b7882f4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6fd4e538450eca7fdaf97ee13da1151
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9130724dda50d4865237527c155eb8a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a2ad4aa308043c2e33d24c1e4f08fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9130724dda50d4865237527c155eb8a4
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2841319d27d2c16dd011e9b77784ee8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 15, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ddac817aef680c124454152bc41ec1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2841319d27d2c16dd011e9b77784ee8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6269a67ff3ba1945b44762365f7cf5c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 112, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77784efa867d657bf1ce0ef372b9d43f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6269a67ff3ba1945b44762365f7cf5c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_023f1b9fad6ecc556bd0fe190f3a01df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a470e826c3547a8bfc41f1f5ba38443d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.095820665359497]], [[1.5141104459762573]], [[2.101196527481079]], [[2.188419818878174]], [[1.9872820377349854]], [[1.8394256830215454]], [[2.481407880783081]], [[2.1891562938690186]], [[1.9146358966827393]], [[1.3316240310668945]], [[2.3351967334747314]], [[1.9384446144104004]], [[1.7108885049819946]], [[2.8359644412994385]], [[2.057889223098755]], [[1.8670446872711182]], [[1.7827861309051514]], [[1.9267771244049072]], [[1.685118556022644]], [[2.0457208156585693]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_d99f54666f47278d525fe2efa591a3ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a66c222ea178545c9719279bd5c6a04
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f945d2c0a2c85fdc94fa5f35a7abb8b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ead3be50dda4ba7cbb128603cd266de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f945d2c0a2c85fdc94fa5f35a7abb8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c17f499aae9519a98796fd4ca60dac75(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9d225eef062524c3beb58ce2104126a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c17f499aae9519a98796fd4ca60dac75
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3562e8d1ef611f6fa0489faa578f449(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fed80f7eb9c67ccf777f0472332bbad1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3562e8d1ef611f6fa0489faa578f449
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fdd00301d8e291eb21b16b7a17712e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cafbbc738f1eae9d751c7022403d345c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fdd00301d8e291eb21b16b7a17712e2
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3ec9d98503c2529c32b027c6b617281(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6181c13ff21621b7bd800b722f2af2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3ec9d98503c2529c32b027c6b617281
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa1f8772b4172c0ae4103748df519a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65bcf5f4139e308198fa6f6f443f1429
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_01451f74aa7967b30e92f5c4313ad590(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9fd6057e3e3e00faead7592d7ac98e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01451f74aa7967b30e92f5c4313ad590
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fed80f7eb9c67ccf777f0472332bbad1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3562e8d1ef611f6fa0489faa578f449
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0d3fc14cd84ce523990d807c764218ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1152, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84068586071c1d68593951baff30fe01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3fc14cd84ce523990d807c764218ff
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc0c26ef840692da73d89b452c178ec6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70dc6b46535b7bca2f98157959e8d1c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc0c26ef840692da73d89b452c178ec6
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9cbd40bc13e1228c3e6efec76c5ed10b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e6cc06831fd30851b649a4e8a16df7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cbd40bc13e1228c3e6efec76c5ed10b
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_77ebde5fb29b15ef8a2aa7c0216bed91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 30, 50], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e8cdb880390b0f9962a014cfd597c64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77ebde5fb29b15ef8a2aa7c0216bed91
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_582ccc7c890802f8374c46b3a9b48a09(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8305f2ce75eab199da9293ed756e5dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_582ccc7c890802f8374c46b3a9b48a09
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_42f1a8d14fa50a34d0fb0a159f46ca0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c1b0b3e0d3b4897e41f27b456df6151(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42f1a8d14fa50a34d0fb0a159f46ca0c
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9c39a250e3245b58c40770234297780f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 30, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4a6396907b183951dcf6938aa5faf71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c39a250e3245b58c40770234297780f
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_90cc1695a703b07253ec6c201cd287e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 52, 52], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e18419739456c1f6df5506302e17e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90cc1695a703b07253ec6c201cd287e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd7bb399c9fe16fb0fa0d2bb5f2f479f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 8, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_254ce9c7ee3803e1f8c4262853dea9a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd7bb399c9fe16fb0fa0d2bb5f2f479f
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3633266c7649e201b5c9d4453f4d5eff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03a5f6234931787d80e211749271cb90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3633266c7649e201b5c9d4453f4d5eff
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6751db0b1dd62008e8343a4722dea6c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 13, 13], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_161e436d21ec87748f9c39462e0a2fb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6751db0b1dd62008e8343a4722dea6c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea926f95b9f1af5906311eaddb82bb22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9841828ca50af753fe49d8ae33c86f14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea926f95b9f1af5906311eaddb82bb22
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c8815d5abaf190c8a1667676e8344b13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 34, 34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_482d30f01f509dd58ff0e6a6244b3f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8815d5abaf190c8a1667676e8344b13
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cb585cc48e0db8120094f90abe16bcb4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 768, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82af9043464f8440ba614c8ce0d31a68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb585cc48e0db8120094f90abe16bcb4
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb883bf7dcd5360111333e325038fba1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 76, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e99200ffc20feb78d8eb28e0ce2a0e47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb883bf7dcd5360111333e325038fba1
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1cc6b632841991f6945766f53b4c87f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef61c702e62547f887ea9c7d76b7ef12
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e3d187f72e959b127802c8f2cb97f878(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab46848ff3abce35e967174b4ae6120e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3d187f72e959b127802c8f2cb97f878
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa1f8772b4172c0ae4103748df519a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65bcf5f4139e308198fa6f6f443f1429
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eb58f9742732cbce16b732331df8a523(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e63ca49b26fe5c817e67651c90c5b167(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb58f9742732cbce16b732331df8a523
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_577e290ea10acab71a5068dc37724d99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f231847a7f7800f4ecb8847799253d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_577e290ea10acab71a5068dc37724d99
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34fb17cdfcd641cb81277645ef8c2d22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c7dc4019b788c1d0a996f9d5f0f7c183(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34fb17cdfcd641cb81277645ef8c2d22
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9fd6057e3e3e00faead7592d7ac98e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01451f74aa7967b30e92f5c4313ad590
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6181c13ff21621b7bd800b722f2af2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3ec9d98503c2529c32b027c6b617281
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_161e436d21ec87748f9c39462e0a2fb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6751db0b1dd62008e8343a4722dea6c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5731f8ee7f1e686df32fe0265743d6ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7f91a8359cf2f037635ebaab6cb78d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5731f8ee7f1e686df32fe0265743d6ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2a1c699dc39f03b4213a1d56157595aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 48, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_228b4b262b863b38e02a4d0c1af06586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a1c699dc39f03b4213a1d56157595aa
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45a5df12eeac8e0577cd0c425917ffcb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3a1d42107ab2a691a4ba0c51f2a2a88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45a5df12eeac8e0577cd0c425917ffcb
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_440a772317e6a0193321e4d1747328cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 60, 100], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e33bcf74b2d89684310051cb4f5c0970(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_440a772317e6a0193321e4d1747328cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ddac7a065b8eb182f4f608eb6e48153e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf6df7d69699043383ee9b2bb5d20416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ddac7a065b8eb182f4f608eb6e48153e
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b5d775c0a05c261e78ee01dd2f349896(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ca4ee58f84278ba9cd2ec0ca50f2e59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5d775c0a05c261e78ee01dd2f349896
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_51f071e53b6f5e87f5d88375e31a473a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5434436e343acd262069c64c60182d5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51f071e53b6f5e87f5d88375e31a473a
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e33bcf74b2d89684310051cb4f5c0970(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_440a772317e6a0193321e4d1747328cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2f8625c89b907c9c1c779512267c8e22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bfae15e9a6b9a5d7000f3cb510017241(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f8625c89b907c9c1c779512267c8e22
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_645be0768017910a1fe08384b7882f4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6fd4e538450eca7fdaf97ee13da1151
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_109434c43e6716307d860a0ebc1f276e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 21, 21], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1655289ee7fa04df9fdf2611941ee00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_109434c43e6716307d860a0ebc1f276e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_482d30f01f509dd58ff0e6a6244b3f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8815d5abaf190c8a1667676e8344b13
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d571753d1c1d32f7b7c708c628be7394(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 34, 34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64df7ae45368b0c75f9d9d08f3612880(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d571753d1c1d32f7b7c708c628be7394
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d32d7bcb7c238d06c7cf70a43dd973a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9235f5ce306b56b10e581f2905d722a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d32d7bcb7c238d06c7cf70a43dd973a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c60149f56365872b6026d18404a946cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_286f298fd67702e80f28c493d5f17c3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c60149f56365872b6026d18404a946cf
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c64cb16575de42f2a858ab4b1e68a49d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 18, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_445ddf8a48861377d4ed6889f57157ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64cb16575de42f2a858ab4b1e68a49d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_350ccafe0a227be9a8ebcde18e64df85(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 18, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aca3e50373353e2dc8f9d10da61a18c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_350ccafe0a227be9a8ebcde18e64df85
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_987c8b970dfe675e8dd06c0354e56292(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_600f79039cb74a9f3750265d62e4db30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_987c8b970dfe675e8dd06c0354e56292
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_79b5a9f707c611da3c7f3aff3c116ca6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 17, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc650f8733a055973b15c1812a2febec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79b5a9f707c611da3c7f3aff3c116ca6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3222fb5088cd82c884b7ca3e616e8805(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 17, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8fa8f13627a86efe60c676c5ca8f171c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3222fb5088cd82c884b7ca3e616e8805
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6988ae2a54acecee2c41ccb3e127e88e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebfd4b687465cc64124ebab9f9435cbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6988ae2a54acecee2c41ccb3e127e88e
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fed80f7eb9c67ccf777f0472332bbad1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3562e8d1ef611f6fa0489faa578f449
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1e825a511c499f9ebcb39fd92f8ca746(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0333f933267abc124827a5d718f2959(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e825a511c499f9ebcb39fd92f8ca746
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6181c13ff21621b7bd800b722f2af2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3ec9d98503c2529c32b027c6b617281
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d203d0a51dc1af2bf1a7c206acd1e4af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_335f5174ede5839827e9baffc1faf6e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d203d0a51dc1af2bf1a7c206acd1e4af
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0d0f3efd91a7b1c26606c74b8dcfeb06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7278bcb1900c29142f6bd7fd344b46dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d0f3efd91a7b1c26606c74b8dcfeb06
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_65ca584d57b70423d4da98283c444ac3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 672, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75ff1621065467b1c819155b5bcc264e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65ca584d57b70423d4da98283c444ac3
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf6df7d69699043383ee9b2bb5d20416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ddac7a065b8eb182f4f608eb6e48153e
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9fd6057e3e3e00faead7592d7ac98e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01451f74aa7967b30e92f5c4313ad590
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03a5f6234931787d80e211749271cb90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3633266c7649e201b5c9d4453f4d5eff
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1295e4c5f8fdb390b421480f2bcab5bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1152, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b2bc17972e2bd00b3e13107d19c5202(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1295e4c5f8fdb390b421480f2bcab5bf
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f9dd2f91919dd3363b2ee948493bf23(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 32, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e591b860f109c0dc40c5bb4d08c44eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f9dd2f91919dd3363b2ee948493bf23
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f86167b7bfba1fc7550aed1637a01861(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f550ff5c552dba5baccc597a5609d0b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f86167b7bfba1fc7550aed1637a01861
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc650f8733a055973b15c1812a2febec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79b5a9f707c611da3c7f3aff3c116ca6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d7685427bf34354e8123d41e5f0f6312(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1257569b4591bfa5d372c8f04ebcaf19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7685427bf34354e8123d41e5f0f6312
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a60543d1087a2f480b97b01aa04d5e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6aca4a2a59cb4f60664c6d8dbf19af9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a60543d1087a2f480b97b01aa04d5e0
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7aaa12450128d1cd4bce954f0ea1dc25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2ef6f3f9385b81333896ebe41af6f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aaa12450128d1cd4bce954f0ea1dc25
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfc8a4548b84346bcb6fc8f04db36085(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e70f0047ff382770bde3fe410255440c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_378433d00ef0a5445bc6cd9e4f4c4101(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 46, 46], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_022aac155761ab8c6268fb582ba09bb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_378433d00ef0a5445bc6cd9e4f4c4101
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3c58f05ac0dd3c7e5c000f7790119fd7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2204, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b98f6e117ef6e97accbe52079c61df3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c58f05ac0dd3c7e5c000f7790119fd7
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b83cba77c2b1f68db6b03e2a31d49669(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2403129c039c1ba00a715a5defe3e80e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ff1621065467b1c819155b5bcc264e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65ca584d57b70423d4da98283c444ac3
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ef6f3f9385b81333896ebe41af6f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aaa12450128d1cd4bce954f0ea1dc25
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7278bcb1900c29142f6bd7fd344b46dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d0f3efd91a7b1c26606c74b8dcfeb06
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b83cba77c2b1f68db6b03e2a31d49669(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2403129c039c1ba00a715a5defe3e80e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e76ce87086b3c0984ab471961b3c51ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[551, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a84dac2e7489fe99b338ad01a575f57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e76ce87086b3c0984ab471961b3c51ff
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_39913c484db272bbb0275b6c9a708230(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_496210ed8358e1483cc36b0643745842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39913c484db272bbb0275b6c9a708230
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f9f00e2539f2656ec00f1a730d8596ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_646f9f3a4a13f3edb1292956c576f45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9f00e2539f2656ec00f1a730d8596ab
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ef6f3f9385b81333896ebe41af6f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aaa12450128d1cd4bce954f0ea1dc25
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_756f842219561bc147d51be200ac7ad9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 60, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3c7cd9546cc8733798683063a4b610b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_756f842219561bc147d51be200ac7ad9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a577bbcc003a211464ae002a528c3df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4fc0b704bebb1a325e42ee464f3d0be0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a577bbcc003a211464ae002a528c3df
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa270719d7cb0ab83af611e7407b6c6a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06a5c75fe1e6436df492668a9d1b9abd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa270719d7cb0ab83af611e7407b6c6a
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0e171ee89c67be23bc9bdd261a51487(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e01cf6d2c17a3a18d6ea865ef9e6314d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0e171ee89c67be23bc9bdd261a51487
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_31f9f5efd89b5032a0b99c8fbc19a450(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_218cd4b147d5d3ed53f7a9bf66ac7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31f9f5efd89b5032a0b99c8fbc19a450
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_daa627c680b942c584fc704b24a132db(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b8cd651e1ab11449e58d825c82342b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa627c680b942c584fc704b24a132db
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e18419739456c1f6df5506302e17e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90cc1695a703b07253ec6c201cd287e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da6214305fa66aec1255d158cb873671(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 32, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_804264fd564f173c0bf75bb3805e92ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da6214305fa66aec1255d158cb873671
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600f79039cb74a9f3750265d62e4db30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_987c8b970dfe675e8dd06c0354e56292
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_79b54e49c771110590d08a15b9cff610(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e82598c20b07704e8668a154373651e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79b54e49c771110590d08a15b9cff610
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b83cba77c2b1f68db6b03e2a31d49669(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2403129c039c1ba00a715a5defe3e80e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_496210ed8358e1483cc36b0643745842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39913c484db272bbb0275b6c9a708230
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6cf4a9afe4f3a6c877d732cbab41cc22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01150fed4d562c8705728ff2be019116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cf4a9afe4f3a6c877d732cbab41cc22
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_161e436d21ec87748f9c39462e0a2fb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6751db0b1dd62008e8343a4722dea6c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e6cc06831fd30851b649a4e8a16df7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cbd40bc13e1228c3e6efec76c5ed10b
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57951228f15fe7641434dbcfb93297e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 23, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20a7b73f4914a233b1d7a8c1a7cc1285(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57951228f15fe7641434dbcfb93297e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6181c13ff21621b7bd800b722f2af2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3ec9d98503c2529c32b027c6b617281
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9841828ca50af753fe49d8ae33c86f14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea926f95b9f1af5906311eaddb82bb22
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9841828ca50af753fe49d8ae33c86f14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea926f95b9f1af5906311eaddb82bb22
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c81195e00e61bc934b9b98525af96a43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8816, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c710f22495255c159f772ba930ba671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c81195e00e61bc934b9b98525af96a43
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5434436e343acd262069c64c60182d5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51f071e53b6f5e87f5d88375e31a473a
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_26eef9b49983589ece2ca9193cd86555(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5dd7201c65816deef865c32dcd855c49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26eef9b49983589ece2ca9193cd86555
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ad6a6df922ea73a398a7988c4467133e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 19, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49a285c904f44482a7197380dda0c095(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6a6df922ea73a398a7988c4467133e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b83cba77c2b1f68db6b03e2a31d49669(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2403129c039c1ba00a715a5defe3e80e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600f79039cb74a9f3750265d62e4db30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_987c8b970dfe675e8dd06c0354e56292
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_228b4b262b863b38e02a4d0c1af06586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a1c699dc39f03b4213a1d56157595aa
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8305f2ce75eab199da9293ed756e5dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_582ccc7c890802f8374c46b3a9b48a09
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b2bc17972e2bd00b3e13107d19c5202(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1295e4c5f8fdb390b421480f2bcab5bf
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03a5f6234931787d80e211749271cb90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3633266c7649e201b5c9d4453f4d5eff
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7278bcb1900c29142f6bd7fd344b46dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d0f3efd91a7b1c26606c74b8dcfeb06
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d21e6f929bc67a612d9ac5df15e127a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc027dc03048b6291fd07ebc1060a4d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d21e6f929bc67a612d9ac5df15e127a
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34bc77b096a7495ba678fb6b26f19304(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2da2a50a839c914298d58a9aa8c057a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34bc77b096a7495ba678fb6b26f19304
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3e0e96f12728de7f10d00089e1055546(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 15, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5310662cc754d072773819f46ccce736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e0e96f12728de7f10d00089e1055546
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a0e80bd4a8985acdedda2440bff842ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 96, 144], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25feb07234baac28c04cb6db189c9a86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0e80bd4a8985acdedda2440bff842ef
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ee393d52cdede652fa1f015bede70011(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 192, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9678333860d030c5c1bed08d8c77084f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee393d52cdede652fa1f015bede70011
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e6cc06831fd30851b649a4e8a16df7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cbd40bc13e1228c3e6efec76c5ed10b
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b5d325f1c13b4f8f22c963e5e4c845ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 68, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_860d790826c93b462a392276720df336(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5d325f1c13b4f8f22c963e5e4c845ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a9b9792ae2d95601c7cbce9cb7bc233a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 68, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3d671ee575e093933666f346b90e25c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9b9792ae2d95601c7cbce9cb7bc233a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25feb07234baac28c04cb6db189c9a86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0e80bd4a8985acdedda2440bff842ef
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa1f8772b4172c0ae4103748df519a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65bcf5f4139e308198fa6f6f443f1429
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fed80f7eb9c67ccf777f0472332bbad1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3562e8d1ef611f6fa0489faa578f449
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_83d43c66b90a05d940bf1a51691fd84b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a153bd888fd3e4f2df2f95809e6aa505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83d43c66b90a05d940bf1a51691fd84b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb45a097ab18f8b8bf52ec1751aabb10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87f38cbb54f823eccc54991db633c301(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb45a097ab18f8b8bf52ec1751aabb10
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9c3ef148afc47e9414f5d3d9faa8ed41(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 72, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6e46e7f912f6faf2517deb7e98d5ed6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c3ef148afc47e9414f5d3d9faa8ed41
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4d3064660926a55f78bb4d2e04521b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 72, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6f38958280caaffbab0d783762da3179(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4d3064660926a55f78bb4d2e04521b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4d4c73c18e2fad7a93a51904fe91870(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 10, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c99c025fc8b25e70a1ba41709593414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4d4c73c18e2fad7a93a51904fe91870
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a153bd888fd3e4f2df2f95809e6aa505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83d43c66b90a05d940bf1a51691fd84b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c540af9e6af34c295e9c943f7896a242(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fce98513427aca7dc6897249f185b749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c540af9e6af34c295e9c943f7896a242
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6aca4a2a59cb4f60664c6d8dbf19af9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a60543d1087a2f480b97b01aa04d5e0
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_646f9f3a4a13f3edb1292956c576f45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9f00e2539f2656ec00f1a730d8596ab
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1cc6b632841991f6945766f53b4c87f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef61c702e62547f887ea9c7d76b7ef12
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6a1579f486de5f926b06cb5b6dc669d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a470e826c3547a8bfc41f1f5ba38443d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.813080072402954]], [[1.8140665292739868]], [[2.9743213653564453]], [[2.884744882583618]], [[2.6331980228424072]], [[1.6588718891143799]], [[1.8394668102264404]], [[2.028409957885742]], [[3.1729986667633057]], [[3.3984503746032715]], [[1.8893815279006958]], [[1.4538657665252686]], [[2.6711843013763428]], [[2.650304079055786]], [[2.484210968017578]], [[1.9773461818695068]], [[3.4528326988220215]], [[2.093771457672119]], [[1.9031968116760254]], [[1.5846881866455078]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_d99f54666f47278d525fe2efa591a3ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a66c222ea178545c9719279bd5c6a04
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_385d06927a3c3e11dce66eb8a2d8b0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47fa4b8a07855cf86aacc9ff1c7fd76c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c980cd0d41167c63afd11b33e7df930a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f4a45eec7f78aef61e2297bd1bfc0b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_712aeef42e3b82458a14ca36f64487f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 28, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b11c390b21d8a6f54e6e8a623ddd753a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_712aeef42e3b82458a14ca36f64487f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3991d0f429b99d1c654b2f0a3e848580(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_572c822cc3ffbc491da6d577dc20089e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3991d0f429b99d1c654b2f0a3e848580
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e31da081c2a643acdfdc1505eae9373(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 120, 200], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3cb3adcaa5b704b134d747a2873335d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e31da081c2a643acdfdc1505eae9373
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ff1621065467b1c819155b5bcc264e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65ca584d57b70423d4da98283c444ac3
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e18419739456c1f6df5506302e17e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90cc1695a703b07253ec6c201cd287e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9841828ca50af753fe49d8ae33c86f14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea926f95b9f1af5906311eaddb82bb22
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0bedcfcd888ed15cc0daf8c7e49139a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf45c85d89c14df63f071882f1b66a5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_847812de5fe31c0b1e03088358a51c61(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 120, 200], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11a73a9080d3960694972e6f7fde71c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_847812de5fe31c0b1e03088358a51c61
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7f91a8359cf2f037635ebaab6cb78d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5731f8ee7f1e686df32fe0265743d6ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dbdb953e5455a64d8871673b4c7eadab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ae7cad588c1fb82592ab32c83d756ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbdb953e5455a64d8871673b4c7eadab
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_faf647cfe6a1f341b42fc82009d71a5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 92, 92], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3027b53d2ffdc665112ed5d3fce4b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_faf647cfe6a1f341b42fc82009d71a5f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5081aca8df0fdab5d576a7255a05c71e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d187d6f2658f79334a8accf437884f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5081aca8df0fdab5d576a7255a05c71e
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ca4ee58f84278ba9cd2ec0ca50f2e59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5d775c0a05c261e78ee01dd2f349896
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9841828ca50af753fe49d8ae33c86f14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea926f95b9f1af5906311eaddb82bb22
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e881b1c21005b5f369328f721f293ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a470e826c3547a8bfc41f1f5ba38443d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.512618064880371]], [[2.1421754360198975]], [[2.6252074241638184]], [[2.2057788372039795]], [[2.862679958343506]], [[2.317392110824585]], [[2.359853744506836]], [[2.177365779876709]], [[1.6993900537490845]], [[1.720333456993103]], [[2.50675892829895]], [[2.556948661804199]], [[3.0345265865325928]], [[2.38387131690979]], [[1.7345192432403564]], [[1.7992361783981323]], [[1.979421615600586]], [[1.703852891921997]], [[1.2495094537734985]], [[2.8299062252044678]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_d99f54666f47278d525fe2efa591a3ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a66c222ea178545c9719279bd5c6a04
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_385d06927a3c3e11dce66eb8a2d8b0c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47fa4b8a07855cf86aacc9ff1c7fd76c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_572c822cc3ffbc491da6d577dc20089e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3991d0f429b99d1c654b2f0a3e848580
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cafbbc738f1eae9d751c7022403d345c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fdd00301d8e291eb21b16b7a17712e2
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ff1621065467b1c819155b5bcc264e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65ca584d57b70423d4da98283c444ac3
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600f79039cb74a9f3750265d62e4db30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_987c8b970dfe675e8dd06c0354e56292
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6aca4a2a59cb4f60664c6d8dbf19af9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a60543d1087a2f480b97b01aa04d5e0
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6181c13ff21621b7bd800b722f2af2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3ec9d98503c2529c32b027c6b617281
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_83a6dc6a1a8f89748d96d0489e2e30e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99633345d17a1990d47005e778d101ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83a6dc6a1a8f89748d96d0489e2e30e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a2dbb8a57a01fcf486ed3cf8ba744881(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_504331282df5184a566e88d687edc644(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2dbb8a57a01fcf486ed3cf8ba744881
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a239d8cc3255e0c2db08847baa6efa4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 38, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_992b3373d804af28fbfacb14c2221494(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a239d8cc3255e0c2db08847baa6efa4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1257569b4591bfa5d372c8f04ebcaf19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7685427bf34354e8123d41e5f0f6312
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0e8e9b0038e16b474f04903c38b01b92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 56, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bfe867fc9ba8b3e35c7edfc3e3c592ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e8e9b0038e16b474f04903c38b01b92
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df1ec3f738805a93e82cbdbf963f0e61(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 576, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28f70772b7d5e0ec5c72a2a01bc619bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1ec3f738805a93e82cbdbf963f0e61
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_40673819ead0db3007ed798f80e2311c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 42, 42], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3a77319c80b5eef594fd98000dc22f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40673819ead0db3007ed798f80e2311c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ca4ee58f84278ba9cd2ec0ca50f2e59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5d775c0a05c261e78ee01dd2f349896
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_036b9117380e675759882a799608c974(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_112c24e2385075aaf63af8ccf3917c9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_036b9117380e675759882a799608c974
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6393e3f061134f394b92c74fa45a910d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be803c75a33a8dc9645e1e17cee9836b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6393e3f061134f394b92c74fa45a910d
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b455e2e15f51860ec12129a06959c7c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9683a0478701ff49c75942a6894175c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b455e2e15f51860ec12129a06959c7c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e8cdb880390b0f9962a014cfd597c64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77ebde5fb29b15ef8a2aa7c0216bed91
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_804264fd564f173c0bf75bb3805e92ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da6214305fa66aec1255d158cb873671
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3a1d42107ab2a691a4ba0c51f2a2a88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45a5df12eeac8e0577cd0c425917ffcb
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_022a65e53cdb8c8e23789de0795e3875(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c182ef6ad7a3484c77b90c7bd6c3884(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_022a65e53cdb8c8e23789de0795e3875
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e591b860f109c0dc40c5bb4d08c44eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f9dd2f91919dd3363b2ee948493bf23
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0418cfabfd44735f6f5b0549585c3e49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_892817e00fefcbdaf83111fff3e1845e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_646f9f3a4a13f3edb1292956c576f45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9f00e2539f2656ec00f1a730d8596ab
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_646f9f3a4a13f3edb1292956c576f45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9f00e2539f2656ec00f1a730d8596ab
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2316825700902d2158d8976fb5b87dae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea48907cff4abf212186add4dcc81a2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e996dbca4b6d0ea3509f233ed0676e43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76ae0442bc433aaea1c825491838310a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e996dbca4b6d0ea3509f233ed0676e43
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e64b5dc66bcaa6fb1588a10c9c805456(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9, 14, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d3ba51bb7d0eff3e42f0c16c60bb289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e64b5dc66bcaa6fb1588a10c9c805456
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_860d790826c93b462a392276720df336(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5d325f1c13b4f8f22c963e5e4c845ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7278bcb1900c29142f6bd7fd344b46dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d0f3efd91a7b1c26606c74b8dcfeb06
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_645be0768017910a1fe08384b7882f4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6fd4e538450eca7fdaf97ee13da1151
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f167accd1cc0e26b9cb0fec88225003d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 48, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42275c3950e5ce491083999aa7421e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f167accd1cc0e26b9cb0fec88225003d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_594e1427b2997d0b60f6f1baefbd3b10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a028b93a2b505be42336fbfeaabc8496
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84068586071c1d68593951baff30fe01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3fc14cd84ce523990d807c764218ff
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd0655a5bb6ee5b1ecd1acebbce5a079(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 192, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19212112b80c8a1dbb94b436bf256855(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd0655a5bb6ee5b1ecd1acebbce5a079
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de090b84e3c1ff632117dca16330f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26289840a20b2d14ee3df6ffc4a9b5da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c02f95b70757017d6b1782375719cc69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2852273db094a7119b9fd3d83cd0c314(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6365d45a592cd40665ab4846299a1ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40f562450e1aa492159f263d18988141(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de339f615ef61a0bcc42bc3aff25e671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c83329cd76083e1296073b5769a56d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d66648bf8daeb645b8ec85b6031a357(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bac1c7ce46775fed3bb51a4b20d6dbd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf1bd93ec9125247f3815e9f8f27024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ae986dc21bc03f185156c6251fe04d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.095820665359497]], [[1.5141104459762573]], [[2.101196527481079]], [[2.188419818878174]], [[1.9872820377349854]], [[1.8394256830215454]], [[2.481407880783081]], [[2.1891562938690186]], [[1.9146358966827393]], [[1.3316240310668945]], [[2.3351967334747314]], [[1.9384446144104004]], [[1.7108885049819946]], [[2.8359644412994385]], [[2.057889223098755]], [[1.8670446872711182]], [[1.7827861309051514]], [[1.9267771244049072]], [[1.685118556022644]], [[2.0457208156585693]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_dcd6f0dd12e3d308319d90f2e9a2e3cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_427f0a8b6d830555478927add3e9a006(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46cca03e4edefb5469994489cb80bc9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb2a903cb1cfecd38b48374b5d4157d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d6786cff1d6bfbeed0bebb8fb0802de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5b09fa642e994cc3eca5d16a379f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c02f95b70757017d6b1782375719cc69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1d618ee2028a6f80d985b0845b1345c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb2a903cb1cfecd38b48374b5d4157d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a60f6059b75a8f121e00fa8c2e9c86d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1c2e7a107a7a933a3355cf3b92f14ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeecea24164e2eb6d84893bc67700a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed6bcb2a94c84a7229c957286a21b000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9397135455fcf385d34a64f85ce9f17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d62e8e855e041665133164e73b4012a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_810e53d69c92b4b7658269ea1e80363f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10ea6bb83c5b73cf1b44edad78dacb15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d99ed2cfdd0cdac9d2a96258c2d5ea41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_040534d0e113bee6307004c2646413db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85205f365883c7ff523844ae31929564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45fe4978feb9d75afecfb0b82eb213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f73bbcc884f549fc12b9563b6ec0a915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0e38e773cbeda0d76a9c21c0de1d242(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3bc8cdd1b620b3feab91694db6a29b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de339f615ef61a0bcc42bc3aff25e671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c164536ea3245eb106fd6122435a0dd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c02f95b70757017d6b1782375719cc69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70a25b856c0494fc56ec788be1ac01c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09716f7107eab049bbfe0bf6ce928221(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_210045099c80084e3db0ec8bab68d80d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1d618ee2028a6f80d985b0845b1345c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5b09fa642e994cc3eca5d16a379f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85205f365883c7ff523844ae31929564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64278eca55e69527bbda270a61b1105b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b0078a31a3e704d9dd384be36e0059c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9f3cf87c0a8b1085c9a795809f9db3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_861b5287eb30102c70b6d4cc589b4517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e1c4d3d452aa471b28fb4b19464b1ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f884dd675362bd75c81df54817468ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac4370fd63155604db8cad2468081856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_861b5287eb30102c70b6d4cc589b4517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3872141db037b7f858322a83d843d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c83329cd76083e1296073b5769a56d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_546fd49144316d5b40d967ddeafda4f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f73bbcc884f549fc12b9563b6ec0a915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79f94ee0faa6665041caef732943576f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f33e41eac19202e8f2a4d6ceae16ee2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c9e5398d9b9b4f23fac9bc20919bef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b43f412cbb322feaa0b729812581a97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cd84b15855d87c22d34d00cb0ebfbad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_133ec6f9f2025a8d368023911d9dc0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1da7173724fbb3e6442a5dd42e711adf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03374b769cb1971c801edb96c867a197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_834e3f765e823947350a275bd2b75ebe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb2a903cb1cfecd38b48374b5d4157d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a66c939668115beb0b51c376ca412ce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5b09fa642e994cc3eca5d16a379f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c392189901a2fc55a315f3b45168737(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9cfc5a9740463ca1626b232eeb77acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d12262105f21440d212a76efc6302ef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e1c4d3d452aa471b28fb4b19464b1ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1d618ee2028a6f80d985b0845b1345c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_040534d0e113bee6307004c2646413db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1341b0ffa74c04cf4157fc2bba583bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_104d3dcefe2ba89cec5c484f937c6656(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5fe2179814000c9833c10aa520ac2a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1da7173724fbb3e6442a5dd42e711adf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dca99801782857d1196d987e82d744d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef84b7e41ea0f8d287e86868e6092740(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7df6e504e76a7d5c4a00c2ddf76fb34c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6daf66b2c81dacfc74719702b0a784e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23546d3144135ce1ffce40c37861a475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d97e5cc26cf23a7944e041976bd912e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de090b84e3c1ff632117dca16330f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d12262105f21440d212a76efc6302ef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7df6e504e76a7d5c4a00c2ddf76fb34c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9cfc5a9740463ca1626b232eeb77acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de090b84e3c1ff632117dca16330f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a5644d75e9216134715914ca9b3d3b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb41e2a6f47712c37fae088c278495b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a7d33f59026272f8d3f9dfaa3da71d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7df6e504e76a7d5c4a00c2ddf76fb34c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56e6c2455ff7e2e576a799163ff23746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ce2eaac6779604649dad587a2dc731a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_305f78ea171baba60502e82a07833125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b83658e3cbcdc3c839beeda247b5e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f03cd8eb30bcfdfc25ef159e5504de6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddaee5f6990559b7328dcfd011613518(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10ea6bb83c5b73cf1b44edad78dacb15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2259c8c27c30d8cf490cf9307d8c9725(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_133ec6f9f2025a8d368023911d9dc0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db899b98e221ebf328442b27327ed850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de090b84e3c1ff632117dca16330f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb41e2a6f47712c37fae088c278495b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccd17fb8846d0c08fc01e9b8b42cc082(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85205f365883c7ff523844ae31929564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeecea24164e2eb6d84893bc67700a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_702546cfd6811b848adf01c594801858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5b09fa642e994cc3eca5d16a379f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45fe4978feb9d75afecfb0b82eb213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45fe4978feb9d75afecfb0b82eb213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb759f9f204892c7b7d85c72f664dfe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac4370fd63155604db8cad2468081856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cf55a1c1d55bc7b25dc449fcf191b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c7011002a85def28098a2137c7829d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de090b84e3c1ff632117dca16330f20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_133ec6f9f2025a8d368023911d9dc0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b0078a31a3e704d9dd384be36e0059c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9397135455fcf385d34a64f85ce9f17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1341b0ffa74c04cf4157fc2bba583bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_040534d0e113bee6307004c2646413db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9cfc5a9740463ca1626b232eeb77acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c9e77bb7e79f363cabf529b884eeade(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9e4d775cafef35bc730869e65ef6755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99b34211adf4d28c89fc3a3b91f0d90d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f2365181e0311f345433784f109f7cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e79150a9bd1daf9c601134277f700d88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeecea24164e2eb6d84893bc67700a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f10761fb3b54c4af357954f3830f9cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_616b588b31a74622782cf4fb10c187ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f2365181e0311f345433784f109f7cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c02f95b70757017d6b1782375719cc69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb2a903cb1cfecd38b48374b5d4157d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0aac2932c3e653f10e4066a4f1698611(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1ba1060e785b6f24b3b3e7aad1af5d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6e5e783971ad77780563ad78ed63f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_727576b7a125f3026b259ffd07334722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5af1882fde41b69010ad50b1779bb74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0aac2932c3e653f10e4066a4f1698611(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70ac88f891cfba655acbc7c15d8890a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef84b7e41ea0f8d287e86868e6092740(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a7d33f59026272f8d3f9dfaa3da71d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de339f615ef61a0bcc42bc3aff25e671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2df324a5295e13f09cdaddd15ea8576a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.813080072402954]], [[1.8140665292739868]], [[2.9743213653564453]], [[2.884744882583618]], [[2.6331980228424072]], [[1.6588718891143799]], [[1.8394668102264404]], [[2.028409957885742]], [[3.1729986667633057]], [[3.3984503746032715]], [[1.8893815279006958]], [[1.4538657665252686]], [[2.6711843013763428]], [[2.650304079055786]], [[2.484210968017578]], [[1.9773461818695068]], [[3.4528326988220215]], [[2.093771457672119]], [[1.9031968116760254]], [[1.5846881866455078]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_dcd6f0dd12e3d308319d90f2e9a2e3cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0d818a0e1afce096facda4d522ddfff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c59aae9cebc10426bd1a803bcae241a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c67388641c37b892ef826950d1ddbc2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32bff73f59264bcf601c0245e5551ce7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8cf76afc3b8ad21fabb10768d8a2242(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d12262105f21440d212a76efc6302ef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10ea6bb83c5b73cf1b44edad78dacb15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45fe4978feb9d75afecfb0b82eb213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e46ba1d3b9fc37c149a58c0b6750be9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d15169c3cf73ffd351cc52b9e59e0729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64278eca55e69527bbda270a61b1105b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73109bcb2bc851e9cb5f3da739499e2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cf428e9f649fde90ab0cb05c9ffea9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_403d52dc8bb95da2f7ce6b1f46848355(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f884dd675362bd75c81df54817468ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45fe4978feb9d75afecfb0b82eb213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c2ac3bf5ae4be2e44eef59ce93ae966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.512618064880371]], [[2.1421754360198975]], [[2.6252074241638184]], [[2.2057788372039795]], [[2.862679958343506]], [[2.317392110824585]], [[2.359853744506836]], [[2.177365779876709]], [[1.6993900537490845]], [[1.720333456993103]], [[2.50675892829895]], [[2.556948661804199]], [[3.0345265865325928]], [[2.38387131690979]], [[1.7345192432403564]], [[1.7992361783981323]], [[1.979421615600586]], [[1.703852891921997]], [[1.2495094537734985]], [[2.8299062252044678]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_dcd6f0dd12e3d308319d90f2e9a2e3cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0d818a0e1afce096facda4d522ddfff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32bff73f59264bcf601c0245e5551ce7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d6786cff1d6bfbeed0bebb8fb0802de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d12262105f21440d212a76efc6302ef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_133ec6f9f2025a8d368023911d9dc0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef84b7e41ea0f8d287e86868e6092740(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b5b09fa642e994cc3eca5d16a379f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e082d6fd07c5c9145ab7c12c74e17c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d37d3541bed191ae23a73148a6347c3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_237f8baeb51c9363af27487d3569d9c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dca99801782857d1196d987e82d744d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a2495d045f8523695e7a713746743ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47fa025230c080967c3feb0c40635610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa1e96f3e4d65ab5ab87e65805bfcb95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f884dd675362bd75c81df54817468ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2460f7b91adb193db472023d6b6358c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62178d7514368d9f00d8d04fbcdb7bb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d2a7dab5c171c33f1cbd2277e96f36
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b5c1995bc47dd146310bce98003dada(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed6bcb2a94c84a7229c957286a21b000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2259c8c27c30d8cf490cf9307d8c9725(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9f3cf87c0a8b1085c9a795809f9db3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c54ae28a25b6f70a6d4f3727f02ad316(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_104d3dcefe2ba89cec5c484f937c6656(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0f6953f94dad5c5cbddd8c0c6d25930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a7d33f59026272f8d3f9dfaa3da71d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a7d33f59026272f8d3f9dfaa3da71d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a26cb7b320d1d68c67eb1d1d977404a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92ee837db7f0cccd904f121efb356cef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f9366d8aa52852be44b461e88bfa66f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f10761fb3b54c4af357954f3830f9cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9cfc5a9740463ca1626b232eeb77acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c83329cd76083e1296073b5769a56d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c8ec0731474ea2c1b2c44e06d315359(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10f72c5558103071afac21f3cefce78b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a60f6059b75a8f121e00fa8c2e9c86d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b7bc250a339a7653628657c52f6ea6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b62ed7256944f66c36559fb75d33d959
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()