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
    class PrimitiveOp_77143eb31096c3543f9d60ee35bc152d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38b4b3dbbfcf6c3867dbbfdde2226c3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77143eb31096c3543f9d60ee35bc152d
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[0], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4cf5cd00b031127558044b25ce5d46c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([4395], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_74ccd3eb6a0db5fa2d98f4d61afccede(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[0], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0860a4a21695f5da810b1d51cf066660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ccd3eb6a0db5fa2d98f4d61afccede
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[0], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7708bdf174d8e23bcb07effd2ffd4c3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02e7604769688f9a36bf2624176fd032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_02e7604769688f9a36bf2624176fd032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_baf1634cc4642838a91880737d61f42f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.007114458363503218, 0.015159129165112972]], [[0.008331325836479664, 0.043586645275354385]], [[0.002317468635737896, 0.01679067313671112]], [[0.11468011885881424, 0.00011547928443178535]], [[0.1574922502040863, 7.65175063861534e-05]], [[0.0005593742243945599, 0.010259552858769894]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_849da67384865ca7e78cd1df2970c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.02296661026775837, 0.030401743948459625]], [[0.01801164075732231, 0.007291101384907961]], [[0.009565435349941254, 0.20005014538764954]], [[0.09065292775630951, 0.005057369824498892]], [[0.0036611619871109724, 0.18642355501651764]], [[0.01378479041159153, 0.022222062572836876]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_94e7ec325fbf4935be599f369c609aaa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6246cfdf1c32ed9545c56b6526636f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94e7ec325fbf4935be599f369c609aaa
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_033d7987ae82fc83824cb88079999ddf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23263609409332275, 0.11114535480737686, 0.009845714084804058, 0.05015848949551582, 0.16455012559890747, 0.024160051718354225, 0.16110481321811676, 0.14081424474716187, 0.1160120740532875, 0.15018466114997864, 0.22150282561779022, 0.23228415846824646, 0.1450996696949005, 0.0655934289097786, 0.23648183047771454, 0.1585187166929245], dtype='float32').reshape([16]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8a1a23bacf58b4ddfadacce567417cbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b0510b072b1d181a080cbe448fdd9c52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7ac05b548a9c4c551a0f54a56eabf274(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de504388c40675b6c69a43e621757b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_895fd06472fffb6cb451bb012c824929(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[0], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab3a3aba8e9238b55e8277c6c89fe35e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_ab3a3aba8e9238b55e8277c6c89fe35e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8ba798e1feca437dae341b8a92b76da6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d573d41f860c53bcc0560ad6c0fc4c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.011946693062782288, 0.22909292578697205, 0.15888167917728424, 0.015180230140686035], [0.23457637429237366, 0.1387505978345871, 0.23476874828338623, 0.039133816957473755], [0.3682340085506439, 0.2067251205444336, 0.04031139612197876, 0.008929014205932617], [0.2650768756866455, 0.06502872705459595, 0.3271848261356354, 0.06669780611991882], [0.0899767279624939, 0.10820017755031586, 0.05213071405887604, 0.06857089698314667]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_647d6425039bd307ab167eaf4c7acecb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d70b478c5a61d918429c4a8477db7a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_647d6425039bd307ab167eaf4c7acecb
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_81175ceed4fac8b24e382eced1e673c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2601255178451538, 0.2813035547733307, 0.3010902404785156, 0.11394605040550232], [0.1276143491268158, 0.1411561220884323, 0.2907564043998718, 0.004007205367088318], [0.3767981231212616, 0.019813083112239838, 0.08903086185455322, 0.06486682593822479], [0.1276143491268158, 0.1411561220884323, 0.2907564043998718, 0.004007205367088318], [0.3767981231212616, 0.019813083112239838, 0.08903086185455322, 0.06486682593822479]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5173a1deb9477c53fa1e809e77a221d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ccd3eb6a0db5fa2d98f4d61afccede
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_89ea218c93eebc8ffbc15cbcebbf2310(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a09122046da5f192b9638de8195d58f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a09122046da5f192b9638de8195d58f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_78dc43609c4fd7791d89b7bd27ec8471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3336668610572815, 0.3895747661590576, 0.027006596326828003, 0.17651008069515228], [0.2637220025062561, 0.02544151246547699, 0.1283944547176361, 0.13914698362350464], [0.04708457738161087, 0.0057485103607177734, 0.3778526186943054, 0.16070452332496643], [0.2637220025062561, 0.02544151246547699, 0.1283944547176361, 0.13914698362350464], [0.04708457738161087, 0.0057485103607177734, 0.3778526186943054, 0.16070452332496643], [0.06639361381530762, 0.2864929735660553, 0.26026010513305664, 0.3718429505825043], [0.06639361381530762, 0.2864929735660553, 0.26026010513305664, 0.3718429505825043]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dd1bd9c3a912665d2ed5f02b964dac28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2a72ca8ab6367501ee4c932512d33193(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53ddadc6c0191e0021b057a229cf28b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a72ca8ab6367501ee4c932512d33193
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5adc3daac58cc726e17e9f22a676319e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_de504388c40675b6c69a43e621757b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6ab522254cecb9ac1946f36671f9ac20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_6ab522254cecb9ac1946f36671f9ac20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8ba798e1feca437dae341b8a92b76da6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_be8e5d4585f72a06044b5897e57221b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_446b0e537bd63e5d21a4b379e0ef6d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8e5d4585f72a06044b5897e57221b1
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0763fb60f3608c6f9b6bc7981f649aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1467335969209671, 0.09901197254657745, 0.17062722146511078, 0.14475056529045105, 0.187338724732399, 0.11645021289587021, 0.18754443526268005, 0.029472628608345985, 0.20399020612239838, 0.044046949595212936, 0.2176395058631897, 0.011811324395239353, 0.2276376336812973, 0.15333817899227142, 0.0458989217877388, 0.1335170567035675, 0.18884292244911194, 0.2357255220413208, 0.03989668935537338, 0.10343722999095917, 0.1528700441122055, 0.17171022295951843, 0.10407722741365433, 0.17606297135353088], dtype='float32').reshape([24]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d4a89d31c5083d8ac74c6eedd3492cd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_143a653f580ef967fad0c0ce94f1876f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_143a653f580ef967fad0c0ce94f1876f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ad5c687cd251fa5695c64cb452a70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_846feef827d454216bb892b2c2bd6d3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.02470981515944004, 0.030571145936846733, 0.15698277950286865, 0.08978959918022156], dtype='float32').reshape([4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_0530463d4f411e7a0f7ef34fd6c842ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1509297639131546, 0.1552731692790985, 0.056120842695236206, 0.15356037020683289], [0.07212063670158386, 0.3970109820365906, 0.12442433834075928, 0.38769468665122986], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079], [0.12647047638893127, 0.19000458717346191, 0.3279547691345215, 0.06161805987358093], [0.12647047638893127, 0.19000458717346191, 0.3279547691345215, 0.06161805987358093], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_c8f29091d1f23c001228c2d6172d5b96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3127082586288452, 0.048425883054733276, 0.1581532061100006, 0.0485401451587677], [0.3431030213832855, 0.06575411558151245, 0.0661439597606659, 0.27380648255348206], [0.06341108679771423, 0.11628088355064392, 0.10398587584495544, 0.028025232255458832], [0.17723333835601807, 0.2572104036808014, 0.20969152450561523, 0.18656745553016663], [0.3127082586288452, 0.048425883054733276, 0.1581532061100006, 0.0485401451587677]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_08bd2e2b1823be913ac8378609c293cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9ad5c687cd251fa5695c64cb452a70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_cb1b4808747f3213fbfba0fb3ad8d5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14037862420082092, 0.3072340786457062, 0.233082577586174, 0.22903814911842346], [0.14534536004066467, 0.06035545468330383, 0.3983846604824066, 0.18683511018753052], [0.041945427656173706, 0.24708035588264465, 0.2164594829082489, 0.12575536966323853], [0.003619551658630371, 0.1542983204126358, 0.1320928931236267, 0.09877075254917145]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_c9be82b37a779006e5c306132eff72f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.sum(input_0, input_1, None, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10ec9a6b58439ff34862782ce1433301(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9be82b37a779006e5c306132eff72f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_446b0e537bd63e5d21a4b379e0ef6d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8e5d4585f72a06044b5897e57221b1
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7695efa726dd5344f89aa265298da191(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_144dc7a4bda71d46fc9fe902db191fcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d125137b97e798a79e92443531e65427(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_606352e16f3ce8adfe8b2381b6c4f503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_15b44821accfaff30fd1147de8db445a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_15b44821accfaff30fd1147de8db445a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5fd6e0d823c8fc1913797c338898abf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9be82b37a779006e5c306132eff72f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_54787495bf6bceed41ff1dd881cfbc7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17318011820316315, 0.0040985941886901855, 0.10722425580024719, 0.24594347178936005], [0.17318011820316315, 0.0040985941886901855, 0.10722425580024719, 0.24594347178936005], [0.2724611163139343, 0.14107611775398254, 0.3538140058517456, 0.029049724340438843], [0.0411318838596344, 0.2469097077846527, 0.05615696310997009, 0.09188510477542877], [0.09708136320114136, 0.41709190607070923, 0.1974916011095047, 0.15154391527175903], [0.022501900792121887, 0.2542913854122162, 0.010012298822402954, 0.1757974624633789], [0.01827526092529297, 0.11807702481746674, 0.04941102862358093, 0.02130529098212719]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_172f6c4046f7a21ca652f8a8283ce6d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e4cfaf96de1de7304b4f39107ad61b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e4cfaf96de1de7304b4f39107ad61b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_2e65403984cd13efe8f8e43421fae953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([4909], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_11c28db26c6c62d247d4899dfe5131a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([1242], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_39b290f3aa6c2d170c89ef544d67dc54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ccd3eb6a0db5fa2d98f4d61afccede
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_98e1501b3613a136fcdf0278221ef8a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6c7e95656839133a88c3b3dc5f7510c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_6c7e95656839133a88c3b3dc5f7510c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_94125f9aa8d9ac8812d8d7ec334f3064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08719142526388168, 0.40703436732292175, 0.14703017473220825, 0.20497220754623413], [0.21442238986492157, 0.18900448083877563, 0.22183440625667572, 0.010784268379211426], [0.21442238986492157, 0.18900448083877563, 0.22183440625667572, 0.010784268379211426], [0.2041129320859909, 0.1454058587551117, 0.39465102553367615, 0.02922683209180832], [0.02670140564441681, 0.1606736183166504, 0.18615491688251495, 0.40934592485427856], [0.3336673974990845, 0.08125244081020355, 0.2736116051673889, 0.33980047702789307]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_f905dd2d33a202374bef7569e6dc16fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2ea2c79d4be4e31ef9feaac6ecf79fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f905dd2d33a202374bef7569e6dc16fc
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59f7409ec9197fbd0e416fbeaa8e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_db01bc48173ed80f6896d765edee63fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3fd96cccb5ac8db5709402148075c355(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db01bc48173ed80f6896d765edee63fd
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a2e12a6737afded41b61b97218625263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d4ead29c8ae65488262ce539b0652518(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d4ead29c8ae65488262ce539b0652518(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_67dd4f2bc5596c6a7a0a01ce54234ff2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cf28536c928e457ca8ba1af7fe0c22b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_cf28536c928e457ca8ba1af7fe0c22b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3e95044af55f12ab317b6b0e85e69184(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f143985e10eb41f56252f38dca934747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f143985e10eb41f56252f38dca934747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4396c76287f2f41fa85b765be3e7db50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_16344afd53d26fc87d987706cbf46fa7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ce4730cb65ba5f15186486e234a1e13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16344afd53d26fc87d987706cbf46fa7
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf88ce40921ac42d3e835d568e89b99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_53ddadc6c0191e0021b057a229cf28b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a72ca8ab6367501ee4c932512d33193
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5f8560f726ef563ed98aac0daa4c5430(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0342906229197979, 0.2616975009441376, 0.11236084252595901, 0.23144856095314026, 0.1474074274301529, 0.14624141156673431, 0.11792205274105072, 0.0706254094839096, 0.16403816640377045, 0.1672552078962326, 0.010639270767569542, 0.04043728858232498, 0.19575585424900055, 0.12154953181743622, 0.2397426813840866, 0.13022302091121674, 0.016859127208590508, 0.00764064583927393, 0.11539359390735626, 0.258635014295578], dtype='float32').reshape([20]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5c7c0babbde5081eda58f8985d506e5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([17457], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4ce4730cb65ba5f15186486e234a1e13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16344afd53d26fc87d987706cbf46fa7
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_aa61e97e8af5f9b53cd5313507663df4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_00baa7162cb1acd818deaf6a54f183d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5cb553998a8af8a2cc48cd47a6f32c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e50c324c403aa7ae8df51524a0c59cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e50c324c403aa7ae8df51524a0c59cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_c9c932fce0d50cfbeda7ffc86a5e04dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39016c3c718b96e0ac175b2111e69f48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9c932fce0d50cfbeda7ffc86a5e04dc
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0db75a1e8834dac9c5528ee82c493da2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e066628d241d012a3f435f428566e85e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05997924134135246, 0.17256204783916473, 0.29755768179893494, 0.03335516154766083], [0.43779996037483215, 0.19319099187850952, 0.01612231135368347, 0.18234547972679138], [0.17438164353370667, 0.24513502418994904, 0.027111470699310303, 0.13506805896759033], [0.17438164353370667, 0.24513502418994904, 0.027111470699310303, 0.13506805896759033], [0.13616040349006653, 0.334640234708786, 0.23136883974075317, 0.01873648166656494]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_59f7409ec9197fbd0e416fbeaa8e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_460ff43be3a1414b62203003ade6e987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_2e709bb2ea6ad6308bb18b8c331f580b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7082d29ca054cce66b3806e9a64b9659(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_20aee50bafc74873e7915222fbf50be6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4e0906eaded89577362bfec5b36937c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4e0906eaded89577362bfec5b36937c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9e7c58d5ff39a08a05c3ad80460ccf03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3046402335166931, 0.07770112156867981, 0.14432917535305023, 0.180344358086586], [0.03918623924255371, 0.019960127770900726, 0.028510063886642456, 0.033728450536727905], [0.017332687973976135, 0.28235897421836853, 0.3564165532588959, 0.1784440129995346], [0.3046402335166931, 0.07770112156867981, 0.14432917535305023, 0.180344358086586], [0.055790454149246216, 0.36793333292007446, 0.027703553438186646, 0.24950526654720306], [0.23903867602348328, 0.20549030601978302, 0.09378381073474884, 0.2969999313354492], [0.055790454149246216, 0.36793333292007446, 0.027703553438186646, 0.24950526654720306]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_aa1454c72de0b09199d854f0254e29b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bf88ce40921ac42d3e835d568e89b99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_38b4b3dbbfcf6c3867dbbfdde2226c3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77143eb31096c3543f9d60ee35bc152d
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_53ff538d9e1ed04497e2838960074f05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4395], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_daf210a42745410a0dc47c40ef621843(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53ff538d9e1ed04497e2838960074f05
        def get_inputs(self):
            return [
                paddle.uniform([4395], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_80d7ea76df7cab71b8f62eed7f2fccc4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8732, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c7537c668a7c7a5617668e590592f175(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80d7ea76df7cab71b8f62eed7f2fccc4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2a765369f8effeb5abb83d2646745173(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e807c2127db5201fc65fc3c8aecfdfa3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a765369f8effeb5abb83d2646745173
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_23f6b988ad50ceb8714def96a28d3964(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 21824, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a49d146228f2c1702eb94cfac0dad838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23f6b988ad50ceb8714def96a28d3964
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a49d146228f2c1702eb94cfac0dad838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23f6b988ad50ceb8714def96a28d3964
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_01bba0a914ebd1f491ac9c5013dccca5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5907a69adc0ff4419ab0ad7983565bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01bba0a914ebd1f491ac9c5013dccca5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.007114458363503218, 0.015159129165112972]], [[0.008331325836479664, 0.043586645275354385]], [[0.002317468635737896, 0.01679067313671112]], [[0.11468011885881424, 0.00011547928443178535]], [[0.1574922502040863, 7.65175063861534e-05]], [[0.0005593742243945599, 0.010259552858769894]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6a6b39241c2d158f03e15b16bc5e7325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01bba0a914ebd1f491ac9c5013dccca5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.02296661026775837, 0.030401743948459625]], [[0.01801164075732231, 0.007291101384907961]], [[0.009565435349941254, 0.20005014538764954]], [[0.09065292775630951, 0.005057369824498892]], [[0.0036611619871109724, 0.18642355501651764]], [[0.01378479041159153, 0.022222062572836876]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6246cfdf1c32ed9545c56b6526636f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94e7ec325fbf4935be599f369c609aaa
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_1fc6813c50b3e0aa20e38d15ae3e4b43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_affa0231f11ba0c619bc610e64018b21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fc6813c50b3e0aa20e38d15ae3e4b43
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23263609409332275, 0.11114535480737686, 0.009845714084804058, 0.05015848949551582, 0.16455012559890747, 0.024160051718354225, 0.16110481321811676, 0.14081424474716187, 0.1160120740532875, 0.15018466114997864, 0.22150282561779022, 0.23228415846824646, 0.1450996696949005, 0.0655934289097786, 0.23648183047771454, 0.1585187166929245], dtype='float32').reshape([16]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2501c76fc228dbc4074f7954b6e23efa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d967ac90ae8e3c6bde485844427c9b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2501c76fc228dbc4074f7954b6e23efa
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_a29f4c1bab50ac88bc0644a2fa738f4a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b583c472e943ab5ff4c3d87484a50337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29f4c1bab50ac88bc0644a2fa738f4a
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2b293ee699f31fb94ec74aff25ed9715(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f580956c432b416aabd035fe6cd5a6cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b293ee699f31fb94ec74aff25ed9715
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_b8e73f18898db6b531f2a02a1d366e03(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4bd78313895267b946031817d4a0c1c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8e73f18898db6b531f2a02a1d366e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d3f89749ccd65e200c61d4249bb58c26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb1ea76330a8a5e6e7e5e0aad40104fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f89749ccd65e200c61d4249bb58c26
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bb1ea76330a8a5e6e7e5e0aad40104fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f89749ccd65e200c61d4249bb58c26
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_560e12b283db9a4883f57e5f27721a38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2050603a276a2955f485b5d686803ea1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560e12b283db9a4883f57e5f27721a38
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_eb83881230c2be7127771fc7af7c350a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a50be060a0845d00258dbae09888955c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.011946693062782288, 0.22909292578697205, 0.15888167917728424, 0.015180230140686035], [0.23457637429237366, 0.1387505978345871, 0.23476874828338623, 0.039133816957473755], [0.3682340085506439, 0.2067251205444336, 0.04031139612197876, 0.008929014205932617], [0.2650768756866455, 0.06502872705459595, 0.3271848261356354, 0.06669780611991882], [0.0899767279624939, 0.10820017755031586, 0.05213071405887604, 0.06857089698314667]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3d70b478c5a61d918429c4a8477db7a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_647d6425039bd307ab167eaf4c7acecb
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_45451e7806a9b8f23c123e0d352d4354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2601255178451538, 0.2813035547733307, 0.3010902404785156, 0.11394605040550232], [0.1276143491268158, 0.1411561220884323, 0.2907564043998718, 0.004007205367088318], [0.3767981231212616, 0.019813083112239838, 0.08903086185455322, 0.06486682593822479], [0.1276143491268158, 0.1411561220884323, 0.2907564043998718, 0.004007205367088318], [0.3767981231212616, 0.019813083112239838, 0.08903086185455322, 0.06486682593822479]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_b39b3d49a514b6e7139b8c486993bfe6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e174e64c8e2ee3a78ee086b6aa0e4430(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b39b3d49a514b6e7139b8c486993bfe6
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_0cff9e87f018fadb9e594fea702136d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e99eef50a533c6dd5a257410cfcd713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cff9e87f018fadb9e594fea702136d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_ef99e95c560c0a197570e9b64d88ef7f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d809fddda5eaa927e6f3d4a37ffe037b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef99e95c560c0a197570e9b64d88ef7f
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d809fddda5eaa927e6f3d4a37ffe037b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef99e95c560c0a197570e9b64d88ef7f
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18512b00567a73e2d61985f70c7c69a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3336668610572815, 0.3895747661590576, 0.027006596326828003, 0.17651008069515228], [0.2637220025062561, 0.02544151246547699, 0.1283944547176361, 0.13914698362350464], [0.04708457738161087, 0.0057485103607177734, 0.3778526186943054, 0.16070452332496643], [0.2637220025062561, 0.02544151246547699, 0.1283944547176361, 0.13914698362350464], [0.04708457738161087, 0.0057485103607177734, 0.3778526186943054, 0.16070452332496643], [0.06639361381530762, 0.2864929735660553, 0.26026010513305664, 0.3718429505825043], [0.06639361381530762, 0.2864929735660553, 0.26026010513305664, 0.3718429505825043]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_79e69cd7ea52f2675476d949b4ae2007(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_143effb5d3314b58c5de56df971220e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79e69cd7ea52f2675476d949b4ae2007
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_53ddadc6c0191e0021b057a229cf28b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a72ca8ab6367501ee4c932512d33193
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2cecba88ecc5ad045b13c158bbc3f8c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5fb414615430dd77f905cd5307b4d435(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2cecba88ecc5ad045b13c158bbc3f8c3
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4bd78313895267b946031817d4a0c1c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8e73f18898db6b531f2a02a1d366e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_62b6e49f3eb2d18ceb55d8bf64650b8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_185705fbeafbaa9d27726ad3045fd013(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62b6e49f3eb2d18ceb55d8bf64650b8a
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_185705fbeafbaa9d27726ad3045fd013(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62b6e49f3eb2d18ceb55d8bf64650b8a
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_2050603a276a2955f485b5d686803ea1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560e12b283db9a4883f57e5f27721a38
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_446b0e537bd63e5d21a4b379e0ef6d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8e5d4585f72a06044b5897e57221b1
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_abf701896c1b5c2cda506d95ebfa0da8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af9baddc4f03f5417be644160d197e05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abf701896c1b5c2cda506d95ebfa0da8
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1467335969209671, 0.09901197254657745, 0.17062722146511078, 0.14475056529045105, 0.187338724732399, 0.11645021289587021, 0.18754443526268005, 0.029472628608345985, 0.20399020612239838, 0.044046949595212936, 0.2176395058631897, 0.011811324395239353, 0.2276376336812973, 0.15333817899227142, 0.0458989217877388, 0.1335170567035675, 0.18884292244911194, 0.2357255220413208, 0.03989668935537338, 0.10343722999095917, 0.1528700441122055, 0.17171022295951843, 0.10407722741365433, 0.17606297135353088], dtype='float32').reshape([24]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_9e3acbc65041fbb551e7cf50223bcb78(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ead18cb5cb40b1978643afa8f1ef05bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e3acbc65041fbb551e7cf50223bcb78
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_937910a6149f49ca4db3889494876c00(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e02794a5eddabd953bb5821ba8b0daf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_937910a6149f49ca4db3889494876c00
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e02794a5eddabd953bb5821ba8b0daf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_937910a6149f49ca4db3889494876c00
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9ad5c687cd251fa5695c64cb452a70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_fc26a21b27eba3cdde678c037676765a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0960bbfbb181f125e882cea616206d37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc26a21b27eba3cdde678c037676765a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.02470981515944004, 0.030571145936846733, 0.15698277950286865, 0.08978959918022156], dtype='float32').reshape([4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_5a73f36534c5f3febd704a5d55d75107(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce52d547e57fd6b6a51f7caca6534ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a73f36534c5f3febd704a5d55d75107
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1509297639131546, 0.1552731692790985, 0.056120842695236206, 0.15356037020683289], [0.07212063670158386, 0.3970109820365906, 0.12442433834075928, 0.38769468665122986], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079], [0.12647047638893127, 0.19000458717346191, 0.3279547691345215, 0.06161805987358093], [0.12647047638893127, 0.19000458717346191, 0.3279547691345215, 0.06161805987358093], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b7389b881138da092f932e63c10eadb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3127082586288452, 0.048425883054733276, 0.1581532061100006, 0.0485401451587677], [0.3431030213832855, 0.06575411558151245, 0.0661439597606659, 0.27380648255348206], [0.06341108679771423, 0.11628088355064392, 0.10398587584495544, 0.028025232255458832], [0.17723333835601807, 0.2572104036808014, 0.20969152450561523, 0.18656745553016663], [0.3127082586288452, 0.048425883054733276, 0.1581532061100006, 0.0485401451587677]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_fbdc79e521f488ee17c21203b9a730dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5be42f93d2b5e6dddb66476d60b3738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbdc79e521f488ee17c21203b9a730dd
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9ad5c687cd251fa5695c64cb452a70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_d4b14492b5693ed40c5ffef38cd64f08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c9fb924340956c6f9eac1e31c577ab7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4b14492b5693ed40c5ffef38cd64f08
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14037862420082092, 0.3072340786457062, 0.233082577586174, 0.22903814911842346], [0.14534536004066467, 0.06035545468330383, 0.3983846604824066, 0.18683511018753052], [0.041945427656173706, 0.24708035588264465, 0.2164594829082489, 0.12575536966323853], [0.003619551658630371, 0.1542983204126358, 0.1320928931236267, 0.09877075254917145]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_80ed203e1f4ee446d2eab177758f4979(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.sum(input_0, input_1, None, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e871d4faff80bcea71bef440a067511(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80ed203e1f4ee446d2eab177758f4979
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_446b0e537bd63e5d21a4b379e0ef6d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8e5d4585f72a06044b5897e57221b1
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_99b88328100de8299c75292e48456c84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea8445d8f11ace50dec1eb6f5a9e9bab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99b88328100de8299c75292e48456c84
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_130cd2fde7cdef298e7f6dda389e421e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3aef3d2205d2c18cb8993f835bab13a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_130cd2fde7cdef298e7f6dda389e421e
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_d50b6a56e0c6d4ddaf39e73e80c9c908(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8816], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e27627733f91d33f053f74a33937c273(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d50b6a56e0c6d4ddaf39e73e80c9c908
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_b5c8be451b55b526063ce859b462f210(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_631651d8b0f96a2da9722c80e221be9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5c8be451b55b526063ce859b462f210
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a9253358acfeaf8b905ef95fcae245f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a48591f2b0ad003bb526fa9855b04075(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9253358acfeaf8b905ef95fcae245f9
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a48591f2b0ad003bb526fa9855b04075(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9253358acfeaf8b905ef95fcae245f9
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_87bacf1654c81504c15bd3ae930a62ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.sum(input_0, input_1, None, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_470fb447f8c4a46f6b22062043060f14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87bacf1654c81504c15bd3ae930a62ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_c6b6eee33bc4502fec4b71794a3b2bc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17318011820316315, 0.0040985941886901855, 0.10722425580024719, 0.24594347178936005], [0.17318011820316315, 0.0040985941886901855, 0.10722425580024719, 0.24594347178936005], [0.2724611163139343, 0.14107611775398254, 0.3538140058517456, 0.029049724340438843], [0.0411318838596344, 0.2469097077846527, 0.05615696310997009, 0.09188510477542877], [0.09708136320114136, 0.41709190607070923, 0.1974916011095047, 0.15154391527175903], [0.022501900792121887, 0.2542913854122162, 0.010012298822402954, 0.1757974624633789], [0.01827526092529297, 0.11807702481746674, 0.04941102862358093, 0.02130529098212719]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_71197a2f67f34b75265409bbca3b2c9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e7feed7ad207cc9304484b520b5d1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71197a2f67f34b75265409bbca3b2c9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_53f357ae2271922fc1cf0f499736a7af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab96440aff452f2adfac6974b9ecd02b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53f357ae2271922fc1cf0f499736a7af
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_ab96440aff452f2adfac6974b9ecd02b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53f357ae2271922fc1cf0f499736a7af
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_1944a668b85d8f4eb6cfd7dea7275dca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4909], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3dfba1a62006982b5b8d00c1609f56e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1944a668b85d8f4eb6cfd7dea7275dca
        def get_inputs(self):
            return [
                paddle.uniform([4909], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_cd8916c2f2893ea1d278daf0da19b31b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1242], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_114af8edfa251d53b9771b46e124ae95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd8916c2f2893ea1d278daf0da19b31b
        def get_inputs(self):
            return [
                paddle.uniform([1242], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_5fae2df3e98feddbe8e570a97067b942(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2434, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8939cb35c1c65ffc23f1cc0968ba4a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fae2df3e98feddbe8e570a97067b942
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_764698473c14f18b7d6a912c6e57a70b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d96ab46978e5d2e58dc8d87582ab65b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_764698473c14f18b7d6a912c6e57a70b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a560b9179e71d6fa6b4327a9e7396612(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a79bf4c6c565bc6fea57eca27253d6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a560b9179e71d6fa6b4327a9e7396612
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a79bf4c6c565bc6fea57eca27253d6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a560b9179e71d6fa6b4327a9e7396612
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_12fce5784b7eafd9cb2912ebce55d187(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a73f36534c5f3febd704a5d55d75107
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08719142526388168, 0.40703436732292175, 0.14703017473220825, 0.20497220754623413], [0.21442238986492157, 0.18900448083877563, 0.22183440625667572, 0.010784268379211426], [0.21442238986492157, 0.18900448083877563, 0.22183440625667572, 0.010784268379211426], [0.2041129320859909, 0.1454058587551117, 0.39465102553367615, 0.02922683209180832], [0.02670140564441681, 0.1606736183166504, 0.18615491688251495, 0.40934592485427856], [0.3336673974990845, 0.08125244081020355, 0.2736116051673889, 0.33980047702789307]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_434501a68dccfcbe97a3e2547f3bf465(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 2, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a38b7f4d630ea2dc4ee3cc36e726da84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_434501a68dccfcbe97a3e2547f3bf465
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_59f7409ec9197fbd0e416fbeaa8e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c1a76ac269f24645e1336c17fccd09e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 2, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4158663c673624be4e4626e5c737a28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a76ac269f24645e1336c17fccd09e6
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4b063ff6a69af8aa6000cabefc30132d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b83a787529cc4b356c6ee0694e664d31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b063ff6a69af8aa6000cabefc30132d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_cbda09308c22f1630cbe69f1744b2697(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_998acea96d0990e2a825d869de5ec742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbda09308c22f1630cbe69f1744b2697
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_998acea96d0990e2a825d869de5ec742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbda09308c22f1630cbe69f1744b2697
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_cdcb0b1979c53318fa03c18ae201559d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48e3638323596146d661c1afcc398030(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdcb0b1979c53318fa03c18ae201559d
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d29900c9dc8684be0e3ac1f0d19afe36(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d51702ec5b879aa40d28f909468a781(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d29900c9dc8684be0e3ac1f0d19afe36
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_6d51702ec5b879aa40d28f909468a781(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d29900c9dc8684be0e3ac1f0d19afe36
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_5570b29155890a52498d0cf11cd9ba75(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08464a6ab95d6ed0e74a6bdea6832d4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5570b29155890a52498d0cf11cd9ba75
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d4dc12028895bc8817dd08fc55dcc68c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ab9c34364a0c5931e43cd9fccb73384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4dc12028895bc8817dd08fc55dcc68c
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_6ab9c34364a0c5931e43cd9fccb73384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4dc12028895bc8817dd08fc55dcc68c
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_1896e33fb6f78c7fc1a95c00ebcd8511(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ae9dbea1777e2d4eff38c0e422089f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1896e33fb6f78c7fc1a95c00ebcd8511
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4ce4730cb65ba5f15186486e234a1e13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16344afd53d26fc87d987706cbf46fa7
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bf88ce40921ac42d3e835d568e89b99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_53ddadc6c0191e0021b057a229cf28b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a72ca8ab6367501ee4c932512d33193
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_0941b9d7f3be9adb3a2864993bcc930f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8fef907296722aa1eefc6d86b754c01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0941b9d7f3be9adb3a2864993bcc930f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0342906229197979, 0.2616975009441376, 0.11236084252595901, 0.23144856095314026, 0.1474074274301529, 0.14624141156673431, 0.11792205274105072, 0.0706254094839096, 0.16403816640377045, 0.1672552078962326, 0.010639270767569542, 0.04043728858232498, 0.19575585424900055, 0.12154953181743622, 0.2397426813840866, 0.13022302091121674, 0.016859127208590508, 0.00764064583927393, 0.11539359390735626, 0.258635014295578], dtype='float32').reshape([20]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_8f8d1e75b21b1d7ac3b39beef2c71a1d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17457], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce095fe114052f1ec5ecaac86db24992(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f8d1e75b21b1d7ac3b39beef2c71a1d
        def get_inputs(self):
            return [
                paddle.uniform([17457], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4ce4730cb65ba5f15186486e234a1e13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16344afd53d26fc87d987706cbf46fa7
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_8ba3ee3da71cc0a1aed44a2b8e1918b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccb7b521eb62cffd75dd82bf438c0e23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ba3ee3da71cc0a1aed44a2b8e1918b5
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_207615440fc3984110e41d8cf2f093c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6aeec66aa321951a905de48adf22c28f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_207615440fc3984110e41d8cf2f093c7
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_5aa03630179bd3a9bb53d565812b8606(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd2b90c9f9a4e6b9381940177635432b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5aa03630179bd3a9bb53d565812b8606
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_0fc2c2e3b8e7c6f42bc9057798bf3b12(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_541aec7b3cf6f97a09cd6cba4571c4f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fc2c2e3b8e7c6f42bc9057798bf3b12
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_541aec7b3cf6f97a09cd6cba4571c4f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fc2c2e3b8e7c6f42bc9057798bf3b12
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_39016c3c718b96e0ac175b2111e69f48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9c932fce0d50cfbeda7ffc86a5e04dc
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_565ca66f3b0a9b753324b149fe5a6d47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[551], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f603bb0ab28c9ea6ee6000518fdf7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_565ca66f3b0a9b753324b149fe5a6d47
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_326d65cad24dcd2062748362b2066966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05997924134135246, 0.17256204783916473, 0.29755768179893494, 0.03335516154766083], [0.43779996037483215, 0.19319099187850952, 0.01612231135368347, 0.18234547972679138], [0.17438164353370667, 0.24513502418994904, 0.027111470699310303, 0.13506805896759033], [0.17438164353370667, 0.24513502418994904, 0.027111470699310303, 0.13506805896759033], [0.13616040349006653, 0.334640234708786, 0.23136883974075317, 0.01873648166656494]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_59f7409ec9197fbd0e416fbeaa8e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_0074de74d35da0b8cab911a4dcf4a2ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_31ed1b4fa037ae275e80ae375bd36016(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0074de74d35da0b8cab911a4dcf4a2ab
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_ce9b79ab6a69d6a3069fb8709fe8d6e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2204], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9937f0ea0b09a1390ed996cf7ba7c491(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce9b79ab6a69d6a3069fb8709fe8d6e1
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_1f9bbf2183b3ee015de34b3331aeafa9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb8045df7c96bb610ef9c4b6f837121b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f9bbf2183b3ee015de34b3331aeafa9
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_bea34f89e68042dc97707730dd69a5cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1eb59b2df749e44b0da330afde6885ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bea34f89e68042dc97707730dd69a5cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_2968aea6ddc676039dfb0e4f2f6aa271(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdab87efdb31978195f94404f73eeb3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2968aea6ddc676039dfb0e4f2f6aa271
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_cdab87efdb31978195f94404f73eeb3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2968aea6ddc676039dfb0e4f2f6aa271
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a41e68d145f6df6c0bb67581e85b973e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3046402335166931, 0.07770112156867981, 0.14432917535305023, 0.180344358086586], [0.03918623924255371, 0.019960127770900726, 0.028510063886642456, 0.033728450536727905], [0.017332687973976135, 0.28235897421836853, 0.3564165532588959, 0.1784440129995346], [0.3046402335166931, 0.07770112156867981, 0.14432917535305023, 0.180344358086586], [0.055790454149246216, 0.36793333292007446, 0.027703553438186646, 0.24950526654720306], [0.23903867602348328, 0.20549030601978302, 0.09378381073474884, 0.2969999313354492], [0.055790454149246216, 0.36793333292007446, 0.027703553438186646, 0.24950526654720306]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_c7e13ec6532cbaeb3a524e89f7432391(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1736a0eb075bf9c1b887ef90311e4e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e13ec6532cbaeb3a524e89f7432391
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bf88ce40921ac42d3e835d568e89b99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68d8464483249990eba24b88000c1544(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78b35c23743c05a28084e5babad4dce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([4395], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_8f44c800411e46248c5ab294ad2a7609(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44d7f512597727eb5520c947c8ef938f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f44c800411e46248c5ab294ad2a7609
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_256292bbbeccbfef50cde6a645d4e339(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b38365f842993341826d5b8d4c3cc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_0d7385a53f751734fdb25c5922a528b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06b5f2c40c7ffda53dbf719dbd5b2d4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_06b5f2c40c7ffda53dbf719dbd5b2d4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ab5055b607ca754b0272fe1de75b577c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.007114458363503218, 0.015159129165112972]], [[0.008331325836479664, 0.043586645275354385]], [[0.002317468635737896, 0.01679067313671112]], [[0.11468011885881424, 0.00011547928443178535]], [[0.1574922502040863, 7.65175063861534e-05]], [[0.0005593742243945599, 0.010259552858769894]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0c9d8ffa18d20d6bc0c7b7fb2047413f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.02296661026775837, 0.030401743948459625]], [[0.01801164075732231, 0.007291101384907961]], [[0.009565435349941254, 0.20005014538764954]], [[0.09065292775630951, 0.005057369824498892]], [[0.0036611619871109724, 0.18642355501651764]], [[0.01378479041159153, 0.022222062572836876]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0658bccf0655ddfdcf8f79a71c37ad32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_55ddf9c6b4c8754cb94927239a065810(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23263609409332275, 0.11114535480737686, 0.009845714084804058, 0.05015848949551582, 0.16455012559890747, 0.024160051718354225, 0.16110481321811676, 0.14081424474716187, 0.1160120740532875, 0.15018466114997864, 0.22150282561779022, 0.23228415846824646, 0.1450996696949005, 0.0655934289097786, 0.23648183047771454, 0.1585187166929245], dtype='float32').reshape([16]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d50ec4cda7705fc6aa4100fb3ffe950f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7a8336d7f1c416ef378969f4b50c1e22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f33ce692a8dfe343d0dd1ab1a4948a09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_56cd1951efaed104b1d802639a1ff174(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0cc5a7577b13915081ca2c6cc9003cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_59fa89aeba43716e12ee77808c4dd270(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_59fa89aeba43716e12ee77808c4dd270(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_15b4fc96c4bbcca1ff279c3bcec675ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_469e2a72aa1c910cab02138545540e87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.011946693062782288, 0.22909292578697205, 0.15888167917728424, 0.015180230140686035], [0.23457637429237366, 0.1387505978345871, 0.23476874828338623, 0.039133816957473755], [0.3682340085506439, 0.2067251205444336, 0.04031139612197876, 0.008929014205932617], [0.2650768756866455, 0.06502872705459595, 0.3271848261356354, 0.06669780611991882], [0.0899767279624939, 0.10820017755031586, 0.05213071405887604, 0.06857089698314667]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_0d14d0bb64afe96837ef208b6f8c47d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_605cd7e21059331ca498fd25cb195d28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2601255178451538, 0.2813035547733307, 0.3010902404785156, 0.11394605040550232], [0.1276143491268158, 0.1411561220884323, 0.2907564043998718, 0.004007205367088318], [0.3767981231212616, 0.019813083112239838, 0.08903086185455322, 0.06486682593822479], [0.1276143491268158, 0.1411561220884323, 0.2907564043998718, 0.004007205367088318], [0.3767981231212616, 0.019813083112239838, 0.08903086185455322, 0.06486682593822479]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9db847cbef871c7761bc5b6008fa66be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f44c800411e46248c5ab294ad2a7609
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dd8cdd9c49218062af81528405dbcab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_76296ed7a7025b52155b64540ec4c196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_76296ed7a7025b52155b64540ec4c196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_c01148a3510bee0464c19fb79fe77291(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3336668610572815, 0.3895747661590576, 0.027006596326828003, 0.17651008069515228], [0.2637220025062561, 0.02544151246547699, 0.1283944547176361, 0.13914698362350464], [0.04708457738161087, 0.0057485103607177734, 0.3778526186943054, 0.16070452332496643], [0.2637220025062561, 0.02544151246547699, 0.1283944547176361, 0.13914698362350464], [0.04708457738161087, 0.0057485103607177734, 0.3778526186943054, 0.16070452332496643], [0.06639361381530762, 0.2864929735660553, 0.26026010513305664, 0.3718429505825043], [0.06639361381530762, 0.2864929735660553, 0.26026010513305664, 0.3718429505825043]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3457b8993b7ff51e693c67e3810f90c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f93e291a0a0563f6318d433d008a3e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bd224957295b1046aadaab7d36465e68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_0cc5a7577b13915081ca2c6cc9003cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f2f69d890c5026cd6424c645412c660d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f2f69d890c5026cd6424c645412c660d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_15b4fc96c4bbcca1ff279c3bcec675ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_470b14361cdfd02ed61680cb2b16ed7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_babe126e5acc72462ec74821e73a3e36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1467335969209671, 0.09901197254657745, 0.17062722146511078, 0.14475056529045105, 0.187338724732399, 0.11645021289587021, 0.18754443526268005, 0.029472628608345985, 0.20399020612239838, 0.044046949595212936, 0.2176395058631897, 0.011811324395239353, 0.2276376336812973, 0.15333817899227142, 0.0458989217877388, 0.1335170567035675, 0.18884292244911194, 0.2357255220413208, 0.03989668935537338, 0.10343722999095917, 0.1528700441122055, 0.17171022295951843, 0.10407722741365433, 0.17606297135353088], dtype='float32').reshape([24]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_37065dc4a8f2d31a5e1a9719c268dad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_204425993a34b16a25edb0f71d3b49a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_204425993a34b16a25edb0f71d3b49a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a04ba462c5ea746d0ad25c55362be7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f6813a4bae9adfa92c8519df502ad6e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.02470981515944004, 0.030571145936846733, 0.15698277950286865, 0.08978959918022156], dtype='float32').reshape([4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_53ca12e219a522a75d32587cb6c5b8ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1509297639131546, 0.1552731692790985, 0.056120842695236206, 0.15356037020683289], [0.07212063670158386, 0.3970109820365906, 0.12442433834075928, 0.38769468665122986], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079], [0.12647047638893127, 0.19000458717346191, 0.3279547691345215, 0.06161805987358093], [0.12647047638893127, 0.19000458717346191, 0.3279547691345215, 0.06161805987358093], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_417a728b336e9df05a4cfd2a33634af6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3127082586288452, 0.048425883054733276, 0.1581532061100006, 0.0485401451587677], [0.3431030213832855, 0.06575411558151245, 0.0661439597606659, 0.27380648255348206], [0.06341108679771423, 0.11628088355064392, 0.10398587584495544, 0.028025232255458832], [0.17723333835601807, 0.2572104036808014, 0.20969152450561523, 0.18656745553016663], [0.3127082586288452, 0.048425883054733276, 0.1581532061100006, 0.0485401451587677]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f8e2607bcbf9d2c58358ef94eb11743c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a04ba462c5ea746d0ad25c55362be7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bbc8bbfaaae1858694591b6fe27f9ee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14037862420082092, 0.3072340786457062, 0.233082577586174, 0.22903814911842346], [0.14534536004066467, 0.06035545468330383, 0.3983846604824066, 0.18683511018753052], [0.041945427656173706, 0.24708035588264465, 0.2164594829082489, 0.12575536966323853], [0.003619551658630371, 0.1542983204126358, 0.1320928931236267, 0.09877075254917145]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_754fed2f70aa3a5eafce174bc4d1acc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.sum(input_0, input_1, None, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c126851cb4a52059651574ef86ac8238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_754fed2f70aa3a5eafce174bc4d1acc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_470b14361cdfd02ed61680cb2b16ed7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3f6b6ec45b6f6f1adec1f2d47ed27344(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_fab3b559fedb220a6c8a2b996b8a03a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4fe8c4cbd19791505ae34c8c575dde43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_c8f45b4218d8271d9ded954b7dd0a168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5feef8bdb1d07239d1bc9c8bfcdf25e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5feef8bdb1d07239d1bc9c8bfcdf25e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e5fe07c8ea3b870bf4ced31786b6f6e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_754fed2f70aa3a5eafce174bc4d1acc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_de70a51510070a3a66e7e1013e4cec51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17318011820316315, 0.0040985941886901855, 0.10722425580024719, 0.24594347178936005], [0.17318011820316315, 0.0040985941886901855, 0.10722425580024719, 0.24594347178936005], [0.2724611163139343, 0.14107611775398254, 0.3538140058517456, 0.029049724340438843], [0.0411318838596344, 0.2469097077846527, 0.05615696310997009, 0.09188510477542877], [0.09708136320114136, 0.41709190607070923, 0.1974916011095047, 0.15154391527175903], [0.022501900792121887, 0.2542913854122162, 0.010012298822402954, 0.1757974624633789], [0.01827526092529297, 0.11807702481746674, 0.04941102862358093, 0.02130529098212719]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_c704ff20cf334773d69e1fe9e0551537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bcca8e9eda77f2f94f4451ecd8ed9aba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bcca8e9eda77f2f94f4451ecd8ed9aba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_11a8023583ebe6720f6bd87841c6e2ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([4909], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_1fbe810471aecd303ac8532954657a72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([1242], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_17e378482dfb8dfea6625685e7d9f5f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f44c800411e46248c5ab294ad2a7609
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3e39b0e78205354e9ad07446c4f4f263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5a3aa0c384a597b94ac83b97d77d7524(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5a3aa0c384a597b94ac83b97d77d7524(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_6043da2ff42ff48418b060e66046e23b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08719142526388168, 0.40703436732292175, 0.14703017473220825, 0.20497220754623413], [0.21442238986492157, 0.18900448083877563, 0.22183440625667572, 0.010784268379211426], [0.21442238986492157, 0.18900448083877563, 0.22183440625667572, 0.010784268379211426], [0.2041129320859909, 0.1454058587551117, 0.39465102553367615, 0.02922683209180832], [0.02670140564441681, 0.1606736183166504, 0.18615491688251495, 0.40934592485427856], [0.3336673974990845, 0.08125244081020355, 0.2736116051673889, 0.33980047702789307]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dc7010f1952ae29f58ceb9df3aaa9d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_121c97acd5bb5015337e1d111daf0125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_60ee61f1bce7bde445ad595c8a1ea2c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_195dfff932a78f77b0e0d5458aafdf63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_839ce26115c90c223a41d39ccf85b4c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_839ce26115c90c223a41d39ccf85b4c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_097597739e1363c15d52fbf033aa2fde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b217a7c983c6ed1adbdce8a067904024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b217a7c983c6ed1adbdce8a067904024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8af3d5279b78e0b2367c7631e8e0cc67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1b8dde04add7e84d7c658db6cc79bfbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_1b8dde04add7e84d7c658db6cc79bfbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9298619f4ded9d6f087045ba05dd25be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7484cfd2b20d4fe43c0bd3a213dcc89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e8da34dc8b39359ce8dcd4f186d1578a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f93e291a0a0563f6318d433d008a3e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a74301a17592c23ef7f9afe8a39d8063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0342906229197979, 0.2616975009441376, 0.11236084252595901, 0.23144856095314026, 0.1474074274301529, 0.14624141156673431, 0.11792205274105072, 0.0706254094839096, 0.16403816640377045, 0.1672552078962326, 0.010639270767569542, 0.04043728858232498, 0.19575585424900055, 0.12154953181743622, 0.2397426813840866, 0.13022302091121674, 0.016859127208590508, 0.00764064583927393, 0.11539359390735626, 0.258635014295578], dtype='float32').reshape([20]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_38f5037a6a3142b6cb89266febae2dd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([17457], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7484cfd2b20d4fe43c0bd3a213dcc89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6af6394c2f7d3c64dea122d9c6ca9cf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8f9179bb084f279eace036a308aae8e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b175020cb17eda1819a96b0975ffe213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_142cac589a2deb0182fdcd4abc4d6c8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_142cac589a2deb0182fdcd4abc4d6c8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_098b990a49e58d2f4a11cacc5eeabb30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e47361bbc6bdcaef6b8b58be86c71fee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_fdc5cb030b080823281f899282fb4fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05997924134135246, 0.17256204783916473, 0.29755768179893494, 0.03335516154766083], [0.43779996037483215, 0.19319099187850952, 0.01612231135368347, 0.18234547972679138], [0.17438164353370667, 0.24513502418994904, 0.027111470699310303, 0.13506805896759033], [0.17438164353370667, 0.24513502418994904, 0.027111470699310303, 0.13506805896759033], [0.13616040349006653, 0.334640234708786, 0.23136883974075317, 0.01873648166656494]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_121c97acd5bb5015337e1d111daf0125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ab8be978244be906c6ccd93b307aa8dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_02254a92124454906188587ca3ae5831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_26a0b39748adc5fcf982520a9ec38aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_be7add0de564f1e428d13208ba3eb4b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fb9f5942c07fc52d8343e205f5fca42e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_fb9f5942c07fc52d8343e205f5fca42e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bd209a19653f21f2150496623d84ff5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3046402335166931, 0.07770112156867981, 0.14432917535305023, 0.180344358086586], [0.03918623924255371, 0.019960127770900726, 0.028510063886642456, 0.033728450536727905], [0.017332687973976135, 0.28235897421836853, 0.3564165532588959, 0.1784440129995346], [0.3046402335166931, 0.07770112156867981, 0.14432917535305023, 0.180344358086586], [0.055790454149246216, 0.36793333292007446, 0.027703553438186646, 0.24950526654720306], [0.23903867602348328, 0.20549030601978302, 0.09378381073474884, 0.2969999313354492], [0.055790454149246216, 0.36793333292007446, 0.027703553438186646, 0.24950526654720306]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_907f63a2d245f9f81f590c3b8416252a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e8da34dc8b39359ce8dcd4f186d1578a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()