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
    class PrimitiveOp_43760c6900be8f9c836e8c54f452bdff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0cfcbf6c0f1d63b021767eb48cf1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 91, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2fc713a82791c7092ddeafd9bca90dc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4736008b95c164966350877da3d77c99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5a03a4153caba4e8065f4962ad6483e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4736008b95c164966350877da3d77c99
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fed6c674ba976a965e19c2b25c512eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([512, 256, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e6fea8c782fad3cb3f030dfe53853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_38df529832c64ad8ce9c3e2194c34100(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 68, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1221eb39499c0abfa4e08a0677c7e337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17b7ca82a377cc539d007dadd9137cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d3f484dd6c81bae93aac363349df2a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6e65858e1d6d923d14dd9d3005d119c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03777ad25cff3134e9ad14c71fd782d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9190d6b9e4a6b8094bfc0b7adf80344(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f37e94bb5555b81a56226884b74c9c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1d5d011434ec39d6960787dda225967(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb7dfa000241af2df04c594c0ae9c975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae5960e5aeb76e2e01b004aeb849f259(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ad0f45db3c06583b342a2c96d2d32c57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f795c3f665969c7ee0ba44a3629db6c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad0f45db3c06583b342a2c96d2d32c57
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_826922103c99652a4c2746f79a883e10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3bfced70fe56d873b878bbb6508bb0f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_826922103c99652a4c2746f79a883e10
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int64'),
            ]


    class TestPrimitiveOp_39ccf5ff2748f98845633bd801dc5d4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b280c7f15f035587f368555ea84c6e06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_82164e9351448e5040dcc655b172fc84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e237691cf03929f717d60ed521ec273c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82164e9351448e5040dcc655b172fc84
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc8477a1288fa42193e610b7ec2033b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ceaf62ea929999b251fc7e527f61b071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01bd77eb14c16a866b7d98e69729829f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e111bb02964e938629c39ba9833be79b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 76, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84c8c6dc39f75f96a274de8b4ecf9be9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e111bb02964e938629c39ba9833be79b
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_53da1ed95998e3a46f02073de3c7049a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_79ba8849e1c822e59b7aa4dfac11cfd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53da1ed95998e3a46f02073de3c7049a
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0347567a8d212a0723ebd8be5e63d3d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8c0d069e5d7e0510df3bfdcab51bf7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0347567a8d212a0723ebd8be5e63d3d2
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5956700776e3fdc696b34167139a4190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a5b81f8d9ade035b3cc3e4170dfc6dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69fd5d46da99182359ed8881d84c1846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cd4c532119c76152a90b26da85d0478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d96fc239263ece0ed21b2c7266e03800(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 15, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4a30697edc1bbaedfc3dd878ee0f24d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d96fc239263ece0ed21b2c7266e03800
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_855b2ea49b4b55714ccfddc3dc06638d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8fe5c47670cb1c24709d67dcbc9fe1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e111bb02964e938629c39ba9833be79b
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f68e9b228aece7bedbbb7959cee508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c551c58a600e4cfa9a48e45ccef4c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da24e5725da52c665838ad4d4eebcb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0a4e5cd36fa4fb473ee8949012eb9d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04ebc2dbc2e2b39789e6d8d1ca53f0fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f36bf57b681cfb1e1a86211d8d24cda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcd7bf53fbfdfa63e815e2425a53c073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f6689e6bd39438d86857e1ba4134a932(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c8c3a6578569aa81f052b9486f88f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6689e6bd39438d86857e1ba4134a932
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03777ad25cff3134e9ad14c71fd782d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a69c023bc6bbe2dc3c7008241674235c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7d9a90a7949ef2ba29eecb60560539a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a118750dde0caa384223fc59ef30acc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82164e9351448e5040dcc655b172fc84
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f290a07cf18746bade329586a5a3cf29(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_821307a961c984c35ce92c8d934fe256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f290a07cf18746bade329586a5a3cf29
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b49450105cd5966efdeade963e61ca8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72b659123705187d90cb9e3c68e8e67a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_379412f33f07c8b23a302ae9e268d748(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3077656951c7b258884be5d09b0ead8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_451d21826fbfa2d1c6d841dc1a088038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3077656951c7b258884be5d09b0ead8c
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b784a0a7a8fbb987e2cd6ff14a40d4fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d96fc239263ece0ed21b2c7266e03800
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c83c637b329d2751ce5414a9d58d935(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d96fc239263ece0ed21b2c7266e03800
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af4f345eed4a768f6122a6e168a2570c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4736008b95c164966350877da3d77c99
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_855b2ea49b4b55714ccfddc3dc06638d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00aa6535221ef57a057c16d505105b15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeff61576b855f546f2e9e00386b5575(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([390, 64, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d34e63a400f28e20ba29101d1226acf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_127ff3ed115e8327e549fc29fa88b993(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b00a90a24be22e14ff3c2885ac14631f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f290a07cf18746bade329586a5a3cf29
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7027045763223edda36bc03a76fd11b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a05e0101336bdb7dc581110af98ec9df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7027045763223edda36bc03a76fd11b4
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df2c44a9a78e8972e897452904377e69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_440034671029a35da52f792143d2aa2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df2c44a9a78e8972e897452904377e69
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b27845b37aa3ce4712861d8147279802(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5369f91c0d19046f8b119534b14da67c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b27845b37aa3ce4712861d8147279802
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc3ede08be863eaded1bce82b0d4275e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 200, 304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b1fe8be7898e693981e4e79916b75b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3ede08be863eaded1bce82b0d4275e
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2a09a7638229ac2962d9f7cee2cb9171(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 5, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55ae49878084a57b8f068dd5751b2a1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a09a7638229ac2962d9f7cee2cb9171
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb7836eba28923e7ae491649b2b8e199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5761fa26f17ebb8e2053f16930387de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a96bb41ddd20017de47b8c311139cf17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f50d7867584b140f4628edb90d57954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f37e94bb5555b81a56226884b74c9c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1d5d011434ec39d6960787dda225967(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b63a77250045dfdb4451f12d687933d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ecbc3fd9fc225af2ea55a052f98d74c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0a4e5cd36fa4fb473ee8949012eb9d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b98802d86e15b7321c2e8a9aca025210(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_662b7f314879b2b869f51b673e2be4c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f60b29e530992e41380fd844d4561f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1280, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55fca90f2089916e69be69be36145b52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f60b29e530992e41380fd844d4561f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a91e82aa598cafdf014d8a410cd756d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20d5a4a1d15f56315b96ff5ce20471ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a91e82aa598cafdf014d8a410cd756d
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_954439826c9b092b9a3dcc65d8f70ebd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69bc1884471d7c965623bee1f49d105f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c062939bb02e6110d7e37d189f90714(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99f48969e8acbe782761ec106a97c5be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53da1ed95998e3a46f02073de3c7049a
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee348f6a3db7e7a894238635287892eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b8ec3c92cbc1809951d9ef75ef55dcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55dd5c35f32da228398b347e599a362d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6a7620438f038770f26353aa51cd8e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c453bf0581a2bed48f7882428d128360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fe3f7d8b1fbbfddd4c33c3f8003ea88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b27845b37aa3ce4712861d8147279802
        def get_inputs(self):
            return [
                paddle.uniform([6, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbfb33d6e6b303e4c765a3e96c2e9fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc96b28f216be4bb85e537c221b9a1ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc8477a1288fa42193e610b7ec2033b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ceaf62ea929999b251fc7e527f61b071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9854037eb216b611ffdbf487c1f3973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f50d7867584b140f4628edb90d57954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acdd8b33472a9b26d5a21fbfeccb3dd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1d5f3aed49768232c59c865b524430d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee348f6a3db7e7a894238635287892eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fa86f4214ca5f0b739992745287b1e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ec5b6729b54e9029c24dfd50d1403aeb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 192, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8bc536d2ace4e3c128d791f3197f1b3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec5b6729b54e9029c24dfd50d1403aeb
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_93b5b2f1c934caddf459c35c039f8bb3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 384, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_31dde02e378172d7737bd1b9b2efbf71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93b5b2f1c934caddf459c35c039f8bb3
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_30e0ebdfbc0353ba92655536173ab655(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8443c334d3c5ee9fb77baa4a773b9c3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30e0ebdfbc0353ba92655536173ab655
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e89c798b48a447df693dbab22a4d1bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad0f45db3c06583b342a2c96d2d32c57
        def get_inputs(self):
            return [
                paddle.to_tensor([[[6], [6]]], dtype='int32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_1c58c14bd7e8607fed84d2d901f87a31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_826922103c99652a4c2746f79a883e10
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int64'),
            ]


    class TestPrimitiveOp_4c8c3a6578569aa81f052b9486f88f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6689e6bd39438d86857e1ba4134a932
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7c3a8c42b2fa567a7a97563055867c9f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff424a815229b3e4d71906a42f3e46c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c3a8c42b2fa567a7a97563055867c9f
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 1, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf42ebff5129726f02e5425743bc1fa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b809712ad679f7f0bf3ef68b02d37159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cfcbf6c0f1d63b021767eb48cf1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fc713a82791c7092ddeafd9bca90dc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c8c3a6578569aa81f052b9486f88f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6689e6bd39438d86857e1ba4134a932
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_975301955e83b98caa7d9fb9055ad636(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b27845b37aa3ce4712861d8147279802
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c54b4fb15635782e6335753965ad7034(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 2, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f2c9f7782d197c40ee2ed3ba858803c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c54b4fb15635782e6335753965ad7034
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cfcbf6c0f1d63b021767eb48cf1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fc713a82791c7092ddeafd9bca90dc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ffd0e5d11586ba954b8a506ac13efaa3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e630c792c22e9c67992fb0b0e1dc1d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ffd0e5d11586ba954b8a506ac13efaa3
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f795c3f665969c7ee0ba44a3629db6c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad0f45db3c06583b342a2c96d2d32c57
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_3f750bd319e1fcf38e05073f69181fb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_826922103c99652a4c2746f79a883e10
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int64'),
            ]


    class TestPrimitiveOp_3c3928dd11766220ed802933f53d8342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efa40503ff7d89160966c8f736dd189b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34949560ebe5e6c7be8be731e68c20ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7de5acbc1f023c033bad13d40fb9e52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa652d2959f5ffab70abcfdbdc672ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3077656951c7b258884be5d09b0ead8c
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_00ee16391ea0b5a1ed822968397a9813(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, 4, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e79a4670dbe633579316dde0638b800d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00ee16391ea0b5a1ed822968397a9813
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8c15bb5ede0bfe14a281c15d40f5b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc96b28f216be4bb85e537c221b9a1ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b078562859302d5e0f1b58a06ce247d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53da1ed95998e3a46f02073de3c7049a
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93c558f3af77f6a1acfb961521386a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce806809df692103e29b7ebc1f3a6eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_310217f2882bba0e130cf7c8d963fb7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69fd5d46da99182359ed8881d84c1846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a82b3adff05cedb9093fd8e00e8c8058(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e111bb02964e938629c39ba9833be79b
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_859ecce422762df2450f81933fb9ad7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0347567a8d212a0723ebd8be5e63d3d2
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_975e35bda9366e1a146f2d429047860f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a91e82aa598cafdf014d8a410cd756d
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb2c1f232359a9fd4e9db8b5eed3dcc6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, 4, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1b4a181b637e6639f792502c07962a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb2c1f232359a9fd4e9db8b5eed3dcc6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a13aa4810f6082920bdad16a196fb6b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d96fc239263ece0ed21b2c7266e03800
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c3928dd11766220ed802933f53d8342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b06a7d56cf3fa6506802f2a2c45b74d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_407ccfc74d2939d41b4eadfc61ae571c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_855b2ea49b4b55714ccfddc3dc06638d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00aa6535221ef57a057c16d505105b15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d01e28ae2a5368dfa61836cc8fd8351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df2c44a9a78e8972e897452904377e69
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01bd77eb14c16a866b7d98e69729829f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02102573cd5c217b337d25ad224a77ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f32b97ec478a4b68a5c06729e05b25b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ee8e61f505bf99330e0797da5f3cb83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc8477a1288fa42193e610b7ec2033b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ceaf62ea929999b251fc7e527f61b071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b30455a1e5e14ca0b9bc54bf2ca8a58(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d415c8096ab1b92fe50c5304da38f368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b30455a1e5e14ca0b9bc54bf2ca8a58
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f68e9b228aece7bedbbb7959cee508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c551c58a600e4cfa9a48e45ccef4c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da24e5725da52c665838ad4d4eebcb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f68e9b228aece7bedbbb7959cee508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c551c58a600e4cfa9a48e45ccef4c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da24e5725da52c665838ad4d4eebcb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_96b36fbd1e3084fce335d43cac3f6d2c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af74f213b00c37b095e2c12963263800(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96b36fbd1e3084fce335d43cac3f6d2c
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af74f213b00c37b095e2c12963263800(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96b36fbd1e3084fce335d43cac3f6d2c
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb7836eba28923e7ae491649b2b8e199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dac08840befc0f8a4de8d61eb5bc0917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4fca373dea2427c57ab8d73bf8172c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5969be8591fd38fdf27347db2a60084d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 384, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2d5d62cc40f0ed898f7d5ac2f8240e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5969be8591fd38fdf27347db2a60084d
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a56831fce47c113cb417ac055016852(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a09a7638229ac2962d9f7cee2cb9171
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259943d386e66d666c09c5bc1b8e5788(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5fa80d79edd09232bf3c46fd5b91cd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cfcbf6c0f1d63b021767eb48cf1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fc713a82791c7092ddeafd9bca90dc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26c6685e7e18951dd6cf650102e1e2a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61fb50400596bbecb7dd2698e1dc7345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d1153b998e1c534566682dcd0d493498(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 100, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49d72c121ab18178506d29ce50123869(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1153b998e1c534566682dcd0d493498
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a746f444ad6098fbd58e2186aaac01bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7027045763223edda36bc03a76fd11b4
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_813b3d52ae9841f075800844ce200021(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 2, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db9fa5c58a728d5b060b6b061d24bc4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_813b3d52ae9841f075800844ce200021
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2da78b9f6c64166a163c0019f083cf23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b27845b37aa3ce4712861d8147279802
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db6cf457d6ad5f485fab9e0ae0033ddd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 136, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95fe7479d0011505f7e71b9dbc1f6de3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db6cf457d6ad5f485fab9e0ae0033ddd
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c8c3a6578569aa81f052b9486f88f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6689e6bd39438d86857e1ba4134a932
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9d5307be8d77c8b1b0c50d5ee5ab29b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 300, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be543fe97430afbdc6fb444d1454cfa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d5307be8d77c8b1b0c50d5ee5ab29b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01bd77eb14c16a866b7d98e69729829f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02102573cd5c217b337d25ad224a77ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb08cbecc134ab2bb6b53327b4818145(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d712b047b6e0af7a137ce74adcf361bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dddd26ff04231a732d0e3acde1488e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67c92f8b1aff2fff6858657cb0814da7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16c8f2450fc11f7f8310fef66437f534(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5761fa26f17ebb8e2053f16930387de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66c36e7b9ee67dc336160339aa5b6353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e11b5a0b3cac291785897cbd8a7415cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a42ddf2a716490b8b8076d8bc38f3cb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce634e73e2d1c0b72b58094e5e12b522(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a09a7638229ac2962d9f7cee2cb9171
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97caadc422a3b4564cbbd18a38a40b63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06a49b9d23331d14fd74bead097e82ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07f04fc927aea790d0de8816e27bf9a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 768, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ae20eb0d0e66efa6832e195d3975283(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07f04fc927aea790d0de8816e27bf9a4
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77589eda3ea3a056dd24125046411c8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b27845b37aa3ce4712861d8147279802
        def get_inputs(self):
            return [
                paddle.uniform([4, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_781c1ee04d3ce9bcb6f67554b5475690(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_127ff3ed115e8327e549fc29fa88b993(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_108ab8ceee33747d9f76cc63e0df14a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df2c44a9a78e8972e897452904377e69
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03777ad25cff3134e9ad14c71fd782d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9190d6b9e4a6b8094bfc0b7adf80344(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_297e2b653483a0413e3294bdc8dc6561(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a09a7638229ac2962d9f7cee2cb9171
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_870c0327b77bc8819a23f0d22ad23aa3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2881e2c553b38aa9000e231119a30dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_870c0327b77bc8819a23f0d22ad23aa3
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eec49a15c3a7749a70d8f78eb356328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f60b29e530992e41380fd844d4561f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_94b6032eaf895cce3135bf2e15190db6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 768, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_476cdd38f05278ef2a327ad630993973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94b6032eaf895cce3135bf2e15190db6
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69fd5d46da99182359ed8881d84c1846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cd4c532119c76152a90b26da85d0478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca1de11730fa8c7508768ca409564066(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d96fc239263ece0ed21b2c7266e03800
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d34e63a400f28e20ba29101d1226acf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f91c7f6a96110f05290056c9a320fe82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb1024008cd7877cc5e8f838cdf2f42e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3995d76548b8d647bd721e5f677308d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bada024690738a58c83a71903bcc6793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f68e9b228aece7bedbbb7959cee508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c551c58a600e4cfa9a48e45ccef4c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da24e5725da52c665838ad4d4eebcb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc8477a1288fa42193e610b7ec2033b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ceaf62ea929999b251fc7e527f61b071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b98a21e63904797b3f99e59f0bf39f26
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75b68756f3c68518518d0d6fd7456026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2161b9724d01d8b153e9c5d1170e1ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61a9456bd68199a37c39438a304f4fb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2161b9724d01d8b153e9c5d1170e1ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af7c1ca63d95b6d5a327425a723658f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_870c0327b77bc8819a23f0d22ad23aa3
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4eba9554b533404887207f7432c2321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6f2353f800bb3a7674ebe0c8fd5c05c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ce89cd09fb2749c78516f7f7328a6fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30e0ebdfbc0353ba92655536173ab655
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab39e5169d88d75841fd987cbb711e2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4672561f15a646292a982a6d6506ee2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2293ca93d51d47b2f1753c921fa252fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 192, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf729a3d203db01f039b52841507673f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2293ca93d51d47b2f1753c921fa252fb
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90b0c1984f230a25bc70a862be95f081(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f90e5a156f23e3c88b6cbdd273d33f7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9cb22a1325fc3bd6ba171c0831cfa13d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed9704902f3eddd76efce3d8f4616978(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cb22a1325fc3bd6ba171c0831cfa13d
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f57deb70457db887be6fd0190c2421cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cb22a1325fc3bd6ba171c0831cfa13d
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89dcb7a4eb7f15bbf3e31dd7c962484e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a09a7638229ac2962d9f7cee2cb9171
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd0a846f7b5b0fd7e40dd9d1d598e1eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b30455a1e5e14ca0b9bc54bf2ca8a58
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb08cbecc134ab2bb6b53327b4818145(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd55fb2bf67e694832f279885325cba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df529832c64ad8ce9c3e2194c34100
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f37e94bb5555b81a56226884b74c9c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e882a196861549f9fe27c6571d6cef3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f39b784a5471a4f0623e023ece630a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f3774104b97234935136e87d9c87c67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0778166a986d041a070803fe015df277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b27845b37aa3ce4712861d8147279802
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_96956a225c1814da05a053dc79806267(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c61e05e9e8d80a62f31f00dee892da21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96956a225c1814da05a053dc79806267
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0dd53a2d7a6466d8e8dedcb773b93bf3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9987aa74682a656c01810fdf818a1b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dd53a2d7a6466d8e8dedcb773b93bf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f85ac4e72a94774cf34be0ab12364746(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3af88800ec5e99b340292a4ba80b358e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85ac4e72a94774cf34be0ab12364746
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d5c014587e071eb4e58243296cfb60c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 91, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8c06ddf682c429c6e765864ddd582c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5c014587e071eb4e58243296cfb60c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac272eceb9d1f5d2fc0655c24daeed99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 128, 16, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c65dd62fbe372c3789d2ec16c94d7caa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac272eceb9d1f5d2fc0655c24daeed99
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c689fc15e8375084c30f3ec978611f7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[512, 256, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7dd6c3893da93fcb7eb6122a3020f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c689fc15e8375084c30f3ec978611f7e
        def get_inputs(self):
            return [
                paddle.uniform([512, 256, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d12c12994f3861a335a9cade2b9a1283(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 84, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7b8bf0994114de7f989661a78386cd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d12c12994f3861a335a9cade2b9a1283
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bced148cb56875e7a79c3a88623988d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 84, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fdda7897aabb8a6e8a62a8b388dc9cf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bced148cb56875e7a79c3a88623988d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f86605b444a070bd6b6ac810ec1b5312(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e1f5c451c7e5a6b1b400ddadf8c8f1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f86605b444a070bd6b6ac810ec1b5312
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c916b6ee4121e5558b66154758794c32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41753ee8d49e4cccf552853dd6b38c6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c916b6ee4121e5558b66154758794c32
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_33c642c5d31e06fe64b06444c30fcf26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc55e41dc7ad2019bf3658b3c7fe87a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33c642c5d31e06fe64b06444c30fcf26
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0dcc2063ba0a2d166bc4c361ac32e53f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78926b2569104a9601782f9823aa7242(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dcc2063ba0a2d166bc4c361ac32e53f
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4511c4b340aefd462a971d38be21bd7d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f92096bdb5af50f5a8ffb79df9ba2ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4511c4b340aefd462a971d38be21bd7d
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c35ed30e4601e082b32d3ce47c31f1b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 48, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_992b2a1928e1ce27be08297da926bfad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c35ed30e4601e082b32d3ce47c31f1b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7b27b5eab1e3e362427526738a20b6d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 48, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6dbb409d26c9b64bc683ac1cd8299ec6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b27b5eab1e3e362427526738a20b6d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87fe8945e355165ebf5f8177ee772479(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 15, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdc74add939ec944aaf43569551b4388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87fe8945e355165ebf5f8177ee772479
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_babfaf546fa8fd4f9d9593ea80fbf6e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 15, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bffd5e18413bd4d7c6131fffb9a9c54f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_babfaf546fa8fd4f9d9593ea80fbf6e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_936d03c83e3366603e601304acde95d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3a2b5138bf31149bfc2741050fc2e70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_936d03c83e3366603e601304acde95d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_07b2d155ad144134654301d473e00a60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af01275ea668020973b94bea11fabfad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07b2d155ad144134654301d473e00a60
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int64'),
            ]


    
    class PrimitiveOp_b89888ac68f0a72748bf92ed24ed6604(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95e823ddbfe32e753ac51635456de590(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b89888ac68f0a72748bf92ed24ed6604
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_88cc419c6970ce695a61567b5b4eefd1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c89158403c8c39c1b59386515af3e1cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88cc419c6970ce695a61567b5b4eefd1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_097c2a4587061d4c04e5c12e3404f3fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11b3f3d276bfc9367416d5b1938e607a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_097c2a4587061d4c04e5c12e3404f3fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c8a5a3db1017e7cc943279f306c75760(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[128, 320, 8, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1af31ac90651b90819a25960adae5e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8a5a3db1017e7cc943279f306c75760
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b9a29474af273a2bbdf2856d84a69b9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e4cb607701d8b35c51df6631e639c6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a29474af273a2bbdf2856d84a69b9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c89158403c8c39c1b59386515af3e1cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88cc419c6970ce695a61567b5b4eefd1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3140b5b601ce08c1a674e6fd8393a6d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 91, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d563b1f8cc5391ae1126074689771a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3140b5b601ce08c1a674e6fd8393a6d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_100036f79ba97371d42ee04c62561c21(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2555401fe7f4ecad4d01cfdda6127928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_100036f79ba97371d42ee04c62561c21
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dcee41f53d26d81e54712849c0da3ed0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 76, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc224ac5da8ef46fa50934b1f1246501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcee41f53d26d81e54712849c0da3ed0
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4de75b07afb231ef8c2f7d155f09b32a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 2048, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a633558e39713a8fa0d789b76902ed29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4de75b07afb231ef8c2f7d155f09b32a
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e3fad68a2fdf315a27e06864d75c51be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1000, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19458f7d6b5011a092f23cae7bfd44af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3fad68a2fdf315a27e06864d75c51be
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2322a6b8314dbaa6d98b424ad00fa333(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 30, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27a64d5016e5a06d84cf7f5f94411ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2322a6b8314dbaa6d98b424ad00fa333
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_50165a86179d8029000e8b38942c3e44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 30, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_693d92af942fe66c7564fd2a959db0d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50165a86179d8029000e8b38942c3e44
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c61add3398336db087aa4ec8682512d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 52, 52], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce4df54d56943600acc8be1d460f1820(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c61add3398336db087aa4ec8682512d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a10d2538167cded3acf2fbc97b295a8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 52, 52], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70fea07aa574ff1abe0b33f719d99116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a10d2538167cded3acf2fbc97b295a8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d3009a133e0c140b231248480338849(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 8, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf35b652595be41770980fd28c13a2c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d3009a133e0c140b231248480338849
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3264f4ff24c5bc0d0e1b141f8a8b3c9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 13, 13], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce307f3931b899a3a40565e0b0fd3720(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3264f4ff24c5bc0d0e1b141f8a8b3c9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0692afaab27b4af51dc2f76a13e2f4cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 76, 13, 13], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_281ee829b15ea96e82be00e7c992af78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0692afaab27b4af51dc2f76a13e2f4cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dfa51ab488f0a113c147ba4320529d06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0f7e1ef75c913f462dc738325a0d9a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfa51ab488f0a113c147ba4320529d06
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_686c9b70cc5d9d2d318500fb27d333c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb481e499769a763c0b55f38b3a77825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_686c9b70cc5d9d2d318500fb27d333c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe8ae371ea886ee6c6efbadd0412887f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 91, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ed861d29a83a2ee15765bd31cd05c63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8ae371ea886ee6c6efbadd0412887f
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_643fe13cdefffa117a2f969978c1eb89(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 34, 34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21057cc4d7ce6dce16d051f304690127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_643fe13cdefffa117a2f969978c1eb89
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a5bcd7ae45e3c9459d93d3b54fb47d0f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 34, 34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e4f598f2e76741e3264ea73fa85d778(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5bcd7ae45e3c9459d93d3b54fb47d0f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1821002ee09a4e267ffafbccb94ed4dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 76, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8ca4c6d5252df8f0ab176456ffa940d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1821002ee09a4e267ffafbccb94ed4dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f04b7d60f32a2cf2e8463e9ecfacb0ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 76, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ae52fb0e099e50113179b0def1e3d86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f04b7d60f32a2cf2e8463e9ecfacb0ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_405384f88b44804a0bcab13226be2b8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_516eeb6465dddad2887dc32092a6857e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_405384f88b44804a0bcab13226be2b8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78926b2569104a9601782f9823aa7242(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dcc2063ba0a2d166bc4c361ac32e53f
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1dba7313b834597e00a00938b70c716d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_427942ddcb0b7ef6d7b53ac1cc0dbeb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dba7313b834597e00a00938b70c716d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c2bf8a482894215ac1d7231ca8222e9f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b22e5c0a6a3f8e5f20d20d69a7611a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2bf8a482894215ac1d7231ca8222e9f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81ea55ad018754efe872e3759a68d2fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 320, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b43be78402785c39e430433e6832e257(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81ea55ad018754efe872e3759a68d2fe
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15fdb09ebac15a64da9d374f8e1fe110(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 160, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06b871768a72b6be8eaba93decb34c19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15fdb09ebac15a64da9d374f8e1fe110
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f69c0552b8a8ddeaafd6cc4f26ca1a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8efd6f69ef211bb13d873b431f30c96b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f69c0552b8a8ddeaafd6cc4f26ca1a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8d930bf83f9e3221d624348bcc78a56c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c51629827ad74fdcae338127a53c5b00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d930bf83f9e3221d624348bcc78a56c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf9237f6958031fb695eaa33aa3eb74b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8bb4aeba5f76dcc4e6cee3609e6786c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf9237f6958031fb695eaa33aa3eb74b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_375e547f9151220f86ee63faefcaa3b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 64, 32, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c25a134bec8ce70950a814563c09ae4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_375e547f9151220f86ee63faefcaa3b7
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a204fca1becb6f083c59646a6b1e76a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 15, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_668e0fa6971e4d6c19a00d93232dc77b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a204fca1becb6f083c59646a6b1e76a7
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4cb607701d8b35c51df6631e639c6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a29474af273a2bbdf2856d84a69b9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fecf4738fc0d98fed18379cc3570c7a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 128, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7fd62f8665f6599a7e73d9d7306eda49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fecf4738fc0d98fed18379cc3570c7a9
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce307f3931b899a3a40565e0b0fd3720(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3264f4ff24c5bc0d0e1b141f8a8b3c9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_350848abe95f0b81533bba46e2896182(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 13, 13], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db00d1214490196bb1299dcad6dbcfd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_350848abe95f0b81533bba46e2896182
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_746e4abf7beb4f9b20c5a5f8d498785a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[390, 64, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1977773824c7a5f935e4bf9af002d44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_746e4abf7beb4f9b20c5a5f8d498785a
        def get_inputs(self):
            return [
                paddle.uniform([390, 64, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_31b853eea9510f97a8c6ac2484fbf58b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_58f4ed89555f3c15a204e5cceae57f5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31b853eea9510f97a8c6ac2484fbf58b
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c020c67ad2d0a2a00be24755fefd71e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62416853549b8459cdf148611efcfbf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c020c67ad2d0a2a00be24755fefd71e
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc420f118fc82dee9fe04461c3f17ba8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 160, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbe314df561d8013c7e972e015005281(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc420f118fc82dee9fe04461c3f17ba8
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4ab99bc71bf63bfe2b4215179a2f5faa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 768, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af22c96c0e64eb6604f40c7fae156373(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ab99bc71bf63bfe2b4215179a2f5faa
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_661d916d78858c8706316513c5f46209(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 768, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b005eb4a091caff68a7e71b0e32979f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_661d916d78858c8706316513c5f46209
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d452595e3b78f46314e191f0739ec28b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 200, 304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76b80286f06ab1952a74f36eb96a3ec0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d452595e3b78f46314e191f0739ec28b
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76b80286f06ab1952a74f36eb96a3ec0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d452595e3b78f46314e191f0739ec28b
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dddef1c8fda6af05b910be609f35441b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 8, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04c9ef8cbe314099ed55b9fb6cbab4df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dddef1c8fda6af05b910be609f35441b
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0507c0a661bfa810b48ea732eb1d47f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f546edb6d7896166722b4edc40e6d27e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0507c0a661bfa810b48ea732eb1d47f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_90416868424ea14fa009663a544c1627(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_274892a11e7d8d8f283ee5ef8482f918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90416868424ea14fa009663a544c1627
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d7e4bbc907e37c5b8235937891176f25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53947a7b0f67808106a832353ae38c67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7e4bbc907e37c5b8235937891176f25
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e89f2d2c3b1aa899b06ec8f07238e157(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c00cbd1cb7edeec046d0a6bde3738d62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e89f2d2c3b1aa899b06ec8f07238e157
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_992b2a1928e1ce27be08297da926bfad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c35ed30e4601e082b32d3ce47c31f1b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dbb409d26c9b64bc683ac1cd8299ec6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b27b5eab1e3e362427526738a20b6d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf3067310fc88192be1670269fa7b0ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 21, 21], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a7be464ae5264b31e198b02cfb7b2b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf3067310fc88192be1670269fa7b0ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aed12683d5d0b56a9435839e79bad628(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 21, 21], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_820c0e60ab9ab9882b4c563dea3484e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed12683d5d0b56a9435839e79bad628
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21057cc4d7ce6dce16d051f304690127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_643fe13cdefffa117a2f969978c1eb89
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6f40df0d7d3285d24e6e855756a8c9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 34, 34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b35806d0ab2714f554bb1e11aff7aefc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f40df0d7d3285d24e6e855756a8c9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7cefb64c33faeca8e1c585bb48cb3d03(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 34, 34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6177b3bf92c4430e234dc3e9f9e490c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7cefb64c33faeca8e1c585bb48cb3d03
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_836fae88924f863ca255c02c205c1a04(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1280, 32, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e925a731ec04912bf10ced656cc061c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_836fae88924f863ca255c02c205c1a04
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e9503f15a67b2e4fa8e794414921cd80(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c9037ad959317ccc92937ce04c3dbccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9503f15a67b2e4fa8e794414921cd80
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da3c61deba5ad7e268a812979f245e1c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 18, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_728869fbef36a2db53c6fdbcf7a690ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da3c61deba5ad7e268a812979f245e1c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a46bf58648b2bf2b4bf0559d8deb8a66(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 18, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6062b02843bc2a6ea08d485ee8fa543e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a46bf58648b2bf2b4bf0559d8deb8a66
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b08030e6f88d1a665e660ef9783ec251(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 18, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1f97a29c9676c42d2e5a31662f79ec3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b08030e6f88d1a665e660ef9783ec251
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45d17c84a7f744f162e69e038e9d9ab0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 2048, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae1226090b17d0db9969f8e2fa531304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45d17c84a7f744f162e69e038e9d9ab0
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2421fdd3affd11f7b89d3772fec88f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 17, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac893bb6bcc9c080b208c2755e28aa0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2421fdd3affd11f7b89d3772fec88f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7ac06ea1ff3411f84b2b8ea57a8e428b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92073e35bc5c70cfe6f3f8f0f31008d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ac06ea1ff3411f84b2b8ea57a8e428b
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23434a46b9e85672e9b64ebffafb5272(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 17, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_981e3f402c958ae544f15bb035fc1d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23434a46b9e85672e9b64ebffafb5272
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_80593eafadddbf7a2370cf6c92250bc2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 100, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aba746eaadb37fabb23f15ce2996daba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80593eafadddbf7a2370cf6c92250bc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_46c32a349e72d4f3018ecbd38683a011(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 100, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_180b0942838fb2567b35dd7a9ef24289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46c32a349e72d4f3018ecbd38683a011
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0d0a9ba824cef0ac0a347e40dc730917(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 96, 96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fef65614e176a54ae29ad34ca6e9bfee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d0a9ba824cef0ac0a347e40dc730917
        def get_inputs(self):
            return [
                paddle.uniform([6, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c660dda854c54632fedc029358df115e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9701dc9377b2ece64b135296ddd979c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c660dda854c54632fedc029358df115e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_84f80aa8a0efb274b985d4e3abe8caa2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f342ba82bcf8cd91664e2bdc8eeec05b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f80aa8a0efb274b985d4e3abe8caa2
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4cb607701d8b35c51df6631e639c6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a29474af273a2bbdf2856d84a69b9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c89158403c8c39c1b59386515af3e1cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88cc419c6970ce695a61567b5b4eefd1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d563b1f8cc5391ae1126074689771a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3140b5b601ce08c1a674e6fd8393a6d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3d080498f76d9c51d3cec9eaac0f36b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1053a0552753f2c52e03465660c26a13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d080498f76d9c51d3cec9eaac0f36b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c00cbd1cb7edeec046d0a6bde3738d62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e89f2d2c3b1aa899b06ec8f07238e157
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17bae2478fb5ea0f7c1e2ee8eb8cf19f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 300, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a593aee9926260218c3a45d69b4a882(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17bae2478fb5ea0f7c1e2ee8eb8cf19f
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_adb732a2466abb09485c2b424f5e9024(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 300, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f926906bd7c096ad9c74c414f709ec6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adb732a2466abb09485c2b424f5e9024
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac893bb6bcc9c080b208c2755e28aa0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2421fdd3affd11f7b89d3772fec88f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f64866bc301bb1a421a7c86c59ec8692(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 17, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30f8479c83b484c7ec842ffd9aa2d97b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f64866bc301bb1a421a7c86c59ec8692
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bc536d2ace4e3c128d791f3197f1b3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec5b6729b54e9029c24dfd50d1403aeb
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31dde02e378172d7737bd1b9b2efbf71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93b5b2f1c934caddf459c35c039f8bb3
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf729a3d203db01f039b52841507673f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2293ca93d51d47b2f1753c921fa252fb
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_151b4e6cefe131d9aed7b11950db77f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e9e0949a4d355ca5208ee47653519656(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_151b4e6cefe131d9aed7b11950db77f0
        def get_inputs(self):
            return [
                paddle.to_tensor([[[6], [6]]], dtype='int32').reshape([1, 2, 1]),
            ]


    
    class PrimitiveOp_2b1ecdeea77b1ee6b250d9bf53f6928f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ecad68a082417f3de564b23a0f0083a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b1ecdeea77b1ee6b250d9bf53f6928f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int64'),
            ]


    class TestPrimitiveOp_516eeb6465dddad2887dc32092a6857e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_405384f88b44804a0bcab13226be2b8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ee860b5c0e8e2bf5b20916eb9195ace4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1, 1, 2048], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11b9b7ccb390a7cd1c7a81230ac263f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee860b5c0e8e2bf5b20916eb9195ace4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 1, 2048], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3b0a0070ee4cd04fd27a196bd1ea3d2d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 46, 46], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54163a171c0f280967e684843e591a2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b0a0070ee4cd04fd27a196bd1ea3d2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4eec03b55c83a79d3b10832685a70e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 46, 46], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30ded1ce91dd5fd764edb64d44382dc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4eec03b55c83a79d3b10832685a70e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9987aa74682a656c01810fdf818a1b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dd53a2d7a6466d8e8dedcb773b93bf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3af88800ec5e99b340292a4ba80b358e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85ac4e72a94774cf34be0ab12364746
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8c06ddf682c429c6e765864ddd582c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5c014587e071eb4e58243296cfb60c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_516eeb6465dddad2887dc32092a6857e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_405384f88b44804a0bcab13226be2b8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c61e05e9e8d80a62f31f00dee892da21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96956a225c1814da05a053dc79806267
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f3f3ae42fb6254e592bc79355120587e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 256, 2, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f2d9522355d6368a824579597c981cf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3f3ae42fb6254e592bc79355120587e
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9987aa74682a656c01810fdf818a1b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dd53a2d7a6466d8e8dedcb773b93bf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3af88800ec5e99b340292a4ba80b358e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85ac4e72a94774cf34be0ab12364746
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8c06ddf682c429c6e765864ddd582c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5c014587e071eb4e58243296cfb60c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e630c792c22e9c67992fb0b0e1dc1d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ffd0e5d11586ba954b8a506ac13efaa3
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3a2b5138bf31149bfc2741050fc2e70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_936d03c83e3366603e601304acde95d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_2efe95c113e1c13189521d77eec5ae9b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c747c050d6f128b0666532e071985a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2efe95c113e1c13189521d77eec5ae9b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int64'),
            ]


    
    class PrimitiveOp_01a3401295ec28b1699cda20d4627201(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a0d1e87f3079870492638551cbba22f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01a3401295ec28b1699cda20d4627201
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_229d4a32adc784fde2139de63e9b3d49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c849d6143b41c948f551438e6535933(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_229d4a32adc784fde2139de63e9b3d49
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d491d477d45e98b1fbf76cae60e71fb3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 60, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4df2e76ab47443474c774b8b35d1a563(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d491d477d45e98b1fbf76cae60e71fb3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_93f23ffcab939af131a5f75a895ab1df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 60, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06b2664ca3f971462af2fa5e3e6572b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93f23ffcab939af131a5f75a895ab1df
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7014fa1590b52f4cffa8bb85e80b7a57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 64, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2fd1c58a4a68fa0fc421f7e75758393d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7014fa1590b52f4cffa8bb85e80b7a57
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_664afa0b2c927ca7fceffd69c2d871ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 128, 4, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_91aa3e7c75f30e98f5cba68874811271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_664afa0b2c927ca7fceffd69c2d871ab
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b480fa1eae8fb46a5988370a70df1a56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ccffb0fb7b078772d89a22fa6169601(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b480fa1eae8fb46a5988370a70df1a56
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f342ba82bcf8cd91664e2bdc8eeec05b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f80aa8a0efb274b985d4e3abe8caa2
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a9f200f2a2894d454b2bb65b32bc3f81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2048, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40b28c677aeeaa5ce1e1c99f190b0484(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9f200f2a2894d454b2bb65b32bc3f81
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_50320887fbcd21421fb1480a8fe4b0c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38f34106a937185fbea487748ab4c124(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50320887fbcd21421fb1480a8fe4b0c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_136687880c7b10199dee5aec5fc474e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_035a9397ca650b9abe04f75e17c8f1a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_136687880c7b10199dee5aec5fc474e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3c91e7c087abeba5e1d7ca91deb7edd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a279ecbff3356594f5f87e0bfbe16278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3c91e7c087abeba5e1d7ca91deb7edd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce4df54d56943600acc8be1d460f1820(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c61add3398336db087aa4ec8682512d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b3cd3f6a665a3c84ed22bbcc4d5eabc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 76, 52, 52], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e44c5562f87a5a30d4fbda16ea6ed4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b3cd3f6a665a3c84ed22bbcc4d5eabc
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_52e63b9c8b9fc9a93e1351112e5c63f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1000, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccd71321c33236f4ad0aeceac9474778(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52e63b9c8b9fc9a93e1351112e5c63f5
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d162eed6310240e05910a3f68de051d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54279e899d2449d050d05670e7d4525d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d162eed6310240e05910a3f68de051d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d5d08e2806bde090d29fb13c5a03cd7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 128, 4, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8440f9130106fb69b628ad2b9f198c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5d08e2806bde090d29fb13c5a03cd7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9987aa74682a656c01810fdf818a1b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dd53a2d7a6466d8e8dedcb773b93bf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0d1e87f3079870492638551cbba22f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01a3401295ec28b1699cda20d4627201
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0c5cc0590761bb66987480effaa9d1bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_662bf06d6d0ac44cc3d443731d5eb6ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c5cc0590761bb66987480effaa9d1bd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_58366c0cd56f2117effbfa7c1aa6bc28(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a57e5524b0ed4b87cab7796c11f552a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58366c0cd56f2117effbfa7c1aa6bc28
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce307f3931b899a3a40565e0b0fd3720(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3264f4ff24c5bc0d0e1b141f8a8b3c9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db00d1214490196bb1299dcad6dbcfd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_350848abe95f0b81533bba46e2896182
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ae20eb0d0e66efa6832e195d3975283(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07f04fc927aea790d0de8816e27bf9a4
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2555401fe7f4ecad4d01cfdda6127928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_100036f79ba97371d42ee04c62561c21
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f1310bbe3306dd31836335e0edb8ab5e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccbb954337e6e4485d3608c5f1688bb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1310bbe3306dd31836335e0edb8ab5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0255861800285f09702f23da585e7dc3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 23, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c76d13fdc6881cbfe387c74abdca932f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0255861800285f09702f23da585e7dc3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fdd817115a22980a3fdc2f593c3c68b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 23, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_572ab750ab972d51563c51d1b358d628(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fdd817115a22980a3fdc2f593c3c68b
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4cb607701d8b35c51df6631e639c6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a29474af273a2bbdf2856d84a69b9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c89158403c8c39c1b59386515af3e1cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88cc419c6970ce695a61567b5b4eefd1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d563b1f8cc5391ae1126074689771a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3140b5b601ce08c1a674e6fd8393a6d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_92f04d28af2b709dd9c57badc5ddb0d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 256, 8, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6d74d21e60324a7f678a11e6cf82bba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f04d28af2b709dd9c57badc5ddb0d9
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f7e1ef75c913f462dc738325a0d9a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfa51ab488f0a113c147ba4320529d06
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb481e499769a763c0b55f38b3a77825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_686c9b70cc5d9d2d318500fb27d333c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ed861d29a83a2ee15765bd31cd05c63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8ae371ea886ee6c6efbadd0412887f
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f7e1ef75c913f462dc738325a0d9a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfa51ab488f0a113c147ba4320529d06
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb481e499769a763c0b55f38b3a77825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_686c9b70cc5d9d2d318500fb27d333c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ed861d29a83a2ee15765bd31cd05c63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8ae371ea886ee6c6efbadd0412887f
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af74f213b00c37b095e2c12963263800(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96b36fbd1e3084fce335d43cac3f6d2c
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af74f213b00c37b095e2c12963263800(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96b36fbd1e3084fce335d43cac3f6d2c
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f546edb6d7896166722b4edc40e6d27e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0507c0a661bfa810b48ea732eb1d47f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a31c06cf600590db21e89b28e514d1a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_696d850d71e37c1d960ada9fdc42b50e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a31c06cf600590db21e89b28e514d1a
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a601b091031eac7260c39d2075c5eb34(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c40470f8f6da67d793d2cec5aed495b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a601b091031eac7260c39d2075c5eb34
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2d5d62cc40f0ed898f7d5ac2f8240e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5969be8591fd38fdf27347db2a60084d
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e7805700b074a4685259ce57eef1187(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25e8a5d79a23012f12d90d5289397038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e7805700b074a4685259ce57eef1187
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9332c8b2af13e58760eaa51c238d44df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 19, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33221b7e98106330bca45c3107a445c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9332c8b2af13e58760eaa51c238d44df
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a8bcb21ae2cae682cef6fe522b6fef8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 19, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc7612620bc7b6f7cb5c4a543187b6e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8bcb21ae2cae682cef6fe522b6fef8
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9987aa74682a656c01810fdf818a1b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dd53a2d7a6466d8e8dedcb773b93bf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3af88800ec5e99b340292a4ba80b358e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85ac4e72a94774cf34be0ab12364746
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8c06ddf682c429c6e765864ddd582c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5c014587e071eb4e58243296cfb60c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08e982410cbec3ac7503782866fc1983(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40a2030127dc804ce1373da6169f4956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08e982410cbec3ac7503782866fc1983
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3af88800ec5e99b340292a4ba80b358e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85ac4e72a94774cf34be0ab12364746
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f9c70f37d3062f268fb9e203e5632444(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ebb74c1c3ad087ef947eee254963193(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9c70f37d3062f268fb9e203e5632444
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_180b0942838fb2567b35dd7a9ef24289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46c32a349e72d4f3018ecbd38683a011
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c483dd4479664417fc57822989fdcb9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 768, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_730cf305f31316df0974abc8343edd0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c483dd4479664417fc57822989fdcb9a
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_824b704e464c7bc351d4ca956136e259(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 256, 2, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22b5ef5cabc6b9bc5f574ab416871ec4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_824b704e464c7bc351d4ca956136e259
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 25], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8295374aeb2bf166a00e1fe30d808c65(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 136, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_549523231831bf7fdb499db12f83b777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8295374aeb2bf166a00e1fe30d808c65
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_549523231831bf7fdb499db12f83b777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8295374aeb2bf166a00e1fe30d808c65
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_516eeb6465dddad2887dc32092a6857e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_405384f88b44804a0bcab13226be2b8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f926906bd7c096ad9c74c414f709ec6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adb732a2466abb09485c2b424f5e9024
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2555401fe7f4ecad4d01cfdda6127928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_100036f79ba97371d42ee04c62561c21
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccbb954337e6e4485d3608c5f1688bb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1310bbe3306dd31836335e0edb8ab5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bfb1937cbc148353d342bd6126006ebd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 68, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7d2ee25501510e680f1d4d4b0659887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb1937cbc148353d342bd6126006ebd
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86dcb3c1227aa168a42257d6f67c6bf1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 68, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f9d469bf1d5ff1fb0d9bf4423515f219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86dcb3c1227aa168a42257d6f67c6bf1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2513f7bda58ddecb27135f45f2ff7b40(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 68, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ed711b36cf437c9ca3f960e27238745(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2513f7bda58ddecb27135f45f2ff7b40
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9c4233f47e68184e306937bd3a3857ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 704, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a53dfd424b5b38097bb26ec50ed47326(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c4233f47e68184e306937bd3a3857ca
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e39676acb1889e3f4eafdcf1f00a6fd5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_66f7370dce3368575a108ee6735dc57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e39676acb1889e3f4eafdcf1f00a6fd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_274892a11e7d8d8f283ee5ef8482f918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90416868424ea14fa009663a544c1627
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d95bbab30bfbb49be121708cf7a9a1da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 72, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d73997bdf2acdab5c8937d1087f5bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d95bbab30bfbb49be121708cf7a9a1da
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_216475af0c3e5666867a888df4aa2d00(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 72, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa0ab27420ebd48ede2f48140c4271fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_216475af0c3e5666867a888df4aa2d00
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b824ae1b7f4d9ab45a994f0ffb7c5ebc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 72, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f837e1e42227a68b2fc8e643710b035a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b824ae1b7f4d9ab45a994f0ffb7c5ebc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_007137c29438173ed775909f6c81be09(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3ee64d26a5406dd36768b25f729b4c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_007137c29438173ed775909f6c81be09
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a090d1905f9c1d54a7e41217951a4c5b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 10, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc02fc8119ec0428e123c0e2ba4ed1c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a090d1905f9c1d54a7e41217951a4c5b
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_091a0ceb78afb8f672f0f96c37b7950d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 10, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72de4ccd4d9d3c5699b05e518eea9cf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_091a0ceb78afb8f672f0f96c37b7950d
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ae20eb0d0e66efa6832e195d3975283(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07f04fc927aea790d0de8816e27bf9a4
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da3f2c4a8ee60eb640deab686f68fdb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 96, 96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d23a3b1c21a322877df6333e7dde1710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da3f2c4a8ee60eb640deab686f68fdb2
        def get_inputs(self):
            return [
                paddle.uniform([4, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e07a6cc5694faaee8202a695c0170b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3ec060f4ed6c65be135b6a0aaefeda0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e07a6cc5694faaee8202a695c0170b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62416853549b8459cdf148611efcfbf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c020c67ad2d0a2a00be24755fefd71e
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_476cdd38f05278ef2a327ad630993973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94b6032eaf895cce3135bf2e15190db6
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78926b2569104a9601782f9823aa7242(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dcc2063ba0a2d166bc4c361ac32e53f
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f92096bdb5af50f5a8ffb79df9ba2ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4511c4b340aefd462a971d38be21bd7d
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c159f8e7953039478e6c9e054b96848(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5943a0ee354fa6d07fc4646d4c04cd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c159f8e7953039478e6c9e054b96848
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31dde02e378172d7737bd1b9b2efbf71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93b5b2f1c934caddf459c35c039f8bb3
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1e8d421d419added7bf76fb090b5bb67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1280, 32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2e33a10a6a1d94ced389bb0a6f1d871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e8d421d419added7bf76fb090b5bb67
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_476cdd38f05278ef2a327ad630993973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94b6032eaf895cce3135bf2e15190db6
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce4df54d56943600acc8be1d460f1820(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c61add3398336db087aa4ec8682512d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70fea07aa574ff1abe0b33f719d99116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a10d2538167cded3acf2fbc97b295a8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f7e1ef75c913f462dc738325a0d9a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfa51ab488f0a113c147ba4320529d06
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58f4ed89555f3c15a204e5cceae57f5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31b853eea9510f97a8c6ac2484fbf58b
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_feb125c40aa008caf9d1b197f6d80496(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc97cf65197caee07ecb5751668faf82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb125c40aa008caf9d1b197f6d80496
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07a333a4600dc18187513c31eaacb7ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b442b4446b56ee529dbce048ee8f451(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07a333a4600dc18187513c31eaacb7ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4b25c923a1d199bcdec37ca58d98e86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 92, 92], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6954c4d18341a13c18ce1c1ecfee8dcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4b25c923a1d199bcdec37ca58d98e86
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c14f5d7ba42fabc1796391d8e439df0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 92, 92], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_786e6800836413717a8e25c9cda1c32d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c14f5d7ba42fabc1796391d8e439df0b
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f7e1ef75c913f462dc738325a0d9a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfa51ab488f0a113c147ba4320529d06
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb481e499769a763c0b55f38b3a77825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_686c9b70cc5d9d2d318500fb27d333c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ed861d29a83a2ee15765bd31cd05c63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8ae371ea886ee6c6efbadd0412887f
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4cb607701d8b35c51df6631e639c6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a29474af273a2bbdf2856d84a69b9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c89158403c8c39c1b59386515af3e1cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88cc419c6970ce695a61567b5b4eefd1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d563b1f8cc5391ae1126074689771a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3140b5b601ce08c1a674e6fd8393a6d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8104fdc9ecc17ed6ea41ed217193c256(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d12ae457081a32d1dfc271d86f664ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8104fdc9ecc17ed6ea41ed217193c256
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ddc1638694fec912f0c30b0fae5ffc8b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a93d1fe70a1daf738ab34e0ec3099cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ddc1638694fec912f0c30b0fae5ffc8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db853d759b464c77cc3fdcf590a1f68b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8cb5b05cd31016a2489c8aafab80213f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db853d759b464c77cc3fdcf590a1f68b
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a93d1fe70a1daf738ab34e0ec3099cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ddc1638694fec912f0c30b0fae5ffc8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2d5d62cc40f0ed898f7d5ac2f8240e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5969be8591fd38fdf27347db2a60084d
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4562a5d77d8f84f053ad347e777c0481(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 38, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bcd768d93c9cb2cac53934da7ddf9aea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4562a5d77d8f84f053ad347e777c0481
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be360029fb96e5ac2418af90cee059e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 38, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_644314bfa32282b25668694cc50bc107(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be360029fb96e5ac2418af90cee059e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bc536d2ace4e3c128d791f3197f1b3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec5b6729b54e9029c24dfd50d1403aeb
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_589b3c22bf0e1397a71f810b91bb84cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 42, 42], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1cc8d365ffb6a8617927797a8ddc4167(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_589b3c22bf0e1397a71f810b91bb84cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3e0f0c5a937414991d0417a5722e0b18(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 42, 42], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4702efab7e3a6f2921ff9cee29b59581(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e0f0c5a937414991d0417a5722e0b18
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf729a3d203db01f039b52841507673f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2293ca93d51d47b2f1753c921fa252fb
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c8c579eaf0df1933c830927e22ec41a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_850863212a4aea527b5a6d9ce58c4f36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c8c579eaf0df1933c830927e22ec41a
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98f1b4b6d3eb90e5b6b3c60a27e45008(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9da04c97bc3e3b64f94e6b42ddd91c64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98f1b4b6d3eb90e5b6b3c60a27e45008
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_41cb7639e0697cee93ad1e0969ed4629(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 512, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cac0da634888a22d009db6f2d8a82482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41cb7639e0697cee93ad1e0969ed4629
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_381e4d7d2d1cf7450459cb4614032e19(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 512, 4, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47ea9c82a344f6c548bb7024537ddf53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_381e4d7d2d1cf7450459cb4614032e19
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_36299e00f03fdf0bb3d00887fd15c37c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b31b356a1d383a06392547d8daca696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36299e00f03fdf0bb3d00887fd15c37c
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7ca12f6be66973e3ca742f094009c868(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 256, 8, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15941f8b15d7bece0a26cd19ce88c79e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ca12f6be66973e3ca742f094009c868
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7d2ee25501510e680f1d4d4b0659887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb1937cbc148353d342bd6126006ebd
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4b72a5f3be761a658c8e8f12d9619dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 68, 68, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd422361e7c6dd0243e4bfeec7c2a77e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4b72a5f3be761a658c8e8f12d9619dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_992b2a1928e1ce27be08297da926bfad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c35ed30e4601e082b32d3ce47c31f1b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5223dea3cff37813ab5fc6cdd9221025(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 48, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6fae47e6406264453a7168a10ae23063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5223dea3cff37813ab5fc6cdd9221025
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c8f45b541557266a5b691223af05cdc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 48, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd4b8948c557992d64f37a4421a13b27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c8f45b541557266a5b691223af05cdc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0da4e91131a96ca94150f25cde107a1a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 704, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3937f51582cc5afa490ab06606480c33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0da4e91131a96ca94150f25cde107a1a
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e630c792c22e9c67992fb0b0e1dc1d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ffd0e5d11586ba954b8a506ac13efaa3
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c61e05e9e8d80a62f31f00dee892da21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96956a225c1814da05a053dc79806267
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cfcbf6c0f1d63b021767eb48cf1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d41d438e3c6e2bd284f66674d4e95f90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64c42728fe64b68b4a65139e0dfc795d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fed6c674ba976a965e19c2b25c512eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([512, 256, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e6fea8c782fad3cb3f030dfe53853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1037ba990fd187256c74e53c0ca261(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17b7ca82a377cc539d007dadd9137cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d3f484dd6c81bae93aac363349df2a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6e65858e1d6d923d14dd9d3005d119c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03777ad25cff3134e9ad14c71fd782d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24f7d328c712a44735cb02794d7a066b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f37e94bb5555b81a56226884b74c9c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_348d4666cf787d3cfdfd4953d28c746a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb7dfa000241af2df04c594c0ae9c975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d11485c5eb28b36203718c4024bc11b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f795c3f665969c7ee0ba44a3629db6c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad0f45db3c06583b342a2c96d2d32c57
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_3bfced70fe56d873b878bbb6508bb0f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_826922103c99652a4c2746f79a883e10
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int64'),
            ]


    class TestPrimitiveOp_39ccf5ff2748f98845633bd801dc5d4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b280c7f15f035587f368555ea84c6e06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d60a08b9605bb2eeacfc796689ba960(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc8477a1288fa42193e610b7ec2033b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33146770b3d6d39a870f1908b47e2a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01bd77eb14c16a866b7d98e69729829f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96653b87800bb7dfcd079bee9856f45c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8299bad157c506c3a284cd05ed69e39f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cf41273ee11f92b6a00a632745e5d75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5956700776e3fdc696b34167139a4190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d74a7f6d25844156b9698000dc934d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69fd5d46da99182359ed8881d84c1846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600b0888c930df711d534f3232077810(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2246ddb6af56b0930667c8a9ae5f056a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_855b2ea49b4b55714ccfddc3dc06638d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020952f6adf1ee1d9df81a060e0ef347(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f68e9b228aece7bedbbb7959cee508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c551c58a600e4cfa9a48e45ccef4c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e607b320c06a7741d3c6be2070fd1c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0a4e5cd36fa4fb473ee8949012eb9d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72815ec561c7d2b6685487ec5cfed577(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f36bf57b681cfb1e1a86211d8d24cda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c52a9806a13af3e672d210d630bca251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c8c3a6578569aa81f052b9486f88f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6689e6bd39438d86857e1ba4134a932
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03777ad25cff3134e9ad14c71fd782d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a69c023bc6bbe2dc3c7008241674235c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7d9a90a7949ef2ba29eecb60560539a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f61ac84bc085d6cca45a850741d7ed90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e907c626d15d418b3f1019d045009ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b49450105cd5966efdeade963e61ca8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72b659123705187d90cb9e3c68e8e67a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_379412f33f07c8b23a302ae9e268d748(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bfefcada37f25121d18ee591ac50f0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38e7318c0bd6c01ad1f1e9bb14d38a4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc8477a1288fa42193e610b7ec2033b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9acecc6e25425c4c85f5694421ee4ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_855b2ea49b4b55714ccfddc3dc06638d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ac169051a5b8e147cabfb35b0a9598(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeff61576b855f546f2e9e00386b5575(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([390, 64, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d34e63a400f28e20ba29101d1226acf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_381a869a87d4707688695edabf53017e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8e5febcfba09d0cdaec14afb9d7bbba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_afb05faa749b13270b7d6167b5441a1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 1, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7982dd9bc3c19eb31ee7afeb82d9d41c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afb05faa749b13270b7d6167b5441a1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a7e2bf1605ab7a8a6cf658217be993d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b556cab3d71b466d3e37f0a32e00e04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b556cab3d71b466d3e37f0a32e00e04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2baf0d6efcceb4b931ff4e738acb6849(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb7836eba28923e7ae491649b2b8e199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5e7c45c8776ebbec658a551d7d60137(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a96bb41ddd20017de47b8c311139cf17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a059a416d070e1a908cdd3216cdf7eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f37e94bb5555b81a56226884b74c9c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_348d4666cf787d3cfdfd4953d28c746a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b63a77250045dfdb4451f12d687933d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3b46dd203fdb2b06ace069cb87e58cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0a4e5cd36fa4fb473ee8949012eb9d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b98802d86e15b7321c2e8a9aca025210(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_662b7f314879b2b869f51b673e2be4c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84b7b291f1901cc7f69fd6906fb06189(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8eba107fcf86dc245b4cc839270c8d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_954439826c9b092b9a3dcc65d8f70ebd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69bc1884471d7c965623bee1f49d105f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c062939bb02e6110d7e37d189f90714(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aed3ffae21afe71da71557e7614bd6ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee348f6a3db7e7a894238635287892eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b8ec3c92cbc1809951d9ef75ef55dcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55dd5c35f32da228398b347e599a362d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6a7620438f038770f26353aa51cd8e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c453bf0581a2bed48f7882428d128360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50afa248676e4ef9b640705245842165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([6, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbfb33d6e6b303e4c765a3e96c2e9fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b00b87457f7f07479df0a0ca00fbd2ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc8477a1288fa42193e610b7ec2033b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33146770b3d6d39a870f1908b47e2a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9854037eb216b611ffdbf487c1f3973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a059a416d070e1a908cdd3216cdf7eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acdd8b33472a9b26d5a21fbfeccb3dd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1d5f3aed49768232c59c865b524430d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee348f6a3db7e7a894238635287892eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9fa9be653ae5c1b1a45dbe138b3a43e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb1109484687f8199a3279b252f0863e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff915d714a75f1a12ac860a5764b436b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4463fd36d9e00a672a6f52e3d2c6c68a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e89c798b48a447df693dbab22a4d1bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad0f45db3c06583b342a2c96d2d32c57
        def get_inputs(self):
            return [
                paddle.to_tensor([[[6], [6]]], dtype='int32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_1c58c14bd7e8607fed84d2d901f87a31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_826922103c99652a4c2746f79a883e10
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int64'),
            ]


    class TestPrimitiveOp_4c8c3a6578569aa81f052b9486f88f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6689e6bd39438d86857e1ba4134a932
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64059aac0f3d1eb4143536415953956b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 1, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf42ebff5129726f02e5425743bc1fa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe28890af6f7178ceaca7c39c0279de8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cfcbf6c0f1d63b021767eb48cf1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d41d438e3c6e2bd284f66674d4e95f90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c8c3a6578569aa81f052b9486f88f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6689e6bd39438d86857e1ba4134a932
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e73dc3bc322be46fd7039704041c59d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_104814c76e84d56b7340b6db3c8d8bbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cfcbf6c0f1d63b021767eb48cf1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d41d438e3c6e2bd284f66674d4e95f90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9ba08e3da0a89bcd9ed7a53716931cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f795c3f665969c7ee0ba44a3629db6c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad0f45db3c06583b342a2c96d2d32c57
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_3f750bd319e1fcf38e05073f69181fb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_826922103c99652a4c2746f79a883e10
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int64'),
            ]


    class TestPrimitiveOp_3c3928dd11766220ed802933f53d8342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_875fee338fd1a0e859214d671854ff5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34949560ebe5e6c7be8be731e68c20ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e83e5d45863c45c215e9f8cc4236606a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_884a251bf18d3466c6640019ba4c218e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721d66d3f9eb1b28f7b54324774eaa66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8c15bb5ede0bfe14a281c15d40f5b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b00b87457f7f07479df0a0ca00fbd2ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7014aa3b336c3cdb6bbe9e16979df1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93c558f3af77f6a1acfb961521386a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce806809df692103e29b7ebc1f3a6eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_310217f2882bba0e130cf7c8d963fb7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69fd5d46da99182359ed8881d84c1846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9ab2e2a393be15047f06a9fc81ef2d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c64c1c9ab51a1580778e9ca40f3960af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0590a9c34dd3b05969211b2941d54920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_366c7d26292f9cdf005cf90f13e9623b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cfcbf6c0f1d63b021767eb48cf1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c3928dd11766220ed802933f53d8342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b06a7d56cf3fa6506802f2a2c45b74d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_407ccfc74d2939d41b4eadfc61ae571c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_855b2ea49b4b55714ccfddc3dc06638d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ac169051a5b8e147cabfb35b0a9598(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c059e7a63621e006d016a387e5c5e1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01bd77eb14c16a866b7d98e69729829f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e78d8f367b7f74580006c4ec371b87c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f32b97ec478a4b68a5c06729e05b25b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4abceeb12587ee9d8093ae154f4479d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc8477a1288fa42193e610b7ec2033b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33146770b3d6d39a870f1908b47e2a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5931a7ab9e5f56916ba83cfe88ca5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f68e9b228aece7bedbbb7959cee508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c551c58a600e4cfa9a48e45ccef4c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e607b320c06a7741d3c6be2070fd1c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f68e9b228aece7bedbbb7959cee508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c551c58a600e4cfa9a48e45ccef4c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e607b320c06a7741d3c6be2070fd1c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08ea9e7dd14f74254e05e7ca986480d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17f1b5edabe22960bb17e32133af983a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08ea9e7dd14f74254e05e7ca986480d8
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17f1b5edabe22960bb17e32133af983a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08ea9e7dd14f74254e05e7ca986480d8
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb7836eba28923e7ae491649b2b8e199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dac08840befc0f8a4de8d61eb5bc0917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4fca373dea2427c57ab8d73bf8172c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20ebcd4fe9e2c4a62afb92de8dd97ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_137fa910e1eacac63cb496b7860ea2f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259943d386e66d666c09c5bc1b8e5788(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f81b6ae8fa80794270d6a80382853ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0cfcbf6c0f1d63b021767eb48cf1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d41d438e3c6e2bd284f66674d4e95f90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26c6685e7e18951dd6cf650102e1e2a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded7771003611349624ba5a5eca12217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61fb50400596bbecb7dd2698e1dc7345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c453bf0581a2bed48f7882428d128360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da6ca47c66dec8b2408d920480ad9dce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afb05faa749b13270b7d6167b5441a1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cc2324aedc5ab0482f6dcc59b58863b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d25424a52aa52f5b1d3f82a5c9fa6a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d25424a52aa52f5b1d3f82a5c9fa6a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c8c3a6578569aa81f052b9486f88f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6689e6bd39438d86857e1ba4134a932
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1d5f3aed49768232c59c865b524430d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3fd63c58545da44e44049f4e364d8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01bd77eb14c16a866b7d98e69729829f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e78d8f367b7f74580006c4ec371b87c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb08cbecc134ab2bb6b53327b4818145(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d712b047b6e0af7a137ce74adcf361bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dddd26ff04231a732d0e3acde1488e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67c92f8b1aff2fff6858657cb0814da7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16c8f2450fc11f7f8310fef66437f534(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5e7c45c8776ebbec658a551d7d60137(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66c36e7b9ee67dc336160339aa5b6353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e11b5a0b3cac291785897cbd8a7415cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a42ddf2a716490b8b8076d8bc38f3cb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bdd33479fdd204689e335e72ac3bd5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97caadc422a3b4564cbbd18a38a40b63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcfdd05836c46b97eecdff98ee6d6227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c059e7a63621e006d016a387e5c5e1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee399294fa1418fe228cde8809d44a80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([4, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_781c1ee04d3ce9bcb6f67554b5475690(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_381a869a87d4707688695edabf53017e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ade498720083244676dd00d8c9b3deb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03777ad25cff3134e9ad14c71fd782d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24f7d328c712a44735cb02794d7a066b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbc5fa578307c56be3b7431f7dbcd051(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff915d714a75f1a12ac860a5764b436b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f27244cee5cdffd0438968696283fde2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ade498720083244676dd00d8c9b3deb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69fd5d46da99182359ed8881d84c1846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600b0888c930df711d534f3232077810(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f68e9b228aece7bedbbb7959cee508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d34e63a400f28e20ba29101d1226acf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f91c7f6a96110f05290056c9a320fe82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb1024008cd7877cc5e8f838cdf2f42e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3995d76548b8d647bd721e5f677308d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab07023a2cf2b1c3b73a91eb2415d8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f68e9b228aece7bedbbb7959cee508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c551c58a600e4cfa9a48e45ccef4c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e607b320c06a7741d3c6be2070fd1c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc8477a1288fa42193e610b7ec2033b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61d57d76c72d4707d73dc858ee5f928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33146770b3d6d39a870f1908b47e2a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75b68756f3c68518518d0d6fd7456026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b866d3a144b5299c958d830e15d16cf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61a9456bd68199a37c39438a304f4fb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b866d3a144b5299c958d830e15d16cf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20ebcd4fe9e2c4a62afb92de8dd97ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4eba9554b533404887207f7432c2321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fd2113aa70bf466b40308a43902b204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb1109484687f8199a3279b252f0863e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab39e5169d88d75841fd987cbb711e2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c54cb507fdda656ff407c4b452acb784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4463fd36d9e00a672a6f52e3d2c6c68a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90b0c1984f230a25bc70a862be95f081(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_368670a6a0eec7184bc2b21f03f057b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2366360bfe8a25d809299c5efed3b887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11e4035634f091dc171103ba230810c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_267c700f69c3e2cd1f03e05424bbfa63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f586cb2817d7e991b549933d1f52a89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb08cbecc134ab2bb6b53327b4818145(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ddeebda2e7a2e9fd88c482577002eff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f37e94bb5555b81a56226884b74c9c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e882a196861549f9fe27c6571d6cef3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f39b784a5471a4f0623e023ece630a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f3774104b97234935136e87d9c87c67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70a483d0bb3bd4b50da21fb15ef9106d
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9ba08e3da0a89bcd9ed7a53716931cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e73dc3bc322be46fd7039704041c59d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43760c6900be8f9c836e8c54f452bdff
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()