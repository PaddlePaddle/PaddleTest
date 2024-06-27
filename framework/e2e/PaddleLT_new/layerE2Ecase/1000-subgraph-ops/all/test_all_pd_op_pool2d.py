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
    class PrimitiveOp_92d09598fae7d3a1ea1b84e6baa370dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f26fd82ba5b630d395e7e9a5d31a2881(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92d09598fae7d3a1ea1b84e6baa370dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6aea13a37c1dbbd1bc18486a2b58b5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_07b05f15d2d2b5ad780b5cb81c2fa413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8a17ca5ad1008a4685f18a496d836511(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ec338516edde5c8d79103e7bc7546c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a17ca5ad1008a4685f18a496d836511
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f60b18c32782486b85c2570baa8b7f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3ad0d5e19a45118f58de90957c900059(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d8252018f8f5738e67e93fd90c420a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_675ccb2eb318b312eebfbe2c007c5847(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_28acbd7bff41329478e2f3bda45cc485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_36073caef441cadf96bf06c581f371c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67cc813f428b36a9189db79671a7f306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1cfe89a64f00ba12a972a09737257ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f2c53f5b1a324e0f8c84fdedfae717b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a49040dceff5deed7246619715c272e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1947d91593f9f331551da07da25cc166(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1046b47a4b64f80853b0017b3d952df9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b91300adb3f323a54e3f218011b58e71(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f159db886dc28bb5f85a863c01097ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b91300adb3f323a54e3f218011b58e71
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_73c32ed17800569d690610fa9d8391b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02f85a1b6743653e86befee655783770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73c32ed17800569d690610fa9d8391b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_01c3f86db5f5939a77157f7dde821c7a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad114e8ad5a5d779de8f14021736f639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c3f86db5f5939a77157f7dde821c7a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_90d6de379fdf09cd0319f6758f29b902(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f39be5c9dcdde1801286a8828f53338(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c5347192ffbe42746536862645ccb69f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39085b6df163a3ded685936849be3f21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_70e713468d2391752171f53650d2f002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_44e6e7c080c82ca5df6dcb039cb7e452(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80bfc27a32ccd342262ce63bb5ee809d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6034bd62d8f204ecc7d61f8ab4fafc01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c3f86db5f5939a77157f7dde821c7a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e63b564a18c387d8833f5d9f0368a155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c538abe00503e0532ce828518939e906(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1152, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f633f31445b98ba553157ed2f32cee22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c538abe00503e0532ce828518939e906
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_90007fa5440e3293eef1ea52a6084202(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7c6d9d8081959b8c972547640934b2ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4b1a5ca113b73465c7909ef23b953c94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d73075c8bbad57fbb91dcd33f572b391(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3d3c037702cacc530ace4bb39844dfef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6dcb622baa13d1c75bef979984bb1bd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b73dca73a513f8bfdcb4f3a20290ded1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3c93423ed6b44e6652522b8f974f8bfc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37b531b6496ef8a7b75b51d890a76cbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c93423ed6b44e6652522b8f974f8bfc
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_06f8757572ee341646b63f01d150ebb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1233bd35a0e638d737e2885406bfc6de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 44, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_069deeec243274ad684a79f45caafe1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1233bd35a0e638d737e2885406bfc6de
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e996260b8ab80b46f2da91a538ca8a61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a17ca5ad1008a4685f18a496d836511
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f2bacfa183b867f9f6dee01eb2bcc699(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c3f86db5f5939a77157f7dde821c7a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9953dd2c73e6fc1e999271e564f4a984(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5aa541987de62741dfb1bff5d63e3499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9953dd2c73e6fc1e999271e564f4a984
        def get_inputs(self):
            return [
                paddle.uniform([10, 1024, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_90ca36bd8b1aa4ccc22047cd12f1a770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_321bff9bf86032eae6e92dd218feb072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e4f5ad31ae1241097639bdf77447c5ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c9936b0253fbcb387618477be99de99a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e445280c24c0e23dc530c4d4ed352b53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_16a1d72f219b214f5ddb731ef79cfa41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8283c626db37566ffafc5fc5c05d9d06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_54459b47e112bfdc12faa986a03e3cd4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74c1b58aa6ce3151c2d99c689cb68665(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54459b47e112bfdc12faa986a03e3cd4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_01ae03a6583bea562bb4147c8d9409dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47688eabcec06fd984b28892054915b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01ae03a6583bea562bb4147c8d9409dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7be10beb307a1f6ebf85fd026650ff13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d8b0463294c64179b489b2129f5f5f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7be10beb307a1f6ebf85fd026650ff13
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_99e87bf9a36a6c3617192ca8102bd517(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d03279d5a9ac817c0efb0238016d89f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e87bf9a36a6c3617192ca8102bd517
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_25f385415b5ae7372876dcf804d49dee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3bb9a5e6f27715255f5932c4aa731194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a15f4716da1cc14d3b1cfcdc9a959a14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6ea6637bb22c43ad6032959ebce206ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7758b91d49c8dc76df895b66cd8aeb0a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b818a4ce5ca76cb71ac634cb743bd244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7758b91d49c8dc76df895b66cd8aeb0a
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_770d5433e6fe5bf8731d650c34557288(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f18cba9db2b837ce26a03fd14169fe2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_770d5433e6fe5bf8731d650c34557288
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d9a4810d8c6adb1bd2744592031f18ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd119b80975a054e63f9abefad35b70c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9a4810d8c6adb1bd2744592031f18ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_55536b4dbf91af2ea29b3e6532ba6eca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6aa8884c66cfa741b3342b2416a4b6b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55536b4dbf91af2ea29b3e6532ba6eca
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_cdeaa5b4bbef167b2636cfde13b8e7e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68c673773c05561e61e0c4c2526bcca7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdeaa5b4bbef167b2636cfde13b8e7e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_986e3af9b72f4a5f706d7fb82030f41f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_770d5433e6fe5bf8731d650c34557288
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0e66d5c3e526dd306cdf49a8a7447b36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9a4810d8c6adb1bd2744592031f18ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_49e879bca4b25c6e3da6e09377f03abc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55536b4dbf91af2ea29b3e6532ba6eca
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_af58246bc2ca19aeeebedbc5f87059af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdeaa5b4bbef167b2636cfde13b8e7e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0d120ff2f3d8dcc5d6262bf62d38878f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_63de511e0daf7c319e1f7dda3a479355(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3d8252018f8f5738e67e93fd90c420a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1b9acd5a27b95960bac9c8dbbb93ca29(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83eab207757e36b55ce0801ed91f0c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b9acd5a27b95960bac9c8dbbb93ca29
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9d71b79b25b0432baccf7005232b9abc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_36db6893c90b715caa6b6ebfa51081d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a9f5a6d0b04dc5d3cc1613255cbf2344(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1417d50a380ad6a51f185870944eed5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9f5a6d0b04dc5d3cc1613255cbf2344
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_97d3f054b34f2602a0e187ef5a601efa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d3873a23a5993fd6aaad967392aaa45a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca1ed8bb2bdadee7aaeaeb8ffde392a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3873a23a5993fd6aaad967392aaa45a
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9b9ff9404aa60d7a789c6267f4cb089f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c3f86db5f5939a77157f7dde821c7a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_da604dc70e77b62fc1758c932cbc3786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a04240dcd7e61c37aa3db234040b64c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1ccc6bd26e598422189709ed4e9ab0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0ab8dcf2f9f20c0b7ca5c33a6cb9ac2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4d40ab242e5f3884da747074fefa5471(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51b69fec723ce6518e75c688798e2bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d40ab242e5f3884da747074fefa5471
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c09e6977c1104e063872c98a1a94b5ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2cb531dfa12188818833b69875ed845c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_12721d90a7c77a5e65d4807faa6af202(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ecfb39dec438e2d2e24fed3acacbb08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12721d90a7c77a5e65d4807faa6af202
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5ef0b0cbf80fbf0f82786ddea9d62631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_11be9f77e5d981f7801cd0524ee7675b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, 1, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_93df9a44542f8d45af332f4417e4c49d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11be9f77e5d981f7801cd0524ee7675b
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_512fc81d4fbc4f54ef05de629f9735fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9f5a6d0b04dc5d3cc1613255cbf2344
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d48f21cc8ce35e847c74ef1bda6721a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_709ddb1bc941f25c201084879063e508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2046c117749eb1c7bcb48646c913f848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_967acfdd95e87237e9638d662ec19356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2f3422db0116f1361ff800a54958740b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3850baaa9b53eaacbb51fee57192edfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0c70643fad40b2d16d847b3bf229df0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92d09598fae7d3a1ea1b84e6baa370dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d963883c687ba21a5a5563ec28ce6969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cc62a75d6ccf294ce4b85cb47d509da7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f1ce6b4823e57d9fdbda513ac458611a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cad758b492d1a0fbd9d3284f805662eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5c1207ad52cdb119916c1cefb3abc292(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([11, 320, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_652c55baf0517a3ad753e45aa8a5e75d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ccbf90aa53d6ce09074a3f33dd8ca0ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f4c96fc47fef7d78f4cbfac43768dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccbf90aa53d6ce09074a3f33dd8ca0ac
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bc109f8e472f2d4c1bccdaf41e7bdfc8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c76f74437bffb815fdbebe421975c187(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc109f8e472f2d4c1bccdaf41e7bdfc8
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_098ba20b663df11e05d32784c98b7424(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82c46a9d6804ed60b7a5d7f45db519d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098ba20b663df11e05d32784c98b7424
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_eb7a4a475847af5fe219f81fb3911dec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d97ff76876a4168f3129ae62a8c8e0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb7a4a475847af5fe219f81fb3911dec
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b0a8418e028f9102f747d5edeb700514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8da520d8a5606513350b9cf4cf1b405e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_95a047951392344f668c8e797dfb8929(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_14f60dadb1dff98fa47ecfdedd9bc168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9a4ba99899d98ce3ce9f39dbafd1334d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ee9955521742d273cecc420d8e05de35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b6ceb092789ceb853dacad2003a209b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee9955521742d273cecc420d8e05de35
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_11a2986c740b0266599a1712c15c07d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fab59017caf3dc4a57a829f624a8f3b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1754866463950a72b74ba9c5218dc87f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d299d817d769d2484c6f71f32ed9483a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bb27be61637286d17ad643f7abe8a607(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee9955521742d273cecc420d8e05de35
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_64206edc21ad000cc18de47ff2e58be1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e63b564a18c387d8833f5d9f0368a155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f4344c6299085aae9467a9d089f35274(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_077d30a07908e85badc640039b029f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccbf90aa53d6ce09074a3f33dd8ca0ac
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4955d87124dcdc72c996da3dbea11cfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc109f8e472f2d4c1bccdaf41e7bdfc8
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_662bd9f58c32ef481d5cf28bf4b792ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098ba20b663df11e05d32784c98b7424
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e3c600ad7fd5669e583ed12c3d7a22f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb7a4a475847af5fe219f81fb3911dec
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a47c7716ed298ea33677990d4d5987ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c4007be955d500491f733be23242b3a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d1635a365d369d62081edfdb9fe56c77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d4dfc35800ed9b6408f9ae730bfd9b80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a041e23a26432ffdea628b8f2f2f734e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e00b9c94a87afadcc5fc585aa35bd1ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a041e23a26432ffdea628b8f2f2f734e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e00b9c94a87afadcc5fc585aa35bd1ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a041e23a26432ffdea628b8f2f2f734e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_251ac5646f9ac5cd1108880a84d43147(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a041e23a26432ffdea628b8f2f2f734e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_251ac5646f9ac5cd1108880a84d43147(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a041e23a26432ffdea628b8f2f2f734e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_49704ef7393f7937a4f5aea21ac917ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a041e23a26432ffdea628b8f2f2f734e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_49704ef7393f7937a4f5aea21ac917ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a041e23a26432ffdea628b8f2f2f734e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f32f1a050d3a68676c7c16c103897495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a041e23a26432ffdea628b8f2f2f734e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f32f1a050d3a68676c7c16c103897495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a041e23a26432ffdea628b8f2f2f734e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_017e500f881db9b3128d85e3b0a072c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2af8e9563ef065b51786ddf5b80011e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d2af8e9563ef065b51786ddf5b80011e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_155bac027a34142e1aef96d5f74e3d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_155bac027a34142e1aef96d5f74e3d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b2e3444cd39ff4f51af51fdc7eeea7fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b2e3444cd39ff4f51af51fdc7eeea7fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_50f1c2f4ef7178c3a355b6113403969d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_50f1c2f4ef7178c3a355b6113403969d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_53a57ed9001b957747de2e660c424e3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ae3845e927efeb79be091662413836f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b9acd5a27b95960bac9c8dbbb93ca29
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d963883c687ba21a5a5563ec28ce6969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_75215c1ca497a60eb7038d390fddedca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fb61b898fa3eeb765f575855566f94aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c538abe00503e0532ce828518939e906
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f2d77b1bada2eeb532948967a81a3462(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8e523bd6f5104a9bfcb50a0ed9217736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2d77b1bada2eeb532948967a81a3462
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_555f2b62bf432232c202d06dfecdc96c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f617236a430f3cc294a5a287918ca611(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7a79f2a3fcddf18d900be92e402ef9ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_aac1249b1ed37788450a3617a2b432c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [14, 14]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12924b02bf64b02a38d14bf59c08e623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac1249b1ed37788450a3617a2b432c1
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_12924b02bf64b02a38d14bf59c08e623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac1249b1ed37788450a3617a2b432c1
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_36d988cd3ebb119f5c8a4d52aeca18c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac1249b1ed37788450a3617a2b432c1
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_36d988cd3ebb119f5c8a4d52aeca18c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac1249b1ed37788450a3617a2b432c1
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_68eb926f8137d6e0b2f3d0a127f350a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac1249b1ed37788450a3617a2b432c1
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_68eb926f8137d6e0b2f3d0a127f350a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac1249b1ed37788450a3617a2b432c1
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fac389809f3f9df9353a019701c62341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac1249b1ed37788450a3617a2b432c1
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fac389809f3f9df9353a019701c62341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac1249b1ed37788450a3617a2b432c1
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e40b9574a70fa89b9b7648711f451907(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f105759939a31a55623b271f17d92453(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e40b9574a70fa89b9b7648711f451907
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8d185be9113a940ff090ce841d392760(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80aac91e0ceaace06928cf336b8ff254(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d185be9113a940ff090ce841d392760
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1a11dd2e133f785629454b1cd30b63c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40d1f00cb694ad58e57f5c54d5882d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a11dd2e133f785629454b1cd30b63c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e2a706b70b2658ea9a6709e97452f10c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2458fc0d9b26c90e11f593bc5806df88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2a706b70b2658ea9a6709e97452f10c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5d9b2d0affdcd6a3f97aae79da4a0a08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c5a43a07681176745c2b2880e8edcb3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b2d0affdcd6a3f97aae79da4a0a08
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3a8678ece21979793bf8078ddea33b31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f4bd60f1be4855816fb7d38afc2cd55d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_341fd2939cdbbcb96ad03fa4478b826c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_20db0b7e9e1143d6bf85e833acc00ecb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54459b47e112bfdc12faa986a03e3cd4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_af041b5d49836cabce617127f14f3932(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01ae03a6583bea562bb4147c8d9409dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_044f5d3bb04cc78c96429c8fac53d642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7be10beb307a1f6ebf85fd026650ff13
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_13e6b3d6987826575e3943ba808fa034(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e87bf9a36a6c3617192ca8102bd517
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d74fde9b603db1fd4f39f28a795b8674(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c3aafda1ffbeef92246f3b2ce43fb2ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0e420e8f71f7e6c201837ac20acaaba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_077ed5bd388775e3dfc1f8672a01c68e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_570bf4fa6fe6efa0f4274bed6b324c5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b52bebcbb03c15bca2ce1c5c01ef207a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NHWC', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1f41c7f5be8fdf9eb708fa977d8a121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b52bebcbb03c15bca2ce1c5c01ef207a
        def get_inputs(self):
            return [
                paddle.uniform([22, 7, 7, 2048], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1bc9ae97131532589bd1a2781cd74ebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_18c8b509f31df4b68e18e267cd4aed70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7d8c4320756659413a7d6d8a8230b17e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f52745688513f68feba41bd8a25a51ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a17ca5ad1008a4685f18a496d836511
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a738abf52f7f8c2c75d0f6c9999fc683(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7758b91d49c8dc76df895b66cd8aeb0a
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6c26d4ef7676960f1395b14b171e3049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1feca8cf6bc2488ae2d872bb91db65e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_518db508c5e134ab24608360f9af896e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_017e500f881db9b3128d85e3b0a072c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_76514242ae84829f30b78172e1c6f9a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e89f99d257c72d1a319507c8e1f176a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4ee0182036757880f3d3862bdac7e5a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1623dd02ea64004015ddee2cc0a8c38f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6b9c514e278a1bc5829d8eaaf8cd61e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c93423ed6b44e6652522b8f974f8bfc
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_69f0bdacade9675832b3c09d53afe2b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_28eb742dc7bed6474ac179a46612c6d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8d368445c6a52015b708645eff35e79d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dae63eff4f0505a948dbc8dc3708fdf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccbf90aa53d6ce09074a3f33dd8ca0ac
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_23c112ae9678bf38640a89c4abb20f74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc109f8e472f2d4c1bccdaf41e7bdfc8
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0b835852357d93ecb2c430224c0cd232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098ba20b663df11e05d32784c98b7424
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_28d2e246cf5a29bf01dfa071944fae62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb7a4a475847af5fe219f81fb3911dec
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fd374379f1dc1e5d8712dbffdec7a412(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_02904cf6c8a77e213034982e7e7596b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d40ab242e5f3884da747074fefa5471
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_15659a90cfbf7ba5d733a1787b593aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ae2f5ca9d926ea6ed374eda1ba4c551a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a4d9240fdcad9948a813c8a3184ce0a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_51d3027807a9320686c89bedaf7d7697(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 400, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1c8f957fccdd4e443ac0f7f5fc4ed51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d3027807a9320686c89bedaf7d7697
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_449a4c8b1c2f906cbb74a33a449e62d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2ea1f1f52193b6c49bccb63c600179c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_86bfe1de590453628393e28592388fc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea1f1f52193b6c49bccb63c600179c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_35c000d7986b26386be34fd0ce977a9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3e36a333e68d50181abaa681544e2440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1ef1a05e6439fd805b5418796ca60d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5f48207e0e8c98f3bc3a45584aad797a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_45476f23ef26f104b7815713fe56ecd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_38a9f1ca58ac1e84523c3928923c3077(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_22afec9c260cb092cedcd53ea957b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdeaa5b4bbef167b2636cfde13b8e7e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 38, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_83b2d4a2d47a0c83a9591a9486aab9e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f1218046008d78dc2236bf809ac0324(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83b2d4a2d47a0c83a9591a9486aab9e0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_521c3a87ca5beeca9218f8cb7afe65eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b9f246774f011bb1755c7603fbd95dd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee9955521742d273cecc420d8e05de35
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_48cb64ef132b80a86cb620f021f3f0ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_21af9ab70b377db353809e406bd0eb9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12721d90a7c77a5e65d4807faa6af202
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f61dad810fe7f91a8ee8b1b24effb894(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a3ad108a23c7dba7649df6b44ec36fa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9a00c5bc3fcd8101e2d21c00a6d4d566(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([43, 320, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_77df9c85df3cdeaa30aaaf6e6dba791e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 336, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd2fe3c4fc23f77eecb3ee3904003c25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77df9c85df3cdeaa30aaaf6e6dba791e
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e59d49abb133b2abc5341c92b90d23eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2d77b1bada2eeb532948967a81a3462
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b91d8789872b7e9b36bf05443d7c79b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1ef1fca4487ab0fc5da6ca9fdaba1529(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bdd89d23a57f3b420c380e5d2c499b56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ef1fca4487ab0fc5da6ca9fdaba1529
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_83f90e5980cd219f748f1aa74c632c9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12721d90a7c77a5e65d4807faa6af202
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_715b152d01f50ae42204165ed6b6492c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1233bd35a0e638d737e2885406bfc6de
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8d99612d1e0e02c7434fbe75ff705b33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5ef65550c5aa8b14e6950987646d56e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9953dd2c73e6fc1e999271e564f4a984
        def get_inputs(self):
            return [
                paddle.uniform([22, 1024, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0c9ef44856e21a1a2db6a7f046f0ca53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d3027807a9320686c89bedaf7d7697
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5e274a6a36e5d1afed9f5717358cf208(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 56, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebb43ede7e737c498148629d6b199cc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e274a6a36e5d1afed9f5717358cf208
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_449a4c8b1c2f906cbb74a33a449e62d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ff48cdfa4ca4968ae9b6372eaa1f14f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83b2d4a2d47a0c83a9591a9486aab9e0
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_037507f6a6b1560feb161575c87e6bfe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5986321c6ea2b1caba51ec6013a185e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_037507f6a6b1560feb161575c87e6bfe
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e087d7eabcc52077596f5fed864b271e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a17ca5ad1008a4685f18a496d836511
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ddd29ecf7a81c6cbed9c831e5fb1bf12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12721d90a7c77a5e65d4807faa6af202
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fab59017caf3dc4a57a829f624a8f3b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f6816635fb5f35e137adc06919b867e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bd0994a46e73d112066ea5d993d58275(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a8c9688caed312bca4f9f7b0fa903c1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e9a96912379835588a99c89fa635308f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b24793abeb62cbfbb6774a32504ec0ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca181ea10d68e78ed2ff740df0bca45b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24793abeb62cbfbb6774a32504ec0ff
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_19a5eef8c9bed6ec5386747cf8106f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24793abeb62cbfbb6774a32504ec0ff
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bf5372f05fb7cd5faff297af6f42bcab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24793abeb62cbfbb6774a32504ec0ff
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7cd11b8416da22c5e80a0d0a04602c08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24793abeb62cbfbb6774a32504ec0ff
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fb61b898fa3eeb765f575855566f94aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c538abe00503e0532ce828518939e906
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c86f3804ac9c27bb30203b56c7c139b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1623dd02ea64004015ddee2cc0a8c38f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1b4ae3cff3f20efb79fa79ff311890ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee9955521742d273cecc420d8e05de35
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2c4eb2ae569b519591bc114409dd53ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_814ddc72113af129b2ccea126ebc6c92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8c6e5edd3ea6b6dde9ae5fc2e097b745(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_83c294259515a3fec5a1f34f08db1928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11be9f77e5d981f7801cd0524ee7675b
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_24b10bb5a02680da26c0c55edb64e8fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 200, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88cfa0c8f6bae6b2cf8bab587aff7eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24b10bb5a02680da26c0c55edb64e8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fff1ecef7bd50f10f7b2391786271a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0c4567c36d8d642039378aae703cc50d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_69430fca5cf237b3cb044643c4a137a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5ade5cc762a573368d3d05b4c3bab07d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ef1fca4487ab0fc5da6ca9fdaba1529
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c5477ac2d9874b9048994792d170c6fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ef1fca4487ab0fc5da6ca9fdaba1529
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_44e6e7c080c82ca5df6dcb039cb7e452(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c5dcba0a808d69b8d24a93c1ae85325c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [7, 7], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_095e82cfef07834f72fe351c3b1e172f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5dcba0a808d69b8d24a93c1ae85325c
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_39085b6df163a3ded685936849be3f21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bd33fd9256105e6890bb6e820e9bfe1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3873a23a5993fd6aaad967392aaa45a
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_470a954b61b3823507e214c49c99e7d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_820c5207994a22dd3c0eabb8ffcc76cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_470a954b61b3823507e214c49c99e7d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_039c62b04ae97dff43bf4f27dc4d3f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b27b8148aa5a30398ce45c01c57b5d10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5c25f2b062a19e9a81c813f4f06e3b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1ce567941669ddd3d607b29fcf1c2bda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1ce567941669ddd3d607b29fcf1c2bda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7d3b7ebbcef65988eebef07dc45e9862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7d3b7ebbcef65988eebef07dc45e9862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3e371f667cc7d4011e85671db7d08aa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3e371f667cc7d4011e85671db7d08aa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_17befb2bb23087ab71aae17eba90bec6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_17befb2bb23087ab71aae17eba90bec6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ebdb76bdd6e5e9f9f9e6099eff7149
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_86d56b701fe0752089aa4e6bf84d7f69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7d2bc50f20b9064b03b24e705bbcdd6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a0299fc6f862eca730fa75572a16ed64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_71c28b2396f3e45e0d18d92f3f37cb57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_288d0ce2de9e0e08a08748aad347b19d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fbd8af2f18a848108541c527965b823f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_54c52afa58db335aa1f5f70c53cd18cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6315e85bf41b829594361f4d0bbcfec3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ae65db496c2b33b438fecdbb52260bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_38d271c24387ddccab77657f14a208df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_770d5433e6fe5bf8731d650c34557288
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e211e39607992cbee93d49613fd0a23a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9a4810d8c6adb1bd2744592031f18ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_452138c536882deab65abb5e90abcacb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55536b4dbf91af2ea29b3e6532ba6eca
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_caf4a671e99259be4b546c9126c06094(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdeaa5b4bbef167b2636cfde13b8e7e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bcc5ca1b9c297dd70a8dae468da8c4da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_50c34f9f963ba37f9f6f13a2d20f7f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bcc5ca1b9c297dd70a8dae468da8c4da
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_345949db6e2f86cf87f73d7643675659(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44a4c9a5aaa6c3c98759dc9d98d5b1d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_345949db6e2f86cf87f73d7643675659
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_df641d38c95d77cc3afe29ef01cc6c7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01e2f20cbd3dc6ca1cbbcdb2d9aaf72c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df641d38c95d77cc3afe29ef01cc6c7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3a8b6dc5dddade630e140a518668c966(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e13361f02b305602058b2f02da60fb12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a8b6dc5dddade630e140a518668c966
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f3bdee4f5839097bff0f02a3ea03c639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24b10bb5a02680da26c0c55edb64e8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f30f1b71a8f65fca34feb875c6d65746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9f5a6d0b04dc5d3cc1613255cbf2344
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e76c7d6e68b4b82d8e5d8f08a409daed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_561a4dd142629616ef3bf98ae67b0615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_93591885e76a709fc85b36750824d5cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cbb1bfd4d5c5bd9bd52568fbcf6a25af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e1c238137505a78b237b93ef8d99498f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3c803ddbeedd2a39cc343a765c503676(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_779f30594393353883f4ce3df48b0600(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73c32ed17800569d690610fa9d8391b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_449a4c8b1c2f906cbb74a33a449e62d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9e1a53c90ccfb18e7673c5b2da3cebec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ac70cba3dd64286c24d6e8e45078de2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_06d0071a7aecb02a92cf5b9b0e62c154(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e5f993ac9ed98e36fa52fb139477f48d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_84ef31f8d46c27197d21e457d060d168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fcc69a639d8c62f4a7a66a08f0ff37c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_734193ec7d9917a7b29579436b71bc84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7d0ca5e305083345dd524b4fcf00788
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4bf4f1adc2b11f5d6e5976575a5581bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bc548a9408c23ad2763c233cfd76e8af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_472e8213b567df8408124125ac3bc67e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9d73671d58b1eb646c6de8115d148305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_635c1df1b7e3417d6e0cbf0a7806dd9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_68aeda0747f7b70f85ba4bbb17679f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2f159db886dc28bb5f85a863c01097ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b91300adb3f323a54e3f218011b58e71
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_02f85a1b6743653e86befee655783770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73c32ed17800569d690610fa9d8391b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a26dd258cd62108b3b03f001b3ca4c14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12721d90a7c77a5e65d4807faa6af202
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e76c7d6e68b4b82d8e5d8f08a409daed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_799fb526745e553dddef8963864b6f0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a17ca5ad1008a4685f18a496d836511
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_93591885e76a709fc85b36750824d5cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b91d8789872b7e9b36bf05443d7c79b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_178063d807f5a88a11eba7863c8f4a48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7758b91d49c8dc76df895b66cd8aeb0a
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b1c7668fff075a542320b463e842c63c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b9acd5a27b95960bac9c8dbbb93ca29
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d823e726adb2918c19766b5d2245c276(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_765eb5db958fb6cb3d331c16a86f132c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77df9c85df3cdeaa30aaaf6e6dba791e
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fa8339a098886d11ba9b7acfa6983331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_037507f6a6b1560feb161575c87e6bfe
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5d9d79cf6c73d3e42da39573f0063bc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e274a6a36e5d1afed9f5717358cf208
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bff76d87b874bebad6de84c875d39071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8f5ced2041cb6016fb270343ca124a96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_eda8ecb5c7f7baa2b832ae30e018afa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1d6e51c8e486263495295f82dc2c2cfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12721d90a7c77a5e65d4807faa6af202
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bcf43d6104bfa38c0b5eb1a7a5781bd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4c47bd6bc7f159da1b32b5a7b7b71c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_881842bca9d200d28d90782bbfca1ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_462232de8e7fee47852227ab067a7bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6ef67698daaf2fc584b016ae653c5f77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c24fe6697402f26ec4eff044e4bf2995(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_429601f45338cf16e36291ef2a5c48bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_814cd5085e4745cc5e73386c9d52098d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7a79f2a3fcddf18d900be92e402ef9ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9839e3f9dae15a0c66dc640bc40ea889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9f5a6d0b04dc5d3cc1613255cbf2344
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1141432d6fb02ca9fecef711721c7974(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36073caef441cadf96bf06c581f371c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ed26af0fd4adcfe3eaa2a45eca8b0cc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c3f86db5f5939a77157f7dde821c7a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e7d6c2205182b1d1282c478622dc8230(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a17ca5ad1008a4685f18a496d836511
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_14742e6519203026218ac0bb83b03b8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b87e9198ae0367848749ff1fa21e287f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccbf90aa53d6ce09074a3f33dd8ca0ac
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fb11d3270aefead7db86cd05d43fb603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc109f8e472f2d4c1bccdaf41e7bdfc8
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_60acce88c6f1945e2cbc66d11c35247b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098ba20b663df11e05d32784c98b7424
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_70337c2310df34a8c81a72569ee96eff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb7a4a475847af5fe219f81fb3911dec
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_73c040f62a8f62bc2a371f2f5e90c514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bc719009396031551a687f0d69aa8938(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9e68138edbb1abe899ed0df61dad3a17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_26a95259a4cd1b4f0b794e47da10929f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24b10bb5a02680da26c0c55edb64e8fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_11e33baba99ee7a20c1aff49f8ba0bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d3027807a9320686c89bedaf7d7697
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e59d49abb133b2abc5341c92b90d23eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2d77b1bada2eeb532948967a81a3462
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_486cace911f4035c86079fca1de68961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8246eeb17e3914ec0fd7fbc594b32707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3873a23a5993fd6aaad967392aaa45a
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3850baaa9b53eaacbb51fee57192edfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bd33d328f0e0a62ce299b04b2e8c71ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ef1fca4487ab0fc5da6ca9fdaba1529
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8e523bd6f5104a9bfcb50a0ed9217736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2d77b1bada2eeb532948967a81a3462
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9d71b79b25b0432baccf7005232b9abc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9033297e814d334c512a1efa958b33e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a17ca5ad1008a4685f18a496d836511
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1e161c5aad1b0ff434324a609c4f30a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9f5a6d0b04dc5d3cc1613255cbf2344
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f966a96b2a92c71bcd4ee2b053e95009(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b9acd5a27b95960bac9c8dbbb93ca29
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3e36a333e68d50181abaa681544e2440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_33350f99491700caa7fa9bc7c144e2e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a17ca5ad1008a4685f18a496d836511
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f285e4d54b013da6c8ad2044a36a04d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9936b0253fbcb387618477be99de99a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7d2bc50f20b9064b03b24e705bbcdd6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5347192ffbe42746536862645ccb69f
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f70a767b3120c89f769162ce7b73131e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_04b4664ef68bc0990f0abad176aa53e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ceb1b2f73649f4960ee421414b4b64ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_960494cc6c82e8f5df8e95c20ddfba05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b10fe276ec1788483736aff621700e88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_185e0eb34a3893edc1d4566b81b0786b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24793abeb62cbfbb6774a32504ec0ff
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dd9a739a251ccf016d2ec8701f0196d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24793abeb62cbfbb6774a32504ec0ff
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9b8d28dc2a1a1367ef4d3dc0944f42e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24793abeb62cbfbb6774a32504ec0ff
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bdbadd03624dc2a609631b5006e63a21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24793abeb62cbfbb6774a32504ec0ff
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b4484e4f574e3da236c6ca5144b77ca9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01c3f86db5f5939a77157f7dde821c7a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0687122628da6d31537e815e8c374b87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_470a954b61b3823507e214c49c99e7d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d4dfc35800ed9b6408f9ae730bfd9b80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a2986c740b0266599a1712c15c07d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f1d6f058d76a85f2b7e4375f3189ae2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90d6de379fdf09cd0319f6758f29b902
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b97df2ea71babc08df1b4b26ab4cc154(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5dcba0a808d69b8d24a93c1ae85325c
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_df7751ec85e562511463ccc5d65fc710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_238d351436ae9dc336d3ff2796e62520(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_be8186f3e17af6236a210e90e9396af5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fb9829e85e2e3e2b4f06623e1112ea35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a48e1cf5a7459c9634f6d605d9d9e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9f59a5be937d6f43cf8fd2d8ae4d9b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_339b4c8a9dddd6ef8d95fa4b1466ce71
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_27a0c838d98a5c04c1c9f127aa668bd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e60f5f47fb9d5cd827b02b5f3eaf8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_402666dd6a91a8c54a2881c89ce5d510(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17c255cda5101b20cd8c2c98dcf4153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f633f31445b98ba553157ed2f32cee22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c538abe00503e0532ce828518939e906
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_49a345f0fc2cc732b57bb19bd3d75afc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_575819fa80c6af22592a0ff2e7a5d1f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_fe174e82af0541fa8f4caf52aef7fc65(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 576, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3013ca8c25047c621e9d7127cd2c270(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe174e82af0541fa8f4caf52aef7fc65
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4262ca1018130b04597ab606cf595518(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61839e1fa9d7eab53d5c32590bd1d5d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4262ca1018130b04597ab606cf595518
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_726946326f7ad0f1e9a987c2be2a63dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 92, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6c6336259a519cef9a4f486f6937841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_726946326f7ad0f1e9a987c2be2a63dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8462f55e40f81396939b012db7807d86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 30, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff5836ced6f828d4ffd85ab8e756cc17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8462f55e40f81396939b012db7807d86
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ce045c35e041000d640d7547c5775e5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2048, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d22f3e3f02b4e2fd2e7a757d976ad0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce045c35e041000d640d7547c5775e5c
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2d8ec86348104494e303572e8d90a7d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 16, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cebd9852340a531a2b476c8660632af3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d8ec86348104494e303572e8d90a7d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3e30d906d8d1e391b123b21221a96a0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 672, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b0e4bd693d0ff659623ae964d538642d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e30d906d8d1e391b123b21221a96a0c
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_687e2fed955bb427c5041490840788a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb4c07a35b096fd2643ee266c9c451c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_687e2fed955bb427c5041490840788a4
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1cc17f6da2afdbc1c61819da5b536ab5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd5f36dcbc7ff60629db70037cf61bbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cc17f6da2afdbc1c61819da5b536ab5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7916d7407e4ee581f9ba8a99578fe5cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 44, 44], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c971f5cb0f025909cbd509f89cc95ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7916d7407e4ee581f9ba8a99578fe5cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_dfb20db930fad299672fb0d2785d460e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 336, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_670492af8004f2fa80e86088c298d062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfb20db930fad299672fb0d2785d460e
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5f7bc49c43218b68df167cd9095a834b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b62ec18dd97a24ac793b9a9be6330de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f7bc49c43218b68df167cd9095a834b
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_291132d6ede22f3d6080a3572176289c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dabce0aee42068c589a06b32b6850b57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_291132d6ede22f3d6080a3572176289c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_77f71c61c8df811ff048b50dfeae8a1a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dee17f62281cf27d278caa6c3a85163e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77f71c61c8df811ff048b50dfeae8a1a
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4e55d2ac38122cda74e7ef160cc42409(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bad7e9758c4cc61d2b52274fb09db514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e55d2ac38122cda74e7ef160cc42409
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_50c34f9f963ba37f9f6f13a2d20f7f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bcc5ca1b9c297dd70a8dae468da8c4da
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_44a4c9a5aaa6c3c98759dc9d98d5b1d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_345949db6e2f86cf87f73d7643675659
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_034a7f2c6debab4fb65284abdb87f128(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25f42256d4bd85fa0efa00ad3016213b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_034a7f2c6debab4fb65284abdb87f128
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3abab7ee2c4cece9c08d2029aed4fff7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b3ebaf1170cda955e296323cfc1e19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3abab7ee2c4cece9c08d2029aed4fff7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4c864fb798dd63de2c83fd68908be7ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 240, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7aeb9f5194de6660230c730f61fc8a2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c864fb798dd63de2c83fd68908be7ac
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a8d6ae70a2616fceb22add88a1d90d57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 60, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a014986dd8db8a2130bc254a450d79a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8d6ae70a2616fceb22add88a1d90d57
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_047551f87f887a7411492bca4626a69a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 672, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cfa5529b448cacfae2a5d065d62ce40d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_047551f87f887a7411492bca4626a69a
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_875db8fdbf0bdc24f3b2bf3327182c04(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 44, 44], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_685c54f9bffb2cec015f2082b7722db3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_875db8fdbf0bdc24f3b2bf3327182c04
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d2a57b9647c7748a46bae0866bf0e2d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30301bef94c58095d99b02a1fd61bee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2a57b9647c7748a46bae0866bf0e2d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9d5b6b50fe7cb0172f7c91d46de6d1f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 240, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_780dbbd809d246db36122f52888abe09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d5b6b50fe7cb0172f7c91d46de6d1f9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_fdeb49fe6d7599755fbaf5d7e10d5264(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1152, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e4d8e38015e6e8087f28150d95e4c9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdeb49fe6d7599755fbaf5d7e10d5264
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ca31d5f0edd58a23aaf9a4d4bca4f147(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 23, 23], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21fc8f5cd96324036cb92697e975fb02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca31d5f0edd58a23aaf9a4d4bca4f147
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ae9a6313ff53d405ebc3ba2fd9b72575(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 23, 23], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_58fa7b1fc45b117b53a20886ae8e323b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae9a6313ff53d405ebc3ba2fd9b72575
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7184ba6c0d4d425300ebc325fc0962f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 23, 23], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2f90abc172ed495f7d6dc08458689ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7184ba6c0d4d425300ebc325fc0962f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8de50775b24b0a14fa6742c59ef83b9f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 23, 35], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c29b146cb1a7de743ed6d59c003c0fc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8de50775b24b0a14fa6742c59ef83b9f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5b41485c545c79ca7205d6aeff5c3873(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b40cd1dfd22b9d95877a5e17c4311e7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b41485c545c79ca7205d6aeff5c3873
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9a627762b1984605fc27e1a1cecf8c66(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 336, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd2796ccaeabe5a464f25f80be43e383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a627762b1984605fc27e1a1cecf8c66
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_501629735c4399a81eb85dbb05bb7aa2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 2048, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6800ff04c6a512b0929fb6affd5a850c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_501629735c4399a81eb85dbb05bb7aa2
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_58d5812478f740f84d836589c562591c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16, 80, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34ab27bea144e67c8cc98a85e3c4c5e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58d5812478f740f84d836589c562591c
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_24685147d8fd7397d06117de9ab77631(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 336, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02f739082314c5b3332fb52da5c1796e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24685147d8fd7397d06117de9ab77631
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1714f7329db648fc62e4f1aa44548828(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 44, 48, 48], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e329e42c3568486d3a5c11e91a0871ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1714f7329db648fc62e4f1aa44548828
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_83344d003b0b602c1f09d9af045bd831(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9603c8f3e4eb49a14ec119ac7bfaad6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83344d003b0b602c1f09d9af045bd831
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_79d9f5946b8bbd964faf93ac37e430ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 11, 11], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a316f0759202051abff4173a5ce51192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79d9f5946b8bbd964faf93ac37e430ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8a30032fdda01fbe8233da281e31edaf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1024, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_794c50838f5f711e84ae78880741ec72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a30032fdda01fbe8233da281e31edaf
        def get_inputs(self):
            return [
                paddle.uniform([10, 1024, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1f0a686ca44e051e7d5cefee4e2a10a6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf01563bf93a530a14648f80d288c4a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f0a686ca44e051e7d5cefee4e2a10a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_01bc9d550541b37ef7e5f459ce2ba9f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c1e269a6b706375d05bb9a62792e247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01bc9d550541b37ef7e5f459ce2ba9f5
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_12c6ee2614f08e5f92b2224df084cbc0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09df4d5e57e5604190c8ec90ce20142c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12c6ee2614f08e5f92b2224df084cbc0
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bbbfaa2147b08b39a81ab5083be25db7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94bea63da60c0f9e3ecbd3a2f0620809(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbfaa2147b08b39a81ab5083be25db7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_91d3cd844cc79fc05435278c24a20a51(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 64, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae0dcb1d17cd7264a1fb45bdcc66093e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91d3cd844cc79fc05435278c24a20a51
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b60acc7a309801c81f0db6194ca7185e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd289203279f0a091e599332a81312f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b60acc7a309801c81f0db6194ca7185e
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7574c55e7475a3dd390624c1f8f7780c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_687a7e3c89f89c8820ec2b77a49462a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7574c55e7475a3dd390624c1f8f7780c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_65cade4334645ec5fc704074448f5cce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_033adadba42c588974c78b415665c7ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65cade4334645ec5fc704074448f5cce
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bda335620af46cc1107bbb9a72ba27f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd2316b098f17ffad79a34b58bde91d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bda335620af46cc1107bbb9a72ba27f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b6b2cdd05eb18fa688ab4d96087f3aec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6aff686362a0159ba9409930e9ec7853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6b2cdd05eb18fa688ab4d96087f3aec
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2b121e474934dc75ad01ad27aa9e0dc8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 240, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e92de994be933bf959384741d934471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b121e474934dc75ad01ad27aa9e0dc8
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ad5060bac3caab0c9dfcf144e582765a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f10457eb5845bf825c1aed62ef4cd31d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad5060bac3caab0c9dfcf144e582765a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3362c94b5c7939b2490d02335c2887c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5baa8973a32132b035283b688723bc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3362c94b5c7939b2490d02335c2887c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_adf6d5616999dbc1fcbb6954481a3edb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_899db4e7cb2553f8576db744eabf0c56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adf6d5616999dbc1fcbb6954481a3edb
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_51040d22cbb069ac060a3ec511d81a53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 60, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cc1963643f68cda697500f4336fadfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51040d22cbb069ac060a3ec511d81a53
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_06c7db145e1cb41aa5a67bf4c31f2a26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 128, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a45636a5fa7e764813c5e2adbaf51bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06c7db145e1cb41aa5a67bf4c31f2a26
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bbcb74a54f76e933edc40ac72fe506e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 64, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ec465b57c2bed24b7684aa3213e03f64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbcb74a54f76e933edc40ac72fe506e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_168b3071ca3121aa2d2ab97abf506bf7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be4e55969568153c1ef58faa051a5a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_168b3071ca3121aa2d2ab97abf506bf7
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d0dfe1c47cca1ea12e46007efe088839(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3aafac2ef3056dc2d2871298f01129a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0dfe1c47cca1ea12e46007efe088839
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f3ad5a1cac8e036d4b14c54225ddb7e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16, 128, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7c7de3e1929cfbd8d6436ac17703d14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3ad5a1cac8e036d4b14c54225ddb7e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e98b2b134a32db9c1b4e2199b5b8e93a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 64, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_645ae5b95301cd250b8cd6ad5bd26d0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e98b2b134a32db9c1b4e2199b5b8e93a
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7c0614234d1d375cecd74af655ec3afa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcc49010fc3b1930cbe3dd548ee09ddf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c0614234d1d375cecd74af655ec3afa
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9608d2f0864fcb057c5c2fe5b96ad167(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f472efddeb36998dc15053851b834072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9608d2f0864fcb057c5c2fe5b96ad167
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0db377a27327a51a47d86147b781c73b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 34, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_752a19a959db977dc03e05b1760378d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0db377a27327a51a47d86147b781c73b
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_db48d692b60cdfde517b08a9c5759fef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 60, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6a5136bd0a5b9eb73ce667e124ed806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db48d692b60cdfde517b08a9c5759fef
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b0e4bd693d0ff659623ae964d538642d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e30d906d8d1e391b123b21221a96a0c
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_cd6121a5d34762e32658f989474bf5f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 30, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f9f027dcb5cd2dd4a0e82daec3d7d51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd6121a5d34762e32658f989474bf5f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6e2455db072facfe08e0fcff4657875f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 22, 22], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_65ee6162c791900d11d57aad132310cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e2455db072facfe08e0fcff4657875f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4b054a0e850fd0fdcd3bccb6beca25d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28c85753409cd8ea2a04122a1142d0d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b054a0e850fd0fdcd3bccb6beca25d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7abd1db9b83c74afb1cf740020a86871(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 320, 22, 22], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa25ba3f642a6c6d11c7425721886997(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7abd1db9b83c74afb1cf740020a86871
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7d4daffb840386b7af8afa6ead197ae8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 872, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1118502af7c0a8746cb9d066df7ea2df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d4daffb840386b7af8afa6ead197ae8
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f23222f758b339427033c67848bd0b76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 100, 18, 18], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83c615d1a2c2c7f5b131b6a57ab07db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f23222f758b339427033c67848bd0b76
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5d7d0d4daa8cccd97adfebe013eabd69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f1b7926de13ee55892f06b58fb7b5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d7d0d4daa8cccd97adfebe013eabd69
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0fe3cd8f58a215c1d121bcb0431fe029(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 22, 22], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c92c8ac5ed89c4c72d15206694a7de8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fe3cd8f58a215c1d121bcb0431fe029
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e66be30064f9728ee739e24f143d1ae5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c90f7606acb0acbce7cca15b9d05c5db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e66be30064f9728ee739e24f143d1ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7a2678596f5cac5f7ee4b75cbc3d3cc1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15ba138de9896586f2fa41c671bc2218(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a2678596f5cac5f7ee4b75cbc3d3cc1
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_32c3bf4a8148b25e84a1119bc3dc756d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_afa92a86b5a8f39684bf3cf39396ee1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32c3bf4a8148b25e84a1119bc3dc756d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_cdc672bbc8c9ac696ddbad5cbeeae5a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b8277d41b88bda21f3e5bf7bb0bc937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdc672bbc8c9ac696ddbad5cbeeae5a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_67032e7c8d7550668fd6ff51c2ee2a1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 336, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e90b683c9226736eb1854cfd190ed21c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67032e7c8d7550668fd6ff51c2ee2a1b
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_013c57745f1db8dd7b8ae9fa392ef4cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 44, 44], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ea7385fc04f3ebe2274adeab1652e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_013c57745f1db8dd7b8ae9fa392ef4cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bcfb841deb722a4c02c1c6e2173d8a3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ecfb621fb19fd745e92b0e56b2583ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bcfb841deb722a4c02c1c6e2173d8a3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1f2a38f94b404ca5a0e993c63e3e4c10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 11, 11], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61ae1c880d46eaea37d8f9409afffbb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f2a38f94b404ca5a0e993c63e3e4c10
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b6ee549d5b41ba42ef19c5b7021e9f38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 768, 1, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0e32aea18cb7196fed139a050890fcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6ee549d5b41ba42ef19c5b7021e9f38
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9d3753b2a358400d1d079e3ccb413668(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 320, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a55d0f6d724e0124155a1f86735bdbf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d3753b2a358400d1d079e3ccb413668
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_adcb2da5989c84c211784642682ee667(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 38, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5bf7744a2bb2a77d28dbe86cab3731ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adcb2da5989c84c211784642682ee667
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_dd431f7af3d67f7ff3c07c2c6b0d4dc4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 19, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23361f6bd12d1c2430101e72b76f386d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd431f7af3d67f7ff3c07c2c6b0d4dc4
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_995f20e3cbd7615294ebbc89883ec0ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10e51da8556dc414cf77912d04e7e694(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_995f20e3cbd7615294ebbc89883ec0ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_af72c244298f32836cf948dbd75c9b14(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 22, 22], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4cd3bd1fa63385cc58f55527f18bf83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af72c244298f32836cf948dbd75c9b14
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_90b8be9f8058fe8d420c2632cf8fd80f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 240, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_044caa740a1140e33132b3388593e217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90b8be9f8058fe8d420c2632cf8fd80f
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d31a096f4fb49422008cd4267d698dee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5400d15f3526567033e8df01f0a6b6f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31a096f4fb49422008cd4267d698dee
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c08070ed0380fd9be9de4c6de3fc6271(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 576, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8759eb8582a3f69f19ae0e2fe415bfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c08070ed0380fd9be9de4c6de3fc6271
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2ee7088fd63366405b33eb50bdb18e1f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 480, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c86e75176fa765d263d46a88da976c88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ee7088fd63366405b33eb50bdb18e1f
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3dcdb7ae8f1cdb189a2ab600102673ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d6b4212c7c55ef246a58ffaa81694e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dcdb7ae8f1cdb189a2ab600102673ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ed633a10a400b78d7b5cb90dfa4e2e7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_315741e2a70d3cff695ce63b2bdc4188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed633a10a400b78d7b5cb90dfa4e2e7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d4deb91b7e8f0a0b81937b552bd6921b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_164f25ead8fffdd8aab5695d0dcf7a93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4deb91b7e8f0a0b81937b552bd6921b
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4eb061fb7c7ed02b38bb418a3faae4c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 320, 8, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_072948deeb2590fe40ca6d01f2657782(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4eb061fb7c7ed02b38bb418a3faae4c5
        def get_inputs(self):
            return [
                paddle.uniform([11, 320, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_52b068e0cbb8ac78af567d27d93139fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 17, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b9a6c5da3c7ad3b3e970e5b17a40f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52b068e0cbb8ac78af567d27d93139fd
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3926e8c047c57a7436f0ee0758ca61e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 109, 109], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da9b177e80ea68324955b9bfebded909(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3926e8c047c57a7436f0ee0758ca61e5
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d50c57f8de3a05b36b1f415768cc3577(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 256, 54, 54], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c865475e54d10e6aadf347dc5f79947(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d50c57f8de3a05b36b1f415768cc3577
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f2a341a9923b67c340fb6f03d4ceddf0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 512, 26, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6ad9da7223ee7712a0a9fb01c239208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2a341a9923b67c340fb6f03d4ceddf0
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f41834fdc9dcc1bca55a5358c4fe7a74(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1000, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac869a2a022eda057e631f6c4876a35a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f41834fdc9dcc1bca55a5358c4fe7a74
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_88176a4407658a083e91efb8caa3c149(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 26, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_432c43f5bf7e65375df47a1de81895c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88176a4407658a083e91efb8caa3c149
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_72897f6ce335324f9bb91d483168e3fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e39533195625f1b364c8ee4cc5e73ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72897f6ce335324f9bb91d483168e3fd
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8a28981afe2a41c8fe1efc2a87f61ec7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 336, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_547dce03dfd9e1a51ef293109ee5c22b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a28981afe2a41c8fe1efc2a87f61ec7
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c2971299904d668880fd159e1e32b3fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 8, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63e452a5e324907545dab16f04906690(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2971299904d668880fd159e1e32b3fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b82ce5a160c434aba76191743bcd9b91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 88, 88], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c37e0384cc227a8272747e01dd8eda4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b82ce5a160c434aba76191743bcd9b91
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_616258614f108268189bfc7134cc786c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 30, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bee024f4a3c8593ed09cef61881d3c50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_616258614f108268189bfc7134cc786c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_32075df2cb0deb0b050d82c8e67ad486(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 144, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67f8898e7aa119a3332e31782886b18c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32075df2cb0deb0b050d82c8e67ad486
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_cad0d326839f5eab39921b7c1e554323(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 2048, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_798ea20813d74100ceb3d92f3e70ab0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cad0d326839f5eab39921b7c1e554323
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_37eb08f64665da1b8c5bff9d896cb4ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ff97fdc23489ef84b4ef5129d2c25a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37eb08f64665da1b8c5bff9d896cb4ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_276424b34690a39dda716e1845c03b9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff07253ca65017c6c281305c4f7020d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_276424b34690a39dda716e1845c03b9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6cb14d005315dd077eeb330600a3b573(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 36, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b74ae9664fb69ca975eb37e118125ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cb14d005315dd077eeb330600a3b573
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_780dbbd809d246db36122f52888abe09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d5b6b50fe7cb0172f7c91d46de6d1f9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4bfb10d5fd05a1096272c17206d87de3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1280, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_796a0919d1f42db17cb807729cc6d80c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bfb10d5fd05a1096272c17206d87de3
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f59c2eb6e58b1d2f92358bb3f86598be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 96, 109, 109], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af693f6e1d4ed33a9d69cf0cff261d3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f59c2eb6e58b1d2f92358bb3f86598be
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_31677538e4e8a9197d492412edc06aad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 256, 54, 54], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8461f2df656a9dffb5ab1b10d4f17612(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31677538e4e8a9197d492412edc06aad
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7400507cf3f4c6f3bf6d94798188b9ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 512, 26, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4923cee1aff88a5757a12305d4c3b0ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7400507cf3f4c6f3bf6d94798188b9ef
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2c27dc3c3ff8528f07e4c4b6db76cbe4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1000, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80edbdd5b15d3cc05d583c0e8a6184b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c27dc3c3ff8528f07e4c4b6db76cbe4
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5b4213ce02be5e5948828fd628e2b6c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_165ea1e67d5c866e5a2fbce61082a3d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b4213ce02be5e5948828fd628e2b6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4dc415eaa4102d2614f6931d1145a7d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1257fefbec40a1af78763732b4e8fe5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dc415eaa4102d2614f6931d1145a7d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8c9ffdfc8f0d7ed46264faa4740faaaa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d08d5ddb94e3d1b520e5184a39d3b9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c9ffdfc8f0d7ed46264faa4740faaaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_86b8a43715a16b5ae8c69fd690ec2e13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 144, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ae7e735482410d6cca03c62249423f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86b8a43715a16b5ae8c69fd690ec2e13
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_aa32c201bde5bae7ecade2a70c0d3745(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3f1bf84f8a279c60d7fbce2e883a971(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa32c201bde5bae7ecade2a70c0d3745
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c3f1bf84f8a279c60d7fbce2e883a971(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa32c201bde5bae7ecade2a70c0d3745
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b0a18dec5e2006c61312422957870cfa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3b8c03477b2d1cd3606845224a61068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0a18dec5e2006c61312422957870cfa
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a3b8c03477b2d1cd3606845224a61068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0a18dec5e2006c61312422957870cfa
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3fd5642c392a4acc481137d58a814d66(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c27c08ddffa0bf15f601b89a19100e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd5642c392a4acc481137d58a814d66
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c27c08ddffa0bf15f601b89a19100e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd5642c392a4acc481137d58a814d66
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3a71a6365216e27f41b0044f68170217(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d7999365472d0b3d38d3fd7800f7672(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a71a6365216e27f41b0044f68170217
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3d7999365472d0b3d38d3fd7800f7672(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a71a6365216e27f41b0044f68170217
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_92985df1e2ff92ec53f4632937d97063(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 672, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf1c6ef7f3c9e3acea6b61ef68c6ebe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92985df1e2ff92ec53f4632937d97063
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_15db0cc3bce3e1c4475e3acc35fd18c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b88ec7faae9715b8c44cbfa8c52ae7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15db0cc3bce3e1c4475e3acc35fd18c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9b88ec7faae9715b8c44cbfa8c52ae7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15db0cc3bce3e1c4475e3acc35fd18c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_688a3786dda751c2d76e032447799c81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c5caff3637714cd1f51632fe8e2a694e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_688a3786dda751c2d76e032447799c81
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c5caff3637714cd1f51632fe8e2a694e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_688a3786dda751c2d76e032447799c81
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_cdbbfb3289be7b95852ddffb76eea58f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b24fd4c98777d7e924ffddfe633edc38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdbbfb3289be7b95852ddffb76eea58f
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b24fd4c98777d7e924ffddfe633edc38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdbbfb3289be7b95852ddffb76eea58f
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_66f4e77fa7a6026728e1979419e3eff4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc3bcc249578bcb0e92da03f1c9b667c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66f4e77fa7a6026728e1979419e3eff4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bc3bcc249578bcb0e92da03f1c9b667c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66f4e77fa7a6026728e1979419e3eff4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_336549f26c6bfa7a57563e7c81d977ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 480, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51cc1a6358103859429d0235e6cb23a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_336549f26c6bfa7a57563e7c81d977ab
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_11473ae5093296e41d9454150637fae6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 60, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c97c09456634a1e588921d7748d168b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11473ae5093296e41d9454150637fae6
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c86e75176fa765d263d46a88da976c88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ee7088fd63366405b33eb50bdb18e1f
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8f5f5dcf39dcd56ba5cad3c39410596b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 336, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8c71ce0f8ede9a55a417bd643a548be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f5f5dcf39dcd56ba5cad3c39410596b
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_fa425dc885fae53860bf568744f730ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1152, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2ef5cb394826e70f8c7ebf540a4a970(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa425dc885fae53860bf568744f730ef
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_db55f73f074ee3d56d9b738de04a71c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 32, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c19b8ff2ee0712a453351cccdb308e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db55f73f074ee3d56d9b738de04a71c2
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1321a25e99648cbda88be343d5f33dd1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 240, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81027abc497232634f6b7b28bb5d655d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1321a25e99648cbda88be343d5f33dd1
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_65fd54b06c1061a1a7754264d8aaf213(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 336, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19e59b2b68cad5d83ad96083fad3a6d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65fd54b06c1061a1a7754264d8aaf213
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1c9ffdcebbf8a20f8432b620811be02c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 480, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ec6e7c97eaed2d6c0683059497942d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c9ffdcebbf8a20f8432b620811be02c
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e6db5c6ed2441eabb8dbebdbb1412499(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [14, 14]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a866b7d89dbdeb24f0e1829874261b9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6db5c6ed2441eabb8dbebdbb1412499
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a866b7d89dbdeb24f0e1829874261b9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6db5c6ed2441eabb8dbebdbb1412499
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e2d7d086277f415d61d05b9ea840c9cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [14, 14]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f523c547df2c3b6160d12fd46ed9d1ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2d7d086277f415d61d05b9ea840c9cd
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f523c547df2c3b6160d12fd46ed9d1ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2d7d086277f415d61d05b9ea840c9cd
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_dd84bda610653516054379d8fade7331(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [14, 14]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e09c8441d21eb9748627666415ce7d91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd84bda610653516054379d8fade7331
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e09c8441d21eb9748627666415ce7d91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd84bda610653516054379d8fade7331
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_484ca0490b99d77bd4a2fa1afece179f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [14, 14]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_144a59ee9d26c1563d09bb12aa24c897(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_484ca0490b99d77bd4a2fa1afece179f
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_144a59ee9d26c1563d09bb12aa24c897(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_484ca0490b99d77bd4a2fa1afece179f
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e885eb2a0c5430f8bdb3b5bb5bd3c1fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 300, 300], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c1dcca8cfc6b0302b30f8d3fbbb24ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e885eb2a0c5430f8bdb3b5bb5bd3c1fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1e1c2dcd6c174236e98e01ea4a5c885a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 150, 150], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d32f2cd13ae2beddd037f18efa30dbe0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e1c2dcd6c174236e98e01ea4a5c885a
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_629103e6ccaf36f50e36843d00f6eb38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 75, 75], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_31b3e0d4b8e2565661d49190861cc805(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_629103e6ccaf36f50e36843d00f6eb38
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b621073e90d8d615091e66d62076ff17(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aba0f04c0233456b3e6fd96287c6db62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b621073e90d8d615091e66d62076ff17
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e72daf08c1eaa99897a03990815003e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 19, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_138fbfb7e32e1824fc0b7b793e1ac01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e72daf08c1eaa99897a03990815003e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7c9d67a4450fb6c9695f0b255026bf16(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1536, 8, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e6e3b50826f7ebbb0c073e8a93444337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c9d67a4450fb6c9695f0b255026bf16
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3360224952ac0dd76f13df1309375361(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4861df04c97ae9676fc50efb50e7023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3360224952ac0dd76f13df1309375361
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_61ce059e4f2895e797b138081553f1cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 60, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e06ddeca29d34bd4e92d1e4cbdae121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61ce059e4f2895e797b138081553f1cb
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2c19a60ce4bf59377958f8397544e443(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73b389be7ddeedc07ebc9d2754e96685(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c19a60ce4bf59377958f8397544e443
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_89783d2e5ea5ebfe1127c10a397b528d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fa0c6fcec0b0af96cc7e5287d46b4e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89783d2e5ea5ebfe1127c10a397b528d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ec5344da400910fa0024c7e87f0d5fbf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b645c4dfa758178b3fe9303d8cc861e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec5344da400910fa0024c7e87f0d5fbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bb7cbbb552f27a864b708d44755e98e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8cfcafaff66b2cc0245e36854262311(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb7cbbb552f27a864b708d44755e98e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8be58fbe809fb379ea2ec32fba1780fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 240, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f07438e1baac094268b07381169eb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8be58fbe809fb379ea2ec32fba1780fc
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_00431678fe3061d4501f799b2fddf932(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1536, 8, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd8c73acb490173d030c1497b7b7b9ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00431678fe3061d4501f799b2fddf932
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_34e2060706c258ae075a8d362cd4f938(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 76, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7b544070cf6422739485f8e3d51ef7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34e2060706c258ae075a8d362cd4f938
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_752b75acad4b3376c7755702cb48b4ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b25e48c967ca4ad5f36b55bf00023b6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_752b75acad4b3376c7755702cb48b4ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a44851e10584f1395e8274659ed4fd9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 20, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc21cbdc2d55a7bf34901088c8144c54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a44851e10584f1395e8274659ed4fd9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_713231105c63f2b7f56fdd2408dcc06b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NHWC', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 7, 7, 2048], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37606d696e548cbeee127a7d325a11a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_713231105c63f2b7f56fdd2408dcc06b
        def get_inputs(self):
            return [
                paddle.uniform([22, 7, 7, 2048], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f6f337489bd23d608d097c03bfbaa4a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c5ac57d01cf12e2c386d15be7627e576(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6f337489bd23d608d097c03bfbaa4a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f955953549efc366ae6042f8c630b410(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a9bbfb078d2c18dc1666933d57ba843(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f955953549efc366ae6042f8c630b410
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_306f07f9f27125a95c6317c2906e64b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b962a95057dd6d771f2c05afdd614b08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_306f07f9f27125a95c6317c2906e64b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c354a864dda7eae9e4ee232de48c26f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 44, 44], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_435355dbadfa1a2276308485f416f517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c354a864dda7eae9e4ee232de48c26f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ec3a4e95b29c8ac0713a07c2c2f7dfec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7367bd45b1caca5c952752638810f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec3a4e95b29c8ac0713a07c2c2f7dfec
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a530b22a1873d64700545e943ccce46c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b21c255ee5c4049ab28185549360222(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a530b22a1873d64700545e943ccce46c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c7dee371dfeb436a99733421e728dcaa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dcb4aea3d8f868f31d6f28ae1f8e22f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7dee371dfeb436a99733421e728dcaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_494bb28b3d416310e1194fe7b7248668(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_940d1d1868bafce972686975db270376(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_494bb28b3d416310e1194fe7b7248668
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bf1c6ef7f3c9e3acea6b61ef68c6ebe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92985df1e2ff92ec53f4632937d97063
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d6f983201367d043438afb831dcbd98e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 36, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c5fb2db5444b28037dd35b2c4283001c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6f983201367d043438afb831dcbd98e
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_918e1ea267b6a915d6304120fdd55986(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bffe59b7cbe079a5cecf50ce8fbfed6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_918e1ea267b6a915d6304120fdd55986
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_38f5e5c019cd0a510c9b22adc697a29c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 64, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5cec0be930e98dd3968fb5edd5fef63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38f5e5c019cd0a510c9b22adc697a29c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f94360a718c6b775638b37e07af123d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 144, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89b4de8e326deb70afabd7374ad5fdd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f94360a718c6b775638b37e07af123d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1e653b27426b73c19abbc2738a74a188(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16, 128, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17a7d82498f0896fa8f1d627d748ce1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e653b27426b73c19abbc2738a74a188
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ce2289b599496ec8fb1b8d437ca8b6f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 21, 21], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e64808a5991dae8f7f4984a65a6ddaeb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce2289b599496ec8fb1b8d437ca8b6f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b8403751b6d45b83d07ccf23d687474c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 21, 21], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbfb1888594e605b630fbf034c2215cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8403751b6d45b83d07ccf23d687474c
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_00fc59d7a788ff029b0b83b38adcd63a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 21, 21], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd034f0bd7dd52156f8d4304676a959e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00fc59d7a788ff029b0b83b38adcd63a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_763b594f9cf2b0161d36ec7717f43f2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 109, 109], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad76a2b583cc34c095be34ae1165e044(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_763b594f9cf2b0161d36ec7717f43f2e
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4e68ea7b93e5d677879ab9539c2b7ebd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 256, 54, 54], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdb4c08f1adf0be9ba8c92188849f0f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e68ea7b93e5d677879ab9539c2b7ebd
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1d3378407e9c53cdad89822ca4caf683(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 512, 26, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_970323d8297a04a63c97df0bb5e23d76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d3378407e9c53cdad89822ca4caf683
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3037ca6958520eb6dfb87aae4850db26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1000, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_190a04bc4a9817b511557ce39cc95ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3037ca6958520eb6dfb87aae4850db26
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b8659658c04ff330503d6a6de20cdf35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5df94c63ea280310088eaa4fda64d002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8659658c04ff330503d6a6de20cdf35
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4adc82b5afbfcb93310eea8385068699(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 15, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6bfda2f467512fada48e3fde5675d0d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4adc82b5afbfcb93310eea8385068699
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2f54d05d5c41bf52c385461d7b7dfd71(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 60, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_abb4e2d8187784b3dc01d7aad0c3ac73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f54d05d5c41bf52c385461d7b7dfd71
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_33dec76b2c46fba887456265200f9576(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 15, 25], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_294ba6b39e56cf86c49442dd792e8f3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33dec76b2c46fba887456265200f9576
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9bd30b675d112c69a986f6d14b77a615(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b03cecd4756d18033d6d0bddadedadf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9bd30b675d112c69a986f6d14b77a615
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2b631416681e1ddd5a955c30b7ec50e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 400, 22, 22], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_817c324d34ee0878a27895f12708f56d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b631416681e1ddd5a955c30b7ec50e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_80b18271eade375043ef82385ce82247(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4eeb19b09320d8d29758eee3ab563eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80b18271eade375043ef82385ce82247
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1a52a00d38ed18963c11cdb22a9f4dab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 17, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57d13642cc10acb7b0fafde1370e55e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a52a00d38ed18963c11cdb22a9f4dab
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_76f2e9f01ab7a64cc55d292abba0a6c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 336, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f118728fee745f3da2d8a88e8d428a35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76f2e9f01ab7a64cc55d292abba0a6c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_46e3f218c6cebaf0e22ded1d1e8afacf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 240, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90ef020de744fdb3b674d9772bcf8a8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46e3f218c6cebaf0e22ded1d1e8afacf
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ed14f70209128babb6a2c100772709ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_caffa56c7593af232b8ed4f7e17222e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed14f70209128babb6a2c100772709ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7ecd14e322717b17f2c5d376456f4d5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc6645dc10e7370887983929d17e42ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ecd14e322717b17f2c5d376456f4d5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9f504d6c8a8862ffa1be47d8e6c24cad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e599acc7d632b4f8faa8a17e3e6b1695(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f504d6c8a8862ffa1be47d8e6c24cad
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_806589f344369a16cb673a60ffdc7583(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 18, 18], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09b549d15b6101ae46bf4f324e5ef8dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_806589f344369a16cb673a60ffdc7583
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_34aa2270de2154f89fcbb843d4ed96df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 38, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83d70e748c3af8ef2cc126ee61be8c27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34aa2270de2154f89fcbb843d4ed96df
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 38, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_61ddab5837181054a989b450897ef106(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dee35d08c80484eafbc4565d1ed1fad4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61ddab5837181054a989b450897ef106
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0ea757a3b3bf9ed4ac80da58ad5ae8b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 9, 9], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f74bfcf4d650084bdad383d34181c5be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ea757a3b3bf9ed4ac80da58ad5ae8b2
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_45c71a3c5c940b4a4d7afdb18cbda27a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be588e54adf79e0ef749e46671461054(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45c71a3c5c940b4a4d7afdb18cbda27a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a97e61fe376f0cb7306f222a4ca4e93c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a95e09a2e3b5e27b30a3300941b216d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97e61fe376f0cb7306f222a4ca4e93c
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_89b921bffe6e12058fa6f4e79cff0a78(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 88, 88], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee3e992c91c8ab5e8e088bf2e7a48978(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89b921bffe6e12058fa6f4e79cff0a78
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_00ec3f6bf38150b59b3ea7039cb7bd31(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2048, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_13b707476f4bbea9548b0290ab5992c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00ec3f6bf38150b59b3ea7039cb7bd31
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0448a3ac0a8ff18007d0b8d1f6c266f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2048, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7483ae6914e9756a0d8dcfbf8100a434(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0448a3ac0a8ff18007d0b8d1f6c266f9
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a12544fc5949b24a4c234b10d56cfa4d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 320, 8, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf2a418bbc7204fe02fb7438bd3975c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a12544fc5949b24a4c234b10d56cfa4d
        def get_inputs(self):
            return [
                paddle.uniform([43, 320, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8774b9fa309d6b0e955fa4b416610896(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 336, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54cd3771583be0246d14ae2b84b8ea15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8774b9fa309d6b0e955fa4b416610896
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6176d12cc74f128738bf04e78a7bc69d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 32, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3201a4257f0e7340c698ec396d914d72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6176d12cc74f128738bf04e78a7bc69d
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0f154cfdc69408dd5d312be2c9dc2e2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 144, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0dc85e09a5487df12101636b4a0df58c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f154cfdc69408dd5d312be2c9dc2e2e
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e9af228d8d1ae6f1fcbb9ca1244a7704(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72cf74a8efedaa4db056847b136547b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9af228d8d1ae6f1fcbb9ca1244a7704
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_648fb2ee8677850eae1546097e239e17(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 52, 52], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14bce0ba617518ffd2a29b3b09757c6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_648fb2ee8677850eae1546097e239e17
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_092e08f380be1287923c60b5f17851f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 44, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8811c1c7c7228837787a1e55e8042d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092e08f380be1287923c60b5f17851f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2e317bab6aef7f3270ec3f9d82c6e793(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d5df4592aaa9a319cb443edddfc96b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e317bab6aef7f3270ec3f9d82c6e793
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5546204a6391c1eebb764c016e5e30fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1024, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8bc083d8fc8ee3e80c23ba1aadc17532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5546204a6391c1eebb764c016e5e30fa
        def get_inputs(self):
            return [
                paddle.uniform([22, 1024, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ac739b4d0db44bef1ac8d89837c9aef4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 400, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75e9f56ce7f21df881a310d0588f7624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac739b4d0db44bef1ac8d89837c9aef4
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_afc8f1cec6a3fb2462d38e9d12d5476c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 56, 60, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44b115863e82fb8258dd2a94ed27b435(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc8f1cec6a3fb2462d38e9d12d5476c
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4eeb19b09320d8d29758eee3ab563eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80b18271eade375043ef82385ce82247
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_35a94357ad63483c901c198139fc2b92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a45e4aa556497346b79e92a2ec1dbf3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35a94357ad63483c901c198139fc2b92
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3e46dfd2d8cea03e5019ffda6777d571(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 15, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd693fefed1b0ce9686cc9c809362662(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e46dfd2d8cea03e5019ffda6777d571
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6e639cdebd5b32fae01490cb11271492(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 26, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb0b6e395649dbaa3f541ded63d1558b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e639cdebd5b32fae01490cb11271492
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_932267e04592c0dde941cd0205d5b0a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 30, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e374f4e57cc51cb3b3debd6d2fc452d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_932267e04592c0dde941cd0205d5b0a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_67f8898e7aa119a3332e31782886b18c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32075df2cb0deb0b050d82c8e67ad486
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a7059af088bb81b42de33ae120f819f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 96, 30, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8a182f5455cdda6858fcd004c72be99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7059af088bb81b42de33ae120f819f4
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b772902d54ba48c91ffa99f4ef63d9bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 96, 60, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9e0f41949c8da8330d779c5c21d9bb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b772902d54ba48c91ffa99f4ef63d9bb
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_52627d46843f7cfc3010a11eb11118d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 96, 120, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1841f3ac6dd280dac99a62f077ef62e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52627d46843f7cfc3010a11eb11118d3
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e75ba1abe5e443b246749afaaf799371(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 96, 240, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0961a8e1274a1fff67f286be8520179a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e75ba1abe5e443b246749afaaf799371
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b106d2a0203c6c61f894d90e4bc7e769(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 24, 30, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03cd3dfa0f94f5b8972e8398933c827c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b106d2a0203c6c61f894d90e4bc7e769
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_dffc1aa5fb6cc938b48658ba8c13f206(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 24, 60, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2652a0d5175a64957af1af7c51b5993(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dffc1aa5fb6cc938b48658ba8c13f206
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e6e2ba6f018bc0a0ee40268afc658892(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 24, 120, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5469e209353cd2b5191906f00f95fe71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e2ba6f018bc0a0ee40268afc658892
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0ea805f0aeb9edce8c41fca086c47e8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 24, 240, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a30d94c1437c831a1e2dd5ce89ed6ba5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ea805f0aeb9edce8c41fca086c47e8c
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e2ef5cb394826e70f8c7ebf540a4a970(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa425dc885fae53860bf568744f730ef
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_13ad91f04fa077e7f3431ce84026c067(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18dd63821af0f886294a9099d592478d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13ad91f04fa077e7f3431ce84026c067
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_89b4de8e326deb70afabd7374ad5fdd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f94360a718c6b775638b37e07af123d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_62258058ace5ddd4a45b396fc4c1e94c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 8, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92362e6f6d8cd153166f074406d43b71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62258058ace5ddd4a45b396fc4c1e94c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d01416e8265707eda8fdda9a16e4f363(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6991581ddab9b6c82d19cef9b892be4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d01416e8265707eda8fdda9a16e4f363
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a9e9070082373931347805ba492022fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_812797197bc77a8bf5a16024436aa28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9e9070082373931347805ba492022fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b18e956cf0cfc8a55065ff79e81dbd0a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad00f284107573b0c188e04911e96c17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b18e956cf0cfc8a55065ff79e81dbd0a
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_acafb81a1a4fd110ae0e33456374af25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 768, 1, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e82843cf563d7c10534983d5b4eeef8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acafb81a1a4fd110ae0e33456374af25
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c1b02b39d3c65fa3053d54068efb9db1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 200, 26, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa62169e3e100bea01ca5a954bebd0c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1b02b39d3c65fa3053d54068efb9db1
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0dd9e5b17f35481ca375a98e4372385c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 17, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f4fd793f9a7713dee7a44187d066d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dd9e5b17f35481ca375a98e4372385c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_06fa0826f0ddba37492eb1abd16e8727(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c633555f1190c020e928505e0c593c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fa0826f0ddba37492eb1abd16e8727
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a1bd04282dbf089e586bc82c90b9d5df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 68, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1de561ebf45eadfaf881c2abc388edd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1bd04282dbf089e586bc82c90b9d5df
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9872623e5fe5a1f16c0f8193288ca1fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f89e14c65718c3b9a8754caaf545eb47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9872623e5fe5a1f16c0f8193288ca1fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_70d7efb583eb2953571f7fad31122d03(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bbfe1c3668518641b6bbbb03a498d7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70d7efb583eb2953571f7fad31122d03
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cfa5529b448cacfae2a5d065d62ce40d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_047551f87f887a7411492bca4626a69a
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_705765239f981c25a1221c327e040f28(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [7, 7], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 704, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17df72b9e470581f7a67996392561c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_705765239f981c25a1221c327e040f28
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7aeb9f5194de6660230c730f61fc8a2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c864fb798dd63de2c83fd68908be7ac
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_55fd48bcb028601c15b674cac449d0ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 100, 44, 44], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca9adb69bb312ad39e904367e9a4670a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55fd48bcb028601c15b674cac449d0ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_51d439d061d35632944c755ab9916df8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40ac37cacafdeb28739d18fd751a13ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d439d061d35632944c755ab9916df8
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b9b887038a38a37dbceb4f2fe4bcda32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1248, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01e36704f520d9c7acf4e4cb9f8a1d4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b887038a38a37dbceb4f2fe4bcda32
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c7971e9f1835546dfe841d05e6022d2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 480, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c91db05c8c1b7659aa140d79fdfe5ac7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7971e9f1835546dfe841d05e6022d2e
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ba4e9a30837c715e22079673bc8bdaca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 36, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39255b1f2130e545b585ad7b51656d7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba4e9a30837c715e22079673bc8bdaca
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8f6b411d814bb9c3ee374929f85ef463(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc52f0e33ecd57ea089db50231abf191(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f6b411d814bb9c3ee374929f85ef463
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dc52f0e33ecd57ea089db50231abf191(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f6b411d814bb9c3ee374929f85ef463
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a5538907b7b9a6fbd4a4afbb3b8235a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3cae7111b59b3c7e0c74dd1fc6aa34d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5538907b7b9a6fbd4a4afbb3b8235a5
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3cae7111b59b3c7e0c74dd1fc6aa34d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5538907b7b9a6fbd4a4afbb3b8235a5
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_aac6230b007ce649adee7ce4c6dd75ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca370b95186a1912e774dcc53117862e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac6230b007ce649adee7ce4c6dd75ec
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ca370b95186a1912e774dcc53117862e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac6230b007ce649adee7ce4c6dd75ec
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ec68031d92c8f5a7720738590aefae01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3107ce1e839d0386564d5b42676de3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec68031d92c8f5a7720738590aefae01
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e3107ce1e839d0386564d5b42676de3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec68031d92c8f5a7720738590aefae01
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_cb51fe32dd3d0d87ae859487329f1fa0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 52, 52], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dfbb2cacf0279396c5c6991a476cfce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb51fe32dd3d0d87ae859487329f1fa0
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_29977c1a4db20c214c959c89a7bbb3d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 240, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0316f9268f96c7b128d0486cae593c1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29977c1a4db20c214c959c89a7bbb3d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_319d176b05989ec72499fd8270187ac8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9267b8d8b1837029e690df69ac0bd410(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_319d176b05989ec72499fd8270187ac8
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e88fb4ee9c3551c48bfef58d92d8e1fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fa655b47d0164ef6808822c62bbcb31a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88fb4ee9c3551c48bfef58d92d8e1fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7f4fb79e5a279399ee26e0052455ef59(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b331edcc356b68203d48a5c3ed67c7c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f4fb79e5a279399ee26e0052455ef59
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c7d2a97ee47b493fee0f09e7737d579b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7290ce270108b2c742ca2062ec6c72ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7d2a97ee47b493fee0f09e7737d579b
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4facce0b1cd92c95c32ea42e2403588b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_79a9d2af23d9a95d791c82afb4c94971(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4facce0b1cd92c95c32ea42e2403588b
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1ee887769a517670e4fb2729263a819e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9b5046773dfa9bcf67119c356ebde5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ee887769a517670e4fb2729263a819e
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ec7dd912f692aea7ff9cff85715e5917(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 156, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a6a8c68a2ec10dd1b8c175b6c7b172e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec7dd912f692aea7ff9cff85715e5917
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_39279b14efe88abdb32d939ef0052e09(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 128, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bca3ea41ad4826396493364511bafef3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39279b14efe88abdb32d939ef0052e09
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_80d5e01d744eb061591d7f19784a55e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 64, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6cf5da72ea021e898faac80e7bb64e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80d5e01d744eb061591d7f19784a55e2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bdca4f4d70363161b7d59459c4eb1c5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06c0a490ba02a486e79d68b314f432a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bdca4f4d70363161b7d59459c4eb1c5f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9be7793ab491638c629b96e6857b7889(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6783fc126fcd7f6eddf0155b1152fc98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9be7793ab491638c629b96e6857b7889
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_50c34f9f963ba37f9f6f13a2d20f7f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bcc5ca1b9c297dd70a8dae468da8c4da
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_44a4c9a5aaa6c3c98759dc9d98d5b1d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_345949db6e2f86cf87f73d7643675659
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_01e2f20cbd3dc6ca1cbbcdb2d9aaf72c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df641d38c95d77cc3afe29ef01cc6c7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e13361f02b305602058b2f02da60fb12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a8b6dc5dddade630e140a518668c966
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0fe391bdbec06df2e4b2fef3c5428de8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 200, 44, 44], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa632428c747b1b798b13009f4e05db8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fe391bdbec06df2e4b2fef3c5428de8
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bb3bb83b3f4f1505938268285b01f2ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 320, 9, 9], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ba1df1a9bee1e5f194e71220f859ded(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb3bb83b3f4f1505938268285b01f2ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6531c62854b50ef058e361a18b85de1f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac607834e6fbba35daab3a194d5da0bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6531c62854b50ef058e361a18b85de1f
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1663a2d518a20f8ff7e1db95792c27b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 34, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6fcce82d06701bea1802382a1adc575e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1663a2d518a20f8ff7e1db95792c27b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8b918a9c6e45d36f170d9ec7798c1164(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 672, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0eb0d0894257447d326e487712297ef8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b918a9c6e45d36f170d9ec7798c1164
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b406c88243813cd4e440e2c4727a153e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 872, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f758b41745c953da18f442f2cdb44bac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b406c88243813cd4e440e2c4727a153e
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5d6a67261ccec3a4d6ca2afb1e692ea7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6eef882c3db6f055288bcae31b372c79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d6a67261ccec3a4d6ca2afb1e692ea7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7212de0971471b78551a90238e0b0a64(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 480, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa541a104354b5519fcb34b475d7864a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7212de0971471b78551a90238e0b0a64
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2527be3c5e9927ff17e45796ffb513d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b63cc39f2309ae9c0abc1a6fbaa45380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2527be3c5e9927ff17e45796ffb513d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4eeb19b09320d8d29758eee3ab563eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80b18271eade375043ef82385ce82247
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_085ac4c5c32f4452509399c53700a0bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 480, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f2fe80ec5b5222d442fef77f5686023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_085ac4c5c32f4452509399c53700a0bd
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_afae9d8e872f368ed39a3d7ccc5b1bb4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 36, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_140bf139aecb730f8e412e5d104a5b6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afae9d8e872f368ed39a3d7ccc5b1bb4
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d5b1d9dfd764f598d893f6361392b27b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8460feadcd032f70f7a51e25314869df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5b1d9dfd764f598d893f6361392b27b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_aa631127ecc138af3de351041d41af18(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 15, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8cefc3f0b2656c40022ff63fab07fea5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa631127ecc138af3de351041d41af18
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_46d1932174dd3e7fb640990751404d96(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 15, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_afad933d02b5a179626cd10f5e369363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d1932174dd3e7fb640990751404d96
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d2bbe578611af1d8bcb465814e3243e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 15, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b2dfd0dcc77969453a3fc3e9ab2417e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2bbe578611af1d8bcb465814e3243e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_933b33797e4ba14c675aa8aed12e06d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 68, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce0fb6a348026c77bd3f6b101fa2c441(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933b33797e4ba14c675aa8aed12e06d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e0b5d6d4c969b1cd112f5370e46dc416(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45d880bff0fae2617537e52b1e664934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0b5d6d4c969b1cd112f5370e46dc416
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8ab242818035640074aefaa633cacd99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b60a179dc1b8ac8a9f0f81490a4942b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ab242818035640074aefaa633cacd99
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_033752b9e579015e6219cbe3a211f054(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 24, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_16f0fee760ff8a7c5b52d2a842de8938(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_033752b9e579015e6219cbe3a211f054
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f33ee800fdde652a4811f5308b4ab79e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 38, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44108ab119a8169815733d17fffa2c51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f33ee800fdde652a4811f5308b4ab79e
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bce9cfbe0b5c58f89ab938e9a45e88ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 16, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8ead5d6eb80d2cf9ca92f61cd7e1ca0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bce9cfbe0b5c58f89ab938e9a45e88ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b04fa0b413c87f6c0f819f128482ce4f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc7dca26de870e62e9313112af61dfdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b04fa0b413c87f6c0f819f128482ce4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_50c34f9f963ba37f9f6f13a2d20f7f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bcc5ca1b9c297dd70a8dae468da8c4da
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_44a4c9a5aaa6c3c98759dc9d98d5b1d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_345949db6e2f86cf87f73d7643675659
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_01e2f20cbd3dc6ca1cbbcdb2d9aaf72c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df641d38c95d77cc3afe29ef01cc6c7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ac607834e6fbba35daab3a194d5da0bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6531c62854b50ef058e361a18b85de1f
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_17e556f7de91588e824085b4f36ac229(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_86df2608b6ce13d33628f95f61a28358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17e556f7de91588e824085b4f36ac229
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0eb0d0894257447d326e487712297ef8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b918a9c6e45d36f170d9ec7798c1164
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0dc85e09a5487df12101636b4a0df58c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f154cfdc69408dd5d312be2c9dc2e2e
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_780a95dea62c81611cf2e2ec58e2316b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 48, 48], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ec6d99dd087be13a8983c0c7e1ba5c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_780a95dea62c81611cf2e2ec58e2316b
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_98c17e1ff14df158e53b007d697015af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46dcd824e474e499ae7208d89bc01607(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98c17e1ff14df158e53b007d697015af
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e684969bc4106f1283341fccaed8a071(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e34d59352f7122bcd441eb4e0381a6c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e684969bc4106f1283341fccaed8a071
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_71bc0585def705d9fad88573364ba6d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 336, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_688b247d1089918290adc3ec7b2b28bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71bc0585def705d9fad88573364ba6d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_35319d454b83e43e287a70f791e870c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d00df82a8ce26e13b3dd9afb94afe567(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35319d454b83e43e287a70f791e870c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_21e774d114ef779131c6dc7f1513ffd8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 56, 48, 48], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_589be493d77ad3c5da385163a43ff45e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21e774d114ef779131c6dc7f1513ffd8
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a2d19b8b3750d8f3a2b5c742fffc3286(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 18, 27], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa1d1abc611dde47e0a11190a31df95c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2d19b8b3750d8f3a2b5c742fffc3286
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_067044bff5200e8c7a093be95eb162f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 22, 33], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55b76f127b3490269d4bfa79e925717a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_067044bff5200e8c7a093be95eb162f1
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_46d988d6086664d92ed1c9e9a80fda40(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 21, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c846889a21c049fd5dfc58c81bec104d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46d988d6086664d92ed1c9e9a80fda40
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0a9d8da8b14c1f673739f1919d061849(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96472ac68425d5fac00fde51f8d5a1f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a9d8da8b14c1f673739f1919d061849
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_395303d2458820b0f70f8ec39f869ecb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 25, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8391803e43352d7a0b5b1bda474bb46b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_395303d2458820b0f70f8ec39f869ecb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_cc635b2e730fefd8d1a06daeeee01049(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80d439163ca3374af6134d723fe206c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc635b2e730fefd8d1a06daeeee01049
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_416918192344fe375f4274782076302b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9ad139c489920df746aa2bfc1f24d83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_416918192344fe375f4274782076302b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1139fa22b9be2c634cac76dcccaeeeda(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 10, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aaeb61c13f2730b3f21149de1c515feb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1139fa22b9be2c634cac76dcccaeeeda
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3154dbd670b742363fd255b98c761854(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 17, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd3ca5ab90b0937ccb9b4f938044eb74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3154dbd670b742363fd255b98c761854
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_9ed23c7d2bd12471f5e43db123628649(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 17, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6420938977eee1f820a46dba99e6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ed23c7d2bd12471f5e43db123628649
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_488a74c1ea38db558ede44aa1de88372(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 17, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e78c849bf0016069f1a6facf8c4f9e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_488a74c1ea38db558ede44aa1de88372
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5ec6e7c97eaed2d6c0683059497942d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c9ffdcebbf8a20f8432b620811be02c
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e271e8cf35dab07dc060a1e1907fac97(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 320, 15, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b4cace3031a2d2b78092b596dae2c79a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e271e8cf35dab07dc060a1e1907fac97
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a5d316ae76ef65e9c0b397ecc3ce00c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 76, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_341401b72592e8731a9c57c47b0b13af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5d316ae76ef65e9c0b397ecc3ce00c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e66f28c1f625d5aff2048b51c09c9851(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 19, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4ead824b1e8790cac63422a18b9a29b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e66f28c1f625d5aff2048b51c09c9851
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f434952806b81e61652d051fab13905f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 18, 18], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c220394a669b769bfaf8b94bfa432816(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f434952806b81e61652d051fab13905f
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8a9b9db42818741223f2b2a3787f37a8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23f543e5333ac8082da06c611d483375(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a9b9db42818741223f2b2a3787f37a8
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_eb4530fec95cf75d52da3745df6d95b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 96, 109, 109], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2e5773b6a8f11c032b326e50874a9e46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb4530fec95cf75d52da3745df6d95b0
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1ba5fa81d2ae7a7d790f763cecac8680(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 256, 54, 54], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76cfbc0a840220fad44f722b0c7ae542(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ba5fa81d2ae7a7d790f763cecac8680
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d0227d493171bf4cb56722df73d5a489(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 512, 26, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d637f742eb5b44eba6c29048ddfe3d8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0227d493171bf4cb56722df73d5a489
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_17f04a309a7ae7f005ac091acf139092(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1000, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f68e21338b58fc97c1dcbf0a95488b7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17f04a309a7ae7f005ac091acf139092
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_22a5c7b3dc57650a60ab8f9e4e926bdc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 19, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ddbc5417ec153c02c6df53869ed98f8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22a5c7b3dc57650a60ab8f9e4e926bdc
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d8bac05a5cef0c78986f6219200a9df6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 19, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b8e4f7244a37f556f7d356d706aebb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8bac05a5cef0c78986f6219200a9df6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7bea1ee696922af6d82d72ae0ec507a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 19, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8de7d69a0e84b56c3bd94c7b830ee734(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bea1ee696922af6d82d72ae0ec507a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0e5cfc16ed0f62d35172699ba60a8041(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 200, 18, 18], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0d883be25a92444e768221d7a5da93f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e5cfc16ed0f62d35172699ba60a8041
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_60902d24a10119170066b32192e6a90d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 400, 9, 9], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e538cc12e085c0ccbc9ee41005d8b5d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60902d24a10119170066b32192e6a90d
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3201a4257f0e7340c698ec396d914d72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6176d12cc74f128738bf04e78a7bc69d
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_76fe1318fb6c4467268a443c2fc56ada(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dfb47df12163cfaa8b8c04a3e0d4bdeb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76fe1318fb6c4467268a443c2fc56ada
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a3b8991087ebbff99e19c469a403a2d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 100, 26, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab1af65e04f36554e1ae24591b846da0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3b8991087ebbff99e19c469a403a2d3
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5400d15f3526567033e8df01f0a6b6f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31a096f4fb49422008cd4267d698dee
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4aa35f4f96a550c62c0370ba0028bd3c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 48, 48], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8db0932a344e92129010c7292a72f84a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4aa35f4f96a550c62c0370ba0028bd3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c19b8ff2ee0712a453351cccdb308e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db55f73f074ee3d56d9b738de04a71c2
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_65ee6162c791900d11d57aad132310cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e2455db072facfe08e0fcff4657875f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_34ddf08013a2e82dfe09ee94ad042488(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 88, 88], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc7e4921a46aafdc0eefdf58bc4bc9b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34ddf08013a2e82dfe09ee94ad042488
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f285ee12a957cf6ea4165154dac9969e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 320, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a747d02936516f768ac6f5108359808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f285ee12a957cf6ea4165154dac9969e
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bd199c07c9b285a08d3de98f9d275730(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 48, 48], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_abb31c2f190ce1f18a599d06157106fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd199c07c9b285a08d3de98f9d275730
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_90ef020de744fdb3b674d9772bcf8a8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46e3f218c6cebaf0e22ded1d1e8afacf
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a4a9fce7b833ca8f0b39826026712156(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 52, 52], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82c9620efb686d16fc921b0f4b63549a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4a9fce7b833ca8f0b39826026712156
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_248cb12918ed6c1bfe0ce56081817e05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1eb2a4f12fa206101d5aeb1901febc44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_248cb12918ed6c1bfe0ce56081817e05
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0316f9268f96c7b128d0486cae593c1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29977c1a4db20c214c959c89a7bbb3d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2963287bdda394013b590a3ba66e2f47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1280, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57ba0012b1093cf185d557584f8a5af0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2963287bdda394013b590a3ba66e2f47
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e5325507bf291cf5ee8ffbe5d88cc894(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 23, 41], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_447441d16c66158f4b92be3fe34163d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5325507bf291cf5ee8ffbe5d88cc894
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3560ffd06ae4e7cd7e81fe1757ab2643(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 46, 82], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6e99f667660e623fb11b4bc9582d984(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3560ffd06ae4e7cd7e81fe1757ab2643
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4f0b756fb68b9622586af221c5d1e66b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 92, 164], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_654c38bf39331378f452de68c87c1595(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f0b756fb68b9622586af221c5d1e66b
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_593b2e09d95e4b0c08a23b7215e3e370(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 184, 328], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f64c0caea05c70d49a80744a3ecd4d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_593b2e09d95e4b0c08a23b7215e3e370
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_d76eaf17510765141a039db12e3f1da6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 23, 41], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b05d548d4b1c455a335f120d0b7f1180(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d76eaf17510765141a039db12e3f1da6
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_796c7ac31bf8edbe5a010ff807d61407(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 46, 82], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_110aa2bab9db85f8ec913a713f24cfc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_796c7ac31bf8edbe5a010ff807d61407
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_20fa019090dec328228f3544dbdcee7f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 92, 164], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c126635c077dfa1cb92dfe4f0ffbd1d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20fa019090dec328228f3544dbdcee7f
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6a3ded16cb66d83065fefe3dd68f3308(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 184, 328], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ec27f6a5f1eb24ed5604817eb56f6094(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a3ded16cb66d83065fefe3dd68f3308
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c895ceebd0dd95ac8513ab793f134af6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 17, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ee2240c2fa654330dbfaf8cc5e4734e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c895ceebd0dd95ac8513ab793f134af6
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3aab6afa88ffebba70fe159b59bfab13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9744afc8c677b65e43925422926b880(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3aab6afa88ffebba70fe159b59bfab13
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1ae7e735482410d6cca03c62249423f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86b8a43715a16b5ae8c69fd690ec2e13
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_223d985c7fabba2bb89243a7b9565bca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7cad5ea4594840989ab50b4a605b66e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_223d985c7fabba2bb89243a7b9565bca
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_38fb0fef30318c1b670cde9ed9fc25f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [7, 7], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 704, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf4f80cf9a31d5aa38f836b89db7693b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38fb0fef30318c1b670cde9ed9fc25f5
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4567c47b78242971546d3237069be36a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_deb6184483554f97ab37634187e024a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4567c47b78242971546d3237069be36a
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ba3f95b8a7035a63d004c10d637b1979(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88b84d6d9821474434987faa13051a46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba3f95b8a7035a63d004c10d637b1979
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bb31175e79c936c0cd30bdc7b6b7e634(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b5380eb05c739b9482d7ff8bc9c2bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb31175e79c936c0cd30bdc7b6b7e634
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_680c0dfab15e275f445de143cd82ecdd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_450b6e4c8d5accc28afbb65dfdbf2d46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_680c0dfab15e275f445de143cd82ecdd
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_3cc179ba88291091d3d36906e584cf6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2e2b8b092e80a6b2fe2ca4e625d6c891(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cc179ba88291091d3d36906e584cf6b
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_566936b8485a210bb84e5c36f5b33769(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 13, 13], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4bbe7b81412caab7536406683d27a2ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_566936b8485a210bb84e5c36f5b33769
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4acde2217e055d5fd8bb8f28863a1554(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 624, 20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ae66c97244040d9dd182976c097f732(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4acde2217e055d5fd8bb8f28863a1554
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6e4d8e38015e6e8087f28150d95e4c9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdeb49fe6d7599755fbaf5d7e10d5264
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e6bf1789aac0b73bd4ae1ffd5e9bb738(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed177c29c235337f14b02bf7b17f58da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6bf1789aac0b73bd4ae1ffd5e9bb738
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_de3ca8d4815042a852169cd046c476d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6625a0f12b3822b288a0e59b4863c5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ba802cb1fab9f65161d106903722ae5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a59a2bdcf3e5bc3973fc3f3f4dd35246(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_eaeeadcf2702789ac2f16f528fce4ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_13d6350a745a41fd3623c53fe22199c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_de43e69b5862e5eecd48ee67fd31bc2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cd1560e8d7e6782df64e6c3764a613b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9cd6663fb19f781789430aabdd3d978d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9a362b10ba49abb5ba9532fe9f52b970(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f34d4592e4413024b41544272ac48833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_65d899e1a6780ab787c3ec1bd4b43f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c22230b00364bc07cacc9eb39fb8f5bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df9053bce1b8f3ce0084cb74a076bd8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45035122af01ef307b89a6910e3b4b89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_79027bd741ce003cc232d0801c3a5e3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_53fbf4fa4b77a9c217abe7fb6fbfc638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_42d3957118b8869a3d9e437e14762c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c6e842d176bb80b2b247dbab62c62885(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_97266c31fb911cba4a8127f441e7ac05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cccf4fa235a5f31effc341bec4688eb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_60ed2e412ccbe00360280665dfe9ea3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7d1a194b916b824fab37c502725da749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_13ae699037ce5390c294317364aa01cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4b31d65e8bbe6d0206285ebd651d6365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ea41e5c48f313322007fa758513eb985(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_49f362f24492f1e48313630bac089738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_088951eac3e4576d818e9daae3c1ff2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_90a7e6ac2e74f4976b6b16be94638800(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a41ff2fb0e173054497b9a3369f86196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8fe97d3f67427327da041297d7ef8498(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3381bccd22c8a92ef5149637907fcbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b26d00add00fb393dc2fee9a39047e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8ef7188dfd52355a83b2b5768d616404(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_579b1399221e325c99cb50c2b6c2e92c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1a2cd834cff2c6ca185d794c940ea2d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_df74ecb1a06aaaeb89fb52616fb6a451(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_550b76aaf8b81f98f08cf2e9c11ca0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0e13b72b8e026cb0ce77e1be140c8550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e5e4f441645f304fb6170ba53a7a496f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_46890f0765ca5bbbadbf16de61a265c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 1024, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_600ba817d3b7e0063404522966fe25fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9b80930b97dc4e7af0c31e9a028ef168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e71f2e11d2ced484cc08b0cd4a28fbe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f31f7f8ca12a3650aaeaf047ff46aa9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d8a6d8e8ae750d3c9dd70c143ba7c56e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fc89da665079541a3dd6dd0cf92bea8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_93188f085a3df35e7e7912d77d4f1769(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f879862a0642e6d906fc82290ec4e6e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93188f085a3df35e7e7912d77d4f1769
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2eef8ea81074aeca26ddff62c32b69c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1afde56a8e49ae9afe4021d40ca1a061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2eef8ea81074aeca26ddff62c32b69c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a826f1db4653e58c50fcead88a156e8d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e500f7504541cb5a2a5515283db5cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a826f1db4653e58c50fcead88a156e8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c40336d8b3f1b56046ed3a7c7ab8a580(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d293b8a32106dc871e95910b9c7779c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c40336d8b3f1b56046ed3a7c7ab8a580
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f04dfd4bc4d9696a3b5326ba52608b9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_54a817c9bf906c94c7d39e35e84280ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e71c9fce85f0605f59e9c66dfe0baff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c7714e9a8e205995a9cafc857057cf9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_07ae881e2d5e1d2a67aeda573fc7b495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dc6ad8989a56a5e2cc58b1bc463239a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c40336d8b3f1b56046ed3a7c7ab8a580
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f8152b14a1c53e54da817290d8d66d66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a826f1db4653e58c50fcead88a156e8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_13872bc7c531fba3fdc0b5c868bd001c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2eef8ea81074aeca26ddff62c32b69c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_396f8566afbb0a17605eb9f68215b673(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93188f085a3df35e7e7912d77d4f1769
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_36841e7ce43369bed56cb6feaa067413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c40336d8b3f1b56046ed3a7c7ab8a580
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5a08eaa4f94736b62a01fdafbfc022bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a826f1db4653e58c50fcead88a156e8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_16588c4a543dcf4a0ec1e2fb447ff255(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2eef8ea81074aeca26ddff62c32b69c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f39c8d7e7cd5364b30f20ddd67f77461(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93188f085a3df35e7e7912d77d4f1769
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d3f238deaa58c1878d4a1563cfe44036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fe367434b1aa836072073885fded13d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cd1560e8d7e6782df64e6c3764a613b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9eda896a5acc0fd37f1d78db1fc52607(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ab14c9f4b7cba75e861b8c5a264324c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a1e126dd3efc8a8a2c9a56fb05d142f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_42737e3b4e18e180b0f0820e564cb937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d7a38339770289f49c980d79792cc863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bb06dc82d1a7cd32c0326ef8aad0d9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_236535a2b484c317ec0fbfeabb305b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4d0f321a877eba3caa840358f895df2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bb6b417c15db4418ae5d3c603f7475ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a58e54f6a1a31afe1c643262a7d46736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6a70acad9f56d233c3f4524dd9fd40e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_42079882c495a6bbac0d63de58ed8eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8491935a4104bf8fbaf7f9b138e2dc35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_49b9a0a5eaedf94bf61bef9c52453df4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f2d98f33192f286ac0aef3e67bd2f2ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_54b5d196a58ebb0f0bf168c005a08393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2a9a310408cc8a27d0ee5795cb4600f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a328b8c788292bf357e6a1d701eba547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_56deaa30294738f8785b50c019c81292(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_36b1a66a4c6bfeb2ef4a662044ecd858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_71d6111cac115abc466928c27a2c4182(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b16fb63aef828ee96286e0b7a484cdbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7aee64d69befc9eb01c4915932f502b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3dc60dfd0c4b0ba0a6e3c8766e0a829c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e6f337f3d36437f54509763593c6364a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9f9ba9786070485ce18fc8d55b71fa31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a16efbc130dc31223e691cb780344371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_52283ee65a4a8fd0d22f4c538f05e2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dbf95886d11c796aeb16ba920769b742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b265335a8fb4595e1cfb7f38f19cd580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 320, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_873cb4390a36b7737f9b66182af12f3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20edc22cd65c69c40ef26c0159b520f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9f68ed50ded346a8dd1be6ad5044713b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4a2d5523eccdbc846002dfdccf189aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_997d26d899705450af43d8ac625e6887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_472313ed6ed1d165ee41c6a4163916c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e67571a73ec9f294674cf7f700a2ee19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1232d65244e424c24a1507a0b54d0c0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_25920e8c23d9f34b20f04743b1a3f219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cda458d8103ac555996fa6ba47463869(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9ffb92d1dc478be1ffe48e902e366881(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b1400674673af6cd20050770909bb3d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ecb56b54a37a1c588036cdff873a6349(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2b5783eb6d3a3ff8b532d86f89e0a6ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_896efb09264da6536be9402ed5de61f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_16b39e5c87392bfe6a755d42cf658bfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ea41e5c48f313322007fa758513eb985(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e1a4a4f517455a3934378b07f2dde1b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9960a48a3be012e279b5cd4c6b34a23a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b7bc97d4c2a2e6ceb003d41caabfa419(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_20569c91559c05695b4648bcaec26407(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b6157d9753f0f16aacdfcfabfe10ce00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_369648345943e5e64571ab8180974e6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e00dd281cae72013db02d5eb3e6101a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1b015fc65bc9771fc54dbb61fa286b9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fc46e96b195219680d320c6600824728(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_428b2da9a96dba390048dbe5ccabf7fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce84a9f9f6080782a3cc74d6acb8b7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_428b2da9a96dba390048dbe5ccabf7fe
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ce84a9f9f6080782a3cc74d6acb8b7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_428b2da9a96dba390048dbe5ccabf7fe
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_191101222ace2181226babbd08727e02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_428b2da9a96dba390048dbe5ccabf7fe
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_191101222ace2181226babbd08727e02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_428b2da9a96dba390048dbe5ccabf7fe
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8a5be1d40614a7c08135336b010dcd41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_428b2da9a96dba390048dbe5ccabf7fe
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8a5be1d40614a7c08135336b010dcd41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_428b2da9a96dba390048dbe5ccabf7fe
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d08ae1a9ce2aafce611d3701a53d09d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_428b2da9a96dba390048dbe5ccabf7fe
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d08ae1a9ce2aafce611d3701a53d09d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_428b2da9a96dba390048dbe5ccabf7fe
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_940bef3ccb93054ddc0cbc7099419aa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6a1e50aca34419b53291c904aafcad7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_003888d1745da9c713e6fdb3d6b57960(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_003888d1745da9c713e6fdb3d6b57960(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7ebdfb689fb417e0aa554a50930e9002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7ebdfb689fb417e0aa554a50930e9002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_be95fbd56ad5ec6e5c99640a0a7beb6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_be95fbd56ad5ec6e5c99640a0a7beb6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_388c3888a25bab069a5a3e54edf1a7a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_388c3888a25bab069a5a3e54edf1a7a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_408b1c7b4fabee99464023e8776eca60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6aa4aaf86643b79aaa1fe278a3b614e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9f9ba9786070485ce18fc8d55b71fa31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_524fe4ef814bd0ebc99feab6115a0939(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_935b813dc98c5cd611774e0541ba263b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5c57f59c0cddc372158ea8fbc602905d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_07e57af58e2b34916d41f8173e628f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0fb77b866fd85a88430c46b234b541f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_67b52f13c48c6cd0d46cc01879f898a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ed7e1aba40905acf84a855a55fde52c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [14, 14]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca7e52a24eea031c6ebeb9c64f881e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed7e1aba40905acf84a855a55fde52c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ca7e52a24eea031c6ebeb9c64f881e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed7e1aba40905acf84a855a55fde52c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_057f334cce533294dba129c0392e9fd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed7e1aba40905acf84a855a55fde52c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_057f334cce533294dba129c0392e9fd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed7e1aba40905acf84a855a55fde52c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3240bf6394da54d58e2dd4be3c319643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed7e1aba40905acf84a855a55fde52c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3240bf6394da54d58e2dd4be3c319643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed7e1aba40905acf84a855a55fde52c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8e83671373f750a0d6610eb9ebd67495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed7e1aba40905acf84a855a55fde52c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8e83671373f750a0d6610eb9ebd67495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed7e1aba40905acf84a855a55fde52c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6636b920cc784be333c8ae4ac455ac4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6557ed89569d4cf0b2c2a5b9aabc001d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6636b920cc784be333c8ae4ac455ac4b
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2f0b1df091d68b5655f60846cf11a9f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6636b920cc784be333c8ae4ac455ac4b
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_43495dd6268793484e4e0d69d32ae64e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6636b920cc784be333c8ae4ac455ac4b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7bee8f3a31866ac3c5802569c7d4cbe1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6636b920cc784be333c8ae4ac455ac4b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bed8d245668544cceea85c84155b0ed0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c7e8462eda0d11e36ec795b873070ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bed8d245668544cceea85c84155b0ed0
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ede8057e4faebe222ba0d0d57cba9f53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6fac02652b6112c703e2c8c96abeb638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_39ee37db155778f075ad928cb2690be5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_198a62736ab3be8302dedbb5f0612b78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93188f085a3df35e7e7912d77d4f1769
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a01f301e3b25a6003610561a81ec2c0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2eef8ea81074aeca26ddff62c32b69c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_89d4b64bda119de41c7f542c46871096(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a826f1db4653e58c50fcead88a156e8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fad1535756744973df6526b680eac06e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c40336d8b3f1b56046ed3a7c7ab8a580
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a4ea49166d8013be71823c91083d3f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9f757cefca0ae5a786aa0083a9696555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_22d372248ea5cb7628b5b3612fb43af2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dab01b5422b45c5dc8f2e9fc7999bcad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b3652cc54ac131a37248aa9710faf4de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_81557119ca5b3b76aca8024ccccbdd39(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NHWC', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f32497f373f346f7e8fae2252be0775(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81557119ca5b3b76aca8024ccccbdd39
        def get_inputs(self):
            return [
                paddle.uniform([22, 7, 7, 2048], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cb099226f4392e9b91d1e2bc927f97ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_88182e0ae0cde4f03e5c9f7d87f6baac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_af2540fadac4a78b2f7e3fbb92098c8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_16def8ac817400013c37081f9e5dc2c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0d283bbe18606e46397cf655f4b05092(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_758ac4934fe29cd0c4f15cc2ed36fd93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fe42e54ee1eb9c44092a76f03c7c0ca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3271bc3679f9bc04764b5dbb9771a445(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_940bef3ccb93054ddc0cbc7099419aa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fe6a26e0ff65ccd1a6b7295e73b521fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d60246a7a7f081dcb452f0ff07326c16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d05b0122fbe33e6db8b5422a77a760bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_64f0bb99e09884340f7e0d0c5ecced57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_072ceeb79dc0c4110fc1d69cd3ba993d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_66d6d93e1dffc959a7391f228ebee2df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_af7f2776bee132041aeec1a31b432752(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f4fb2728ee039e17a91082e397944bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_16536428b0f739d59553143eb4b018f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0a38ecd6b07719129fd85d017a46dabc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f640e9da0e4ae42e25f59dffded74351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_08044032ba521e64c9b5031168d28b1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c304cf030c5c1c0a6546e862e8779e48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1ab6efc834354dea44fd9c5766913bb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9033c0c8a0cc4b26f872381ad2c4fb1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_81e84b98d843cc291b6d349042e7ea54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1f0a46da1cf3868f2baeab47382e7335(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0c7ab05b3db6b0554126f38ef00829a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7b19bd3f0b79c61806996794077e6297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2423327014e78eac2b716cf097cb3341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_64aa7c123e3042b79baa8ab70256a9e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d5d1fc57cf30adddd6d64b74118537d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_84e3f62b2b17b19a59dd0ede4e3fad57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2b7572bdf9340f524ba8cdca3533094b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_52c9c41e48ac4fba3f5d561473e46ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ed267a78d1b4e2634082c7815801b995(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b69c0c232b2c8ac51df9e38f678d06e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93188f085a3df35e7e7912d77d4f1769
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 38, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_4cdbd6bb5c6a66af57dd562e7f013b83(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e24d150ac92bb2f9cd735d484a3632e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cdbd6bb5c6a66af57dd562e7f013b83
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_98a7b6194a5eb72c72d6f694cab1ea27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cf940fc158da2ed7046dd9695251c28f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_61c86004bb063247d2a5e24f7e18fe79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cdb68f60c24081d7ec4800adb1add6e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_736bedfa67138a3320e2802dd6abe2bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c5c386663bca837a9923bd552b05eb0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0a1f21dbb8defcec8902c7f632de1799(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 320, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cd84ddd370cbe2593cd70a7eede2d00d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_53c757165f20074f71e1f953fde4c5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5f586387debf2e8d0af77e9b6911c3b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a951fdfc0c8fa16e42d9dd24a6093a4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3beef234a6a8e03266c9f8359ec62825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_66f3b86a20143421cd28ef482d10bc09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0aaafb4d7bc2ed84a914e37f12e54050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0b450354a9a4a4a0fee26eedf814ea19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 1024, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_686488869163e8d6cc90a692d5ab3158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9233239585262d7fd5cc6e5f7011570f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7b19bd3f0b79c61806996794077e6297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ee7445e864eab9ba9e6621ded68a0a61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cdbd6bb5c6a66af57dd562e7f013b83
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3575dce697ceff0cdae3e4f7615c732e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f78db565e72f28c90303208495f5448d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_62330958a4ac8b558ade87a633ea5b57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b1400674673af6cd20050770909bb3d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ef79287b94cc13fd7ee3981671b5e885(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b9bc3e070d3f73b14b830dfaf068c6f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6df5a54bc46fa2e687ad9dd681297cc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a0a0e4c53e350218b0522fb22c3e3ec4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9aeba56841896e988817f1dce63fd16e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c5ffce3b92287a5cca2fc9296919f9c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6ffede869105d2984d37b362d5a844ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e198e57f2ab7735f55ff74161a8e09da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_935b813dc98c5cd611774e0541ba263b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_86e2c88dc8bf681a827b3ee9b4cef4e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_64f0bb99e09884340f7e0d0c5ecced57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_10ef1084d7fa5ac5a54494ed4c94c951(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_07414b1a842f406d8725d14f11914621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9cfe231297c5ec372b86c334fc2e5c7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b6719a1c968d08538c3ac87806e554e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_53c251926be3ab14fa32dab1fdb9e929(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 49], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d45d2c60727b11c15c4f6027a506fefe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_608654ba75da43b1261d9baa8389c0f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c3c9adf516432e7c879baf5893450cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9191d88b74805c6037915cefc111b00a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e6ad3790db7cfffab01234e2c55cdc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_33eaf9506bb6813c32311bf090e10746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7d1a194b916b824fab37c502725da749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_2aa0174b1479c75b645597f92ec90137(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [7, 7], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e191425c76df11f168dce549d73ce49d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2aa0174b1479c75b645597f92ec90137
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cccf4fa235a5f31effc341bec4688eb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_389879a93d09c4acbe065db8a3e34ed3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_45e5cb3d1fdc1e3ebf9088f175b3a0fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6b75980ae05cfce19c5db6b70b9de551(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0225920e66b93a8c446e377e42827dab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2840d155eff77d657ec848f95e96b556(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ac98cd1400f1ba25d5998d4801930354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ac98cd1400f1ba25d5998d4801930354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_914cebd51654ad59f477a34d0b43aeed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_914cebd51654ad59f477a34d0b43aeed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_749b9a440ca171a8c9f101b94c48b945(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_749b9a440ca171a8c9f101b94c48b945(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_56e845938ee6fd67c3884a9b1e23ad77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_56e845938ee6fd67c3884a9b1e23ad77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a1e50aca34419b53291c904aafcad7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([28, 28], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_29c90aedaa0b9a620eb5660432f6dd5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_252d4c33cfa5c1a261b460961de0219f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_886c5ff46706fde1b92d19d97fc0ac4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dd674ae5a4d530b5cc3dd0f399ab7514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1db9ad35cea1e2eb188ca0860e7f6124(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_615f37c146478ad8dee34bf7e7418e7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_16f024b5074e64f6ee2b338d997dff62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0093679f5066a694d0efd37e311eb02c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_90b2d3a1a0ad4dfa41696dba8e50614c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ca316770b793eb5fdc7944fdfadb1e9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c40336d8b3f1b56046ed3a7c7ab8a580
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([16, 16], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b08363c1d79347433e279086d6df1dd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a826f1db4653e58c50fcead88a156e8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 8], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b5a2be92abb3971d4f95fa1403907add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2eef8ea81074aeca26ddff62c32b69c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4, 4], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_870c368946e46797019a11db4c1a29cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93188f085a3df35e7e7912d77d4f1769
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_53fbf4fa4b77a9c217abe7fb6fbfc638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_42d3957118b8869a3d9e437e14762c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fe58e2099086a916d6b87b857ecb805b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_16085f3b498e191bef2b018abc734582(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_896d15d80829fa1d43db9566bf56ffe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a336ed330c9fe9b2f15e423b152ea67e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f4c5993c440026f03cd9b82a4e7bec74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d0f1bd137e4c5064a31ffc5d337553c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c4c19e820a22e578d8aeb0a64fde2dad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6d1439d48975f9e35ca5454bd804bced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2bc7c8bb597bb147fe4e085230716eb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ce5ecc34b9777b2f7032e60d6b9061d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_698163d5b58fcb2b7f62c0d815aabbc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7b19bd3f0b79c61806996794077e6297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3f84652bc754999f5581f2791e99866b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ba85074804b9f96f0bc42687323b94ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_782bc07a81d795f8e0f816d258350d48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d24925e9e811fc05563697a9c7a12ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_15ab6d34bfd0448c94a3e73c37bbca57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1b2e9704ea59b26790748bb7576bbb26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7befebe5d325622a6efe1ee7b2019200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6c41f07855eee9d33bc9523d1eaf9c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d1a25b5b35026e72eaf6d268b491bef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_21ca8cce348443638ab49d82ebe252ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_69508a459f239a42402c32ee482470d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_736de455b9d14c04d28049ed006c213a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_93b96c6a3124fd7977327c59f303fce5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_53fbf4fa4b77a9c217abe7fb6fbfc638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_42d3957118b8869a3d9e437e14762c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fe58e2099086a916d6b87b857ecb805b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f4c5993c440026f03cd9b82a4e7bec74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9f3e02a51e50a4454c4efaaf3420ce0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c4c19e820a22e578d8aeb0a64fde2dad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5f586387debf2e8d0af77e9b6911c3b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6aac77af84b06a84cdb2de1b2f0512f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a520eed18ae7d22e436eed6ad5be3a80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b274e55c5392d7412f67b8d46ca4a6b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_63cb6cf80ec821a05cd10d301df70ffc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f2329cf590f88a49489773c3f53431b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0cfb3582b0ce519b2d9e2bcf0685f5ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0636ee3e58f1d3897f5210e203c93273(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8f52638ed639ded4665464c53aa98be7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4d62809f2e00c69e44612b64f438665b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3f1da5781574620392b8c2839ea349eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_de4163a4b8faa3c1bd953531a42ac5dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe97d3f67427327da041297d7ef8498
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0eb7b4a4a8d36b3798fab9e7d8c8b240(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d72d4adb13aa8eeaee0a8fcbd407ee3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d033067e7cd8410894ff155ac863588e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ccd1f36ca1bf30a5c8ab57dcb36c016a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5321ff71aceb1c37d77a3a0a97086e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ec3a6ceb2d151b690442859dda67b15f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_67b52f13c48c6cd0d46cc01879f898a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dcc673251cd4cb2d19f3b73ea01b1b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_175544c6b368756f230684a5f25ad421(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_71e690e8822f636e99009ec3b5b5aa7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_71e2f2e679477dcb6b74545ba9b0e79f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d79d0c823e19dfc4680f1373d7bbecc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_385bfcd8c5385d6dc11b90db955b0f76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c00c60eeb278ece8ac9e705d34171ec3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 54, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_273f77886e9913dd8017a9593376ac0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd3e92e4405a271ee815f365d6ec76b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2b9dc7c4a3bdbdde47919aeba3b8e7ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2da2c35bfd9c8e4e4ae55c1400a836e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_184259ba129f2cd8c197a9b0fc884a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d8e44a16191aa0f47f72a9843ce6843b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_69da4c3e24686ea01ea15c941ab63b48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a40433b4ef3a626d4cab631e7cb2e1b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_53c757165f20074f71e1f953fde4c5b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3fa15d3daa18529b70295bd75585cf70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e56c5ab3c1309f50db666f24204ec06e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3dc60dfd0c4b0ba0a6e3c8766e0a829c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e9c57a7157f0f736a9f39b94322e4423(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5c57f59c0cddc372158ea8fbc602905d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ab14c9f4b7cba75e861b8c5a264324c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_afc6fef86beb7a1418e2711f78837ae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fdbc5c132ad3f72e2db6739c862d8df4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_65f5edbb04cba8fe5d3c4b60f1df58c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d5d1fc57cf30adddd6d64b74118537d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_21877e018ed86487174287acd477e4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6471b95a504c49d51cef41d1108cf2cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_252d4c33cfa5c1a261b460961de0219f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a96bde17d52973acda1e51a136e165fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a6eaa63f943d29e2892ec2fc5f7079e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_49b1f13e2e10b41e265f5e8858b95a84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4d80a50eacf31cf8535251875497f9b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bb850c3436dfa9c4cd57806ccba5f011(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3550d1ae7891eeda85211aed60819551(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9a8dee46460b4087c5e3626f07471971(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4bf72b0f6a8b33c33fecaa3e2883a130(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6849231b38cda391f2763343d5bbcb88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_3b6df99128ad6d327f3e7c4f846085f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_44bb58566b96ccd5f7884b8d427c7752(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fc46e96b195219680d320c6600824728(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_74140eb2182bcdc3bb06b784d450de0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c3d55f7c55b7a81b0cc91717888e652f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2aa0174b1479c75b645597f92ec90137
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e6ab8207bed54ecbcf66d1d0c2fa922d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8412159648a43e1b6749668715058826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f14588b8fa68409016f8f36c82dcd7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fd81b50ef861c2da0e403b5209d512a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea4764c7abd62ba9ef331a96d443bd10
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([5, 5], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5418545e18a1a9a4da64e132fac2b536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d9c7a801f25bf285f03cb38131880b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([9, 9], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0bc069ab949890e84cb62be8a48efc9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8ed9a041cbf3d9174c0c3f89e4b8a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([13, 13], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d911c7099a3f992c3b8b3988c54e945b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_49f362f24492f1e48313630bac089738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_afeb12e7c173c5100531132e14321a5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de3ca8d4815042a852169cd046c476d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1], dtype='int64').reshape([2]),
            ]


    

if __name__ == '__main__':
    unittest.main()