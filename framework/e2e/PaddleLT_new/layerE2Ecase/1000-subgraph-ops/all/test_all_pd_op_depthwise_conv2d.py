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
    class PrimitiveOp_32577352f40171113a95e3f9030bc984(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_417ae517a51b06ccc9a9b5d1bb271796(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32577352f40171113a95e3f9030bc984
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a17c476b0413b24e3b4fecadb799ff73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ce35be59b1a30ca07d06e6eb4dc5b9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a17c476b0413b24e3b4fecadb799ff73
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf895beeb93a6fdae1be3bd0d71ff72f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b787e6e15187685918efca3704bd216d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf895beeb93a6fdae1be3bd0d71ff72f
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d2b108da1f569fc1b82ed39131f7ca19(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2f33b4f878f0caddea9511c8440c60f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2b108da1f569fc1b82ed39131f7ca19
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2f33b4f878f0caddea9511c8440c60f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2b108da1f569fc1b82ed39131f7ca19
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ff5c4c88dd2b4f3ee77f8b1f60fb943d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0efe895bb6fe5da3bbfa3468d9ca97e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff5c4c88dd2b4f3ee77f8b1f60fb943d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2479e74e7d3a3311a55bcd43a9e1ec7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c462d9bac23ba2d4c2ae25d60932e61f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2479e74e7d3a3311a55bcd43a9e1ec7
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e38470a51fa4615539634c62b6eb8ba5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 1280, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1280, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1280, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d573419af859ce68e89cd5eb3284996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e38470a51fa4615539634c62b6eb8ba5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8f70989d8029b685f7ac59a8ee99a948(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9024b45f7ff55ac809af58ab4169d80b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f70989d8029b685f7ac59a8ee99a948
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d5cab6091088d6e00021210038e4f486(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb9805b2e6c74c8c79b13cbdbd186472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5cab6091088d6e00021210038e4f486
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0cf3f9fe05b13e0083cdad7171251f43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48c7e0996e15a4b4782344db6bff3a7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cf3f9fe05b13e0083cdad7171251f43
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_76a6b2b332b6417709ec220986b89d08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d571ac592a79b013a5a41bcc5c128400(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76a6b2b332b6417709ec220986b89d08
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_37d9791f7e077c84b10ff258c844c7d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fdebec744a83d89446527d7a2a0f671e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37d9791f7e077c84b10ff258c844c7d3
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f064cd9cfdbe6628d88d7dc8b1ff4e3b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c8adc7fd8fe918fa5be51a9274e4033(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f064cd9cfdbe6628d88d7dc8b1ff4e3b
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef41077daeb45abe8b9b4b466da435ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd13b23396a0fe74811094d35616abc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef41077daeb45abe8b9b4b466da435ef
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f9116f77a59d2609501957e53e8e1392(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 192, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12528ad2fab389b2ca99f9a591076f0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9116f77a59d2609501957e53e8e1392
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4748cc85526bb20790c3db29798e397(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 384, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1706ce2f5466fac0f4506c376e811b76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4748cc85526bb20790c3db29798e397
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_043c6e0c64f62db7a84335b4e3589a83(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1e96d32bf537d2a33bb4e71498da1e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_043c6e0c64f62db7a84335b4e3589a83
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bce47adf52e9fbe59f9838b762167856(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5432962ae855578db759ad7910c3c73f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bce47adf52e9fbe59f9838b762167856
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f27486b6e70edb2efdf5c950f6d38fa4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7fb60bff67b9ac517ca992aeb3d823b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f27486b6e70edb2efdf5c950f6d38fa4
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a93e75f8e8b5112f1df942560d3ea716(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46f0f415ee818b73b120532f12b28ee8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a93e75f8e8b5112f1df942560d3ea716
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_373b24e81e95323af61cd12021904ac4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c54d21ddccb05fad4fadabf68095801a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_373b24e81e95323af61cd12021904ac4
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9dd991dda462604163ec5407bdb12d10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_197b60d669213d2a902fc36d9ac89923(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9dd991dda462604163ec5407bdb12d10
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_63d75d68905eee154ceb53482db5235b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 9, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_975e99d7a9fc1e7900e3715b5a7e269f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63d75d68905eee154ceb53482db5235b
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_65e9e27fc9887338e7213cf804567d20(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bd1c42a304b144c34bf46e2d36ae9c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65e9e27fc9887338e7213cf804567d20
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_830a3761dea33f012d78caf22a30f8bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7682f02bdced38800380f5f8a56f96bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_830a3761dea33f012d78caf22a30f8bb
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a54993d084333dc57db657e15b322453(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb6e2b461dfee10586fbdd2f456bae2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a54993d084333dc57db657e15b322453
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a206f7c042092d3d95b53af0de972b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce2fdf7297cf2c9145e00098e10a5c6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a206f7c042092d3d95b53af0de972b0
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc565bc50825813aaf4cf3591c0ad5c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 9, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f93320d3ca4ce29d34231b74993d821c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc565bc50825813aaf4cf3591c0ad5c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_36d4be8dd1cfbe2ddca91b35dc98285a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 384, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d601729d5553feb43728b98e03f5e50f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36d4be8dd1cfbe2ddca91b35dc98285a
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_736aceb4c8035f14285eb767229e1383(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1de7f92d01c11f057aea3b714877e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_736aceb4c8035f14285eb767229e1383
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d151ed9c972273acd5ca423ac12160aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_434ab3bfa7340fce249253fc749253b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d151ed9c972273acd5ca423ac12160aa
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1476ff981867de2ba944e98cf273b365(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76e1d56490446b91a297cec8b054ebea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1476ff981867de2ba944e98cf273b365
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c01d78a7a52ef510fada63733fa024e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [4, 4], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 9, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3ce916a508d160c47edb72c49891ce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c01d78a7a52ef510fada63733fa024e
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19113f2d63deec4c11f094895060b319(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [5, 5], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 11, 11], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_168ec4600a837d0be593840f60fcc32f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19113f2d63deec4c11f094895060b319
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0a2c5736508c07e20a3f5015c10976f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 768, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdd00739ed5753fab7eb133d08659fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0a2c5736508c07e20a3f5015c10976f
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_821ee3d4c7d1f924d6c2985c79ff7a87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e38470a51fa4615539634c62b6eb8ba5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ccbc10a652c31608de4fe37a050d44ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 768, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3cea07370574244eeeebb21870854c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccbc10a652c31608de4fe37a050d44ec
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_002f92c38b2a102d86b6bddd539020b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_588a356a012636c2e1358e29f14d0ff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_002f92c38b2a102d86b6bddd539020b2
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a6f7407e792aca136035804eadc4384(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c97707e8958afa4983d2959875fc9ea9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a6f7407e792aca136035804eadc4384
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6bf667e1ff18eee7a556f1f4243ec349(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce213e8d060f13ef6815e992beba26a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bf667e1ff18eee7a556f1f4243ec349
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b7c26674c9509c80c7ddda04ce220a3c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba0e80cb696fe5e265e0ae795f6a7503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7c26674c9509c80c7ddda04ce220a3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ecb2430047338e5c7111d417d840168(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0a0901b7776319fbe1bdd61c28763d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ecb2430047338e5c7111d417d840168
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a8e4bf4b1a842dfb5a415cb99fce1d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f9b7fd5ec422127da40f0a10aade990(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8e4bf4b1a842dfb5a415cb99fce1d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e61376c50abba75e8a1df680fb6a1cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab9a846bc89a0d3e7b3113c675145960(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e61376c50abba75e8a1df680fb6a1cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b78f9760635c4f2a4a9a38bd117f2a41(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be022b79e3017488f37a9b54e8e94fb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b78f9760635c4f2a4a9a38bd117f2a41
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2257fbbe5ae9041078f233870a501b54(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 192, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a835f7614a7b9ad43a396b87619cbf76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2257fbbe5ae9041078f233870a501b54
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_01829624f097bae972a2afe51cebf998(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48411bb4a4bfcd518a6389514c515cef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01829624f097bae972a2afe51cebf998
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_247742350b8ea011c5da8be7b177656a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9989d6817699fad0824039d534880683(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_247742350b8ea011c5da8be7b177656a
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c277984bedfb8b827efd15d2fb641d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fa821d45cc2ace7feea980bcbba63e3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c277984bedfb8b827efd15d2fb641d5
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_333645d14ead5126ec64d9dde97c7c52(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 48, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_718bd9f1be6bee2ae3cd8e43c82955ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_333645d14ead5126ec64d9dde97c7c52
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57ec109f22a13d27b96debf6456e55e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 48, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa66bd07fe00d9929ee34cb09e76ea4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57ec109f22a13d27b96debf6456e55e8
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4d180537180b65caea1ce73accc75ed9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 48, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac33cd077c7786bce5921f308e1b9c00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d180537180b65caea1ce73accc75ed9
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4153e3f78ee906e7666e43c98225479e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_26da9e9654d82b0de7c749b2654f2b5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4153e3f78ee906e7666e43c98225479e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26da9e9654d82b0de7c749b2654f2b5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4153e3f78ee906e7666e43c98225479e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dd9258b18a48c14b90abfec4610d77df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7bc67e7df753df2fd376473229efa2e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd9258b18a48c14b90abfec4610d77df
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2128c2b4300e7d8ba7eda8e0a8896129(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b87758806bfdaf40775faa99d92a1bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2128c2b4300e7d8ba7eda8e0a8896129
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_04fa18a7bd56bdb2157123418db40db1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 1280, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1280, 32, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1280, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_273f7fc1bb1a8aa041db0bb17f306a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04fa18a7bd56bdb2157123418db40db1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_54b392e8fb9df02a88a4b048d1e08a05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f5cc79950689d868300118d95a0910b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54b392e8fb9df02a88a4b048d1e08a05
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_24127bf7b2291238d4f66e58a38dd01f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_434f8779cc5fc0784336d0a589325cc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24127bf7b2291238d4f66e58a38dd01f
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_29b10932d1a7f63a920cab9207cbdb86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5aad6d6b8ed3c78181deade5dc1caf5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b10932d1a7f63a920cab9207cbdb86
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ae29b4399195df663eb90ca617ec86d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9419f118d2ac87c1c2d9a561bc0ad6a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ae29b4399195df663eb90ca617ec86d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_78d5d46528922cd9462a50b45dc4c85a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 160, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7cc90e4b3d0ffb68acd9712d669b45b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78d5d46528922cd9462a50b45dc4c85a
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_974336207b741a788ffebc947fdfacb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 160, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd0953a51218ea816acfa53829774f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_974336207b741a788ffebc947fdfacb2
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b71ee382c694e20e8efec17e6a3aa140(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 160, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cc4bce5a339e121ce748a34eaad3492(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b71ee382c694e20e8efec17e6a3aa140
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12528ad2fab389b2ca99f9a591076f0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9116f77a59d2609501957e53e8e1392
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1706ce2f5466fac0f4506c376e811b76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4748cc85526bb20790c3db29798e397
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b69ceca43ac7b292da2d5f5d7c41e2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 80, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f7f72141336487e1ee5d87f1c430c90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b69ceca43ac7b292da2d5f5d7c41e2a
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_52d620f69dec18726f7dc5714d2cd824(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 80, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d6264a63f1a66cf29a0baaf894e6a2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52d620f69dec18726f7dc5714d2cd824
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_254d09f25fdbe4774ef081365a650771(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 80, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3c36ee8a1480a134450c1d4287ca0b05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_254d09f25fdbe4774ef081365a650771
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d74e8fa6e411573a3db4676d4a3925e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 300, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ab7c45cc3654244ec4dc5ea495a0342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d74e8fa6e411573a3db4676d4a3925e6
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e37dbef8003d1ae00c36b29c17101e05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 300, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3263d49b67cfa44b623813a7877ce1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e37dbef8003d1ae00c36b29c17101e05
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b38497dd8044dd9629accdd56507fe5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 300, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_300182ab257dfc4f98dba6fe1090c4c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b38497dd8044dd9629accdd56507fe5d
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8149a266c1726bbcdffe9913c8872039(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 300, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 9, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2a9f0a7582084ae204017d5acdf29b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8149a266c1726bbcdffe9913c8872039
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bd1c42a304b144c34bf46e2d36ae9c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65e9e27fc9887338e7213cf804567d20
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81a03b11deaf74c904da423e70217b66(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 90, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a92ee2900a73850ef588b8d7d33fd8b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81a03b11deaf74c904da423e70217b66
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3aea40a314a413bf6f71e7e48c9bd3a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 90, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_829130fe65f4c41869a1709485b0faf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3aea40a314a413bf6f71e7e48c9bd3a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bd06ca87d34eaac8f57e32279cfc0e86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 90, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2803490683ee7923342e71a2447024e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd06ca87d34eaac8f57e32279cfc0e86
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c717e40fae1ac09741a73790f71a7c2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 90, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 9, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70ffcb64bafffdf7effe43f2d72737b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c717e40fae1ac09741a73790f71a7c2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d601729d5553feb43728b98e03f5e50f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36d4be8dd1cfbe2ddca91b35dc98285a
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6db7cfa602fe57c36704a2b735ef5f4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1762e7e80ef4421b5ee8a0ddfe67096a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6db7cfa602fe57c36704a2b735ef5f4b
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f81ff0594ddbb2564ff4e448ff7789b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0bde1b3b6a0fe4a4017c99cd6223ff19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f81ff0594ddbb2564ff4e448ff7789b2
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0fc5d85e251fd0ec598baa751fffee75(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b52f207e60a56233470b5c5fa563c1c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fc5d85e251fd0ec598baa751fffee75
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b0a8ba97b735b49ea0a63b92e9324266(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [4, 4], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 9, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b201c7bb77cb512ee1e6ff7b2ab848ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0a8ba97b735b49ea0a63b92e9324266
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87ed08948d1ef491e715792207b10b9f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [5, 5], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 144, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 11, 11], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_422c6437e25d6657d1adff5e329c6fd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87ed08948d1ef491e715792207b10b9f
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdd00739ed5753fab7eb133d08659fd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0a2c5736508c07e20a3f5015c10976f
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1c9806b2cb50a4ddcb62d44b7bb42770(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 1280, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1280, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1280, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b33244131d46194ecce74d193256cfa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c9806b2cb50a4ddcb62d44b7bb42770
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3cea07370574244eeeebb21870854c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccbc10a652c31608de4fe37a050d44ec
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc25508d73009063d16afdf5e992a793(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 256, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e89d7551ab69155b6572d0305e58f71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc25508d73009063d16afdf5e992a793
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9acd0200fefed4095eaac13bf8d3b9c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 256, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23d249cad54ba66bd68f27eca93f7109(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9acd0200fefed4095eaac13bf8d3b9c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_458c4baad12570601bc77ede3b0768de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 256, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99a6dee72b8949bacc4665ce3c727d8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_458c4baad12570601bc77ede3b0768de
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_84e9d159cace40263d9a656495a05bcc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 256, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab6d83eb12ca9bd85003d52438f53b8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84e9d159cace40263d9a656495a05bcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17e4c7243457090b2b2888c774dff6ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_904b2b838deda10878c022d331928dba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17e4c7243457090b2b2888c774dff6ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_03996b6a5132d3d528316e92b8bc37ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f3ab9a7e009f7c88de2b1b55066567c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03996b6a5132d3d528316e92b8bc37ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f03929fd81f7e45732ff6732b4d84352(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a6d5b9b0ae3b400d62f1ee627af44bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f03929fd81f7e45732ff6732b4d84352
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c6a3929e28e75c5c790b2882bcadd7b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a426c828377737ad091c370b42cdd57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6a3929e28e75c5c790b2882bcadd7b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a835f7614a7b9ad43a396b87619cbf76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2257fbbe5ae9041078f233870a501b54
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b374de66ca1524df288b398fdc27a80b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 240, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23b74b0dddc0fd756766103fbe07fd22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b374de66ca1524df288b398fdc27a80b
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_68491180e0265605a0af2e66546c7b35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 240, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8af18d874cd9e6a5e24d28ce4d639fd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68491180e0265605a0af2e66546c7b35
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa821d45cc2ace7feea980bcbba63e3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c277984bedfb8b827efd15d2fb641d5
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a85a2635b04a8beef6ba46b46dddb875(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67c3a483edd6d80d5940a6848fd49045(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85a2635b04a8beef6ba46b46dddb875
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8e0b1e99999c3c35151fb6b1b8ee5462(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17ff2d99089bcb5d118e06ee88ff5489(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e0b1e99999c3c35151fb6b1b8ee5462
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c4e88a54f19799f216ee7aabe1ecf08f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae9d0a5aa5fbe47ea5ca069110e4d7bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4e88a54f19799f216ee7aabe1ecf08f
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_465ed68a0be0214b82dce363cce5e2db(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdda131bf0a3a4b2bff459c6fd110bb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_465ed68a0be0214b82dce363cce5e2db
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdda131bf0a3a4b2bff459c6fd110bb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_465ed68a0be0214b82dce363cce5e2db
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eb2f9908d6be903ada91c63a20d93273(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_145666cb603704ee02b585213581381d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb2f9908d6be903ada91c63a20d93273
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5a7b79d5a95ee3e00d9d5ade100d36fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e980a9863ec80ff9e6acbe4d134c3f03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a7b79d5a95ee3e00d9d5ade100d36fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c8178b1e61f2e04d3f173c7820fb3e99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 1280, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06dc996177eb5e844ed37812906c017f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8178b1e61f2e04d3f173c7820fb3e99
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d1a6cedc421ab0417759602292453142(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4c88c1821dd3dbd2178b07023934afc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1a6cedc421ab0417759602292453142
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db47b9aa8a293a21b4f694de50ae8934(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4048a9546689b28ac0518f4b284e0ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db47b9aa8a293a21b4f694de50ae8934
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8939ca7f7ccc78a5defdf1d67d2127d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_770fd380d324bab5400f103d381fe296(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8939ca7f7ccc78a5defdf1d67d2127d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b1ee267faac2fbcbf61ea072112eaa84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6871bd60401325982077d45f7eb77702(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1ee267faac2fbcbf61ea072112eaa84
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db3c2e5e91ebcd1aff8bafca7dca9ebd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_598d180d36cb76237d84bd14bcbc30c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db3c2e5e91ebcd1aff8bafca7dca9ebd
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d2a9308ab6f53dc94809c6633d0ebe0d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90f1eb65ac23956e1ec03570739748bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2a9308ab6f53dc94809c6633d0ebe0d
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4328045ab64305c03813abd472997fef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6c68ee0a2a0825a86d91dee915cffe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4328045ab64305c03813abd472997fef
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d1ca9c7f237504c4f1ae6e9ed74f0ab9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_732e683cd9df08fb51a9e3a5a9368458(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1ca9c7f237504c4f1ae6e9ed74f0ab9
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_80e310d239b6ec1c87f3c0ce161b582c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9263c0f2e78a74f233c07b25377585a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80e310d239b6ec1c87f3c0ce161b582c
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_02243e6486310320f56ede107f31e402(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_caa0c61b7e4731a2add565d376e4e588(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02243e6486310320f56ede107f31e402
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac7099542d9150d96cf65ae7cd8f51d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6b3a4ad1ca5c96f79d2872ee7bfbae5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac7099542d9150d96cf65ae7cd8f51d2
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_94fddc6aa30ab3612854f74a9c94ac57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fdb594fccc90507e543d098b1d1605c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94fddc6aa30ab3612854f74a9c94ac57
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_60b48737ab301a0910a1293db491f9dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5aa4b717b97982e9187682d68ea3712d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60b48737ab301a0910a1293db491f9dc
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0269cdaf155976a69c179cdf1eb51e48(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9720f41de848c5b1db16e6f93914aa66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0269cdaf155976a69c179cdf1eb51e48
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dfba01175b4ec311d50a708e1f9bbe38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d44f384c08009ecad11fd2367a20320e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfba01175b4ec311d50a708e1f9bbe38
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_61f3bd67b9f93594d0dfcc6def70a9fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5663d10074915c16b11c5246f71b413f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61f3bd67b9f93594d0dfcc6def70a9fe
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c197d53d1b3e72522b91ce62f5e13b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9fe6a5a07cda8fd05af67e466a81c01b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c197d53d1b3e72522b91ce62f5e13b4
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6892fd271db4db96fc68dc9b04e5572(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34e01002ada0b224db181e966d55b53b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6892fd271db4db96fc68dc9b04e5572
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_760fc3938c4b78f5f286f64d85ea68cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28f8ca574b6ea0754c5cb67cf02dde76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_760fc3938c4b78f5f286f64d85ea68cc
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_557f8cc58150224d61fab250b3b84c60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5c5c65a5aa1b0c3350a2e3e13323369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_557f8cc58150224d61fab250b3b84c60
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db3b1bc50cb48a340e89edf0436a8318(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab7a21203601343007da7625f80b21f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db3b1bc50cb48a340e89edf0436a8318
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fe1ad8f83a7a00d94fe73547809efbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80e310d239b6ec1c87f3c0ce161b582c
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_67c17a52366a6ad67e1a9f436bf793cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44bf945a73514cb010a4143ca891e346(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67c17a52366a6ad67e1a9f436bf793cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e291c59dacd727eea7e5a8b9df712e37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f95c5ffc4b6a69ede3352c425bb033a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e291c59dacd727eea7e5a8b9df712e37
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1514c7b88992e629c8905e91547414a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ee1165c4937fdbf7e0f9fca37fc18d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1514c7b88992e629c8905e91547414a9
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_872a10331acfbae96bd866c18fc27070(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [4, 4], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_00e289f14795c29be6ec32d9c0212db3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_872a10331acfbae96bd866c18fc27070
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_75e72b505a0a591538a6c092443de4b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [5, 5], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4def55f5b7f12cb73adafe25ddaf8290(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75e72b505a0a591538a6c092443de4b4
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d95dd47eb6509dd29ad3aa3c1fd1f28b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a263cc12f46b971bf2fa995fe586bb98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d95dd47eb6509dd29ad3aa3c1fd1f28b
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de8b50443fe436bdb3b1a6d19b7c4e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8178b1e61f2e04d3f173c7820fb3e99
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b037d34e3c75e41dd50cdc5a8f62f798(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d95dd47eb6509dd29ad3aa3c1fd1f28b
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_069ad8cb4b0f8e280b1b008434d68112(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c3fb76e6f5aed86bb3afa41e4a24ce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_069ad8cb4b0f8e280b1b008434d68112
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4de72c10d14dbdeacf6ef4e40bfd543(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e4886c7f1c2483593d85e1354bf693c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4de72c10d14dbdeacf6ef4e40bfd543
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7d5a57985dd39df4dc9ab349bbacbecd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d755561b15c592e670f6e9c8a834f554(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d5a57985dd39df4dc9ab349bbacbecd
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ef53663e237d7e220e410cc7fea97d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10146cf36db89d9eb3211beda8348d71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ef53663e237d7e220e410cc7fea97d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_02db1494ee701b95b653df4471bdd5e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3476cad4841e42af6291703b20965f04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02db1494ee701b95b653df4471bdd5e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ab6bed3d506439e3e90efb126557b156(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_504deb4974267c94a09d6fcfe9a62ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab6bed3d506439e3e90efb126557b156
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cae38f028a9613ce74d18e82e5aaf427(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22f43595c107b7ae11562b5937037f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cae38f028a9613ce74d18e82e5aaf427
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_54f3da9fa96d8a15fc8a6b5d3007ba37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1f5df0f275ae8473acb5558b35ffb536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54f3da9fa96d8a15fc8a6b5d3007ba37
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79581f982ae95985a11786f69b82d4a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1ca9c7f237504c4f1ae6e9ed74f0ab9
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_82197d2f45335df2eda984603ad17608(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7d33fd97c412409c8d0d7b18905f631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82197d2f45335df2eda984603ad17608
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_43c12e4da9230222e5df087c3c93d3ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1f39531b6d55bf8df4fcf983fb4b23f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43c12e4da9230222e5df087c3c93d3ff
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_366a930ee890a1fa22375d07fb785b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c197d53d1b3e72522b91ce62f5e13b4
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()