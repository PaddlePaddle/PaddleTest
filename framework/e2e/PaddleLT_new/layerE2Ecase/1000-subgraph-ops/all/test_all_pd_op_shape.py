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
    class PrimitiveOp_274dcdc33ea895a1a6bfe132d21f4317(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 24, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7f5623ebd6cb7324664931f40209205(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_274dcdc33ea895a1a6bfe132d21f4317
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e586ccabd555c6de3e9035e85131eaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_753f32fd2d04b4713dd2d3e9aa57e0a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_51d3c17a5a83078bf414f003f6d28eed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1dcbe83b56453ed9978ee6ed798a8aa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d3c17a5a83078bf414f003f6d28eed
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_092c9fe4432d39cc35d4233fa4191624(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f9125974c873dd7e327d33dea3cfc64d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3735a25ea0bb7b4ece1dd9ac2d0b67fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efd1853b9820068883ad373bb880adf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a91fde7eacf6e7dbc7fa0d45586744d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97c46c190a182c5b72306e94d1abbcf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45d5d9dfe74280697699f2ed68ffe28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_06615f427d164bd59f20c5b4c0dab205(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5781da24b99e814ab90a9aebc44f26b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06615f427d164bd59f20c5b4c0dab205
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9052362698bc5041f38515b8fc8dc194(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4588c7cc83ada80bb03f1ca7e9780657(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9052362698bc5041f38515b8fc8dc194
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7f9655ce31166d4ea668936afe2d0422(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be8b9f2c850cfc3c610f773d7ece1b8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f9655ce31166d4ea668936afe2d0422
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc6b534256b345877d9ecfcbd36bfd7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 7, 7, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_626003b5a3982a6ad985d182261b0014(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc6b534256b345877d9ecfcbd36bfd7c
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_139726f9ad93d3792c52933c4f454c29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c022b15aeb0779c03374042a4a12875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bce0074d61f842652fa88a1b125a9850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23dac26063e0f7718091511ff4cddceb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9d94fcaf6f3a1005815205e2416633a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3af91785e0ea7aeba420b039dfdf12ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_950a5e5aede532d792126f65fcdba459(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3af91785e0ea7aeba420b039dfdf12ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8995400aae4bb8e2f603d18e87502325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([12, 288, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ccd010ac9edb5ac57c7f9ea74ae7470(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9052362698bc5041f38515b8fc8dc194
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8693d85dc5bd6f20e0f8cc0673bf82bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c813968f6f47a9af33c0d0b7713af138(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c285f4b4013f414c13a77cdb95d3d2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c813968f6f47a9af33c0d0b7713af138
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebeed7afdd570c315f214de75a6e90fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f9655ce31166d4ea668936afe2d0422
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4de291aba87d25278f050a9a85c9d05f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06615f427d164bd59f20c5b4c0dab205
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74f9aa2251859b074dccd11de0933050(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 14, 14, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4524254ebf46b5a34f2808035080d5c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74f9aa2251859b074dccd11de0933050
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a729df3b110b8c26ba701f6bee526d08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c2a8ceaec9879e9f2996bbe2de5372f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a729df3b110b8c26ba701f6bee526d08
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f62111b52f3fd9fc6cf4a7c74d4d7f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_69ef785ce37af44088a3317050a5bfd1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6be291e756f754aa53465cf5c3059d7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69ef785ce37af44088a3317050a5bfd1
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c2cc54f11843edc3207164129be3d3d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 28, 28, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_009d13654430ed4cbe57365382b482f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2cc54f11843edc3207164129be3d3d1
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03586d6251cf6775545b55c0334769fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d3c17a5a83078bf414f003f6d28eed
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6f6f4a771bec704459d45ed44ae388d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d335b97fa3be2e98755a1d4981bd7a99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac52dc531364be75487cf042f633233f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 20, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3acd638de3a05a9a15a80a9c927be716(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac52dc531364be75487cf042f633233f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ceefde63f6d11126635804f01040154(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a91082e380f04b3336f3c03b0f8568c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fecce13bb6b7fe9ab216f68974772d50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91082e380f04b3336f3c03b0f8568c2
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_29abfe96ba2365611db70ad8b4c0732d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 56, 56, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea86b66b393ae773707fdeb5563ce26c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29abfe96ba2365611db70ad8b4c0732d
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_124b4c75930a412c8d6d1ecf220eeca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec9213d6354a0e752f9b4e41e8e5d132(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c813968f6f47a9af33c0d0b7713af138
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6b439ecb13a2df6ef9569e2ed65f74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c022b15aeb0779c03374042a4a12875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cdd0c2254946073a7616bdc1b9d8ebf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2954c353a8199e1756a253a65a10719b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b12a5c0ed79ebec24a66847fde49797(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2954c353a8199e1756a253a65a10719b
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_faa4e0244140a301cee90e6cd17fb4e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f9655ce31166d4ea668936afe2d0422
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a680243a70bc31c6e04b622618557506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6dd8699a45bbc63727d14dd404a45578(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_58f725c88a592ef7b74ac1e38fbcaaca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f65b8d8dd8ca9f1a3d90d6537840086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bc9de8ea12ba79d6df9b75723e9b1fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2f63d0ebdf6bbec8d5d4f6b010cdba5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3adebe1582dde27ad4d5ea0a18ac1d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2f63d0ebdf6bbec8d5d4f6b010cdba5
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c6281f3ec19b2b847d4db2e7aa4dcd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f9655ce31166d4ea668936afe2d0422
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df5bcb80972a3d1a77844a86896f01fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc6b534256b345877d9ecfcbd36bfd7c
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7863bbf4c8be0c4fcf74a2c92f2d4d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73480580f9f4151241330c0ae3e6544a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7863bbf4c8be0c4fcf74a2c92f2d4d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7703cd872ca652b8572e69720fc20848(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2d3c67bc76ba06e4db3e71deff6f50a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7703cd872ca652b8572e69720fc20848
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee8e914a41aaed7819739d17263c564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f90ee800696d75c47b9ac37a88455d64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2681e6e0e4141639fb4c9c5a215f7ebe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85d262590c839e4ca17a78beb3d3f6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_82ec5e53b2777a25433b3ae2c3fb911f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e10fb0207ae0feebf4984f51ba2a83c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82ec5e53b2777a25433b3ae2c3fb911f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a680243a70bc31c6e04b622618557506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a502a6a696ee46b162694bbe6942d936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93135260365be33e676c83dfe7394abb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_beff53e6d2b8e0d2cb5322608d3e252f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab2fb56d04f997cc6e640d93f59d33dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_beff53e6d2b8e0d2cb5322608d3e252f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13b1eff1f7985bca864fdd4f9cb35958(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_beff53e6d2b8e0d2cb5322608d3e252f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cdde27ae7ff5c51ade019b56c9a8135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_337c7d2d3c77aaceb81930cb51e6135c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 28, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d838ad487e5493849a14be5b1d7f61c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af61365688a1cfb8c58cffb9f2387236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af61365688a1cfb8c58cffb9f2387236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c022b15aeb0779c03374042a4a12875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fecce13bb6b7fe9ab216f68974772d50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91082e380f04b3336f3c03b0f8568c2
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea86b66b393ae773707fdeb5563ce26c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29abfe96ba2365611db70ad8b4c0732d
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa88130b2b655e9a0e84e113ba58ae80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0862bec01e16568d837abbe17943a6bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6878052bd3b48b73ce9a3b1b528fa2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_511bb01a56a8bcac581b7dcdee0f397c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69399b30103a6e78d614136b5e3c7e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69ef785ce37af44088a3317050a5bfd1
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70d2eff864bfbbd5d83dc9f40b090bd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2cc54f11843edc3207164129be3d3d1
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d27016f228d65c431dc6569365641e94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06615f427d164bd59f20c5b4c0dab205
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3681f36cd7b1ea291b9b688f1578d4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23b4364dad0b39e15bc0873c019c619a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4de291aba87d25278f050a9a85c9d05f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06615f427d164bd59f20c5b4c0dab205
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4524254ebf46b5a34f2808035080d5c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74f9aa2251859b074dccd11de0933050
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8741cbe5339c793961c626e67ff61b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82ec5e53b2777a25433b3ae2c3fb911f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_995d1bd401f6b116e95a82e89698365a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a729df3b110b8c26ba701f6bee526d08
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_121f7cc0749f52826e35ed619cfe028f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 464, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6b439ecb13a2df6ef9569e2ed65f74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_086f2600cf2758b0a67583da5b842cab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7863bbf4c8be0c4fcf74a2c92f2d4d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aafee2af0c8274014d20fc809310a953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7703cd872ca652b8572e69720fc20848
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79b44ee3cd47b229e26176ce1ffff2c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_beff53e6d2b8e0d2cb5322608d3e252f
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c11326e3c118051b4a310959a21c3bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06615f427d164bd59f20c5b4c0dab205
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7010186995d36cdb0622d51c27973a68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74f9aa2251859b074dccd11de0933050
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba3b704514e713a671f0c42c2573db25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_868dcc9778512f6b70bdcd09a61394d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 14, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e37368c6c6186d0a64a77b06ab980f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91082e380f04b3336f3c03b0f8568c2
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9916c48d25a6f8acd99fc05bc2f9fab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29abfe96ba2365611db70ad8b4c0732d
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f8d750fc9d061c0c92732a7721c25a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d50d2357e8a27f72b59b4eb095b01c43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6adfe4c1b2145f874a82d8b2176cfdfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58f725c88a592ef7b74ac1e38fbcaaca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58f725c88a592ef7b74ac1e38fbcaaca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58f725c88a592ef7b74ac1e38fbcaaca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cb4af8cadd22974b80613ed4b88def3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_161293ee1330124b0daddf685378da35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb4af8cadd22974b80613ed4b88def3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69399b30103a6e78d614136b5e3c7e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69ef785ce37af44088a3317050a5bfd1
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70d2eff864bfbbd5d83dc9f40b090bd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2cc54f11843edc3207164129be3d3d1
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3da78c7c4bc41b5866121cb284d57903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4b77f94fe0b26b063666ad049da6b32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6336f943dbc6991a7320af2ab89f7774(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_984640033ed5ac36ecb8c5c3b79722d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c21cd8bf5bacf2b0dd344f54fcceee6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5bfa59266db2a75c004b7110bca6b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58f725c88a592ef7b74ac1e38fbcaaca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3681f36cd7b1ea291b9b688f1578d4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3681f36cd7b1ea291b9b688f1578d4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3681f36cd7b1ea291b9b688f1578d4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d61784edeb74a7794897901b053d79b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb4af8cadd22974b80613ed4b88def3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c11326e3c118051b4a310959a21c3bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06615f427d164bd59f20c5b4c0dab205
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7010186995d36cdb0622d51c27973a68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74f9aa2251859b074dccd11de0933050
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_05a51bf8377ed14b638ec25650925f2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 15, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f0971b33ac0cbcd09576ceea7cd379a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05a51bf8377ed14b638ec25650925f2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4de291aba87d25278f050a9a85c9d05f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06615f427d164bd59f20c5b4c0dab205
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4524254ebf46b5a34f2808035080d5c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74f9aa2251859b074dccd11de0933050
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_faa4e0244140a301cee90e6cd17fb4e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f9655ce31166d4ea668936afe2d0422
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e1a4b286ef3b50e8c7b4da5638c29d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b40f81bdd9acca9e73a0eff9be1a3c5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e37368c6c6186d0a64a77b06ab980f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91082e380f04b3336f3c03b0f8568c2
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9916c48d25a6f8acd99fc05bc2f9fab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29abfe96ba2365611db70ad8b4c0732d
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307bd7b03fe9f08998748723c31b5231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_822cf049feaf9613ff7997dbb98f6882(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7594f5d702257c3fa6a136a56c5eeaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_822cf049feaf9613ff7997dbb98f6882
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ae406ac0225ba346e2ab0ebec567be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be8b9f2c850cfc3c610f773d7ece1b8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f9655ce31166d4ea668936afe2d0422
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_626003b5a3982a6ad985d182261b0014(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc6b534256b345877d9ecfcbd36bfd7c
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c6281f3ec19b2b847d4db2e7aa4dcd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f9655ce31166d4ea668936afe2d0422
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df5bcb80972a3d1a77844a86896f01fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc6b534256b345877d9ecfcbd36bfd7c
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e8a7af76a69bd30cbb620c22a502e83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51d3c17a5a83078bf414f003f6d28eed
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25321af94b9bc76e12f1764c11b17d7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_beff53e6d2b8e0d2cb5322608d3e252f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66d7266171c7201794998353bd49b934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97c46c190a182c5b72306e94d1abbcf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45d5d9dfe74280697699f2ed68ffe28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c0f62a43370e52b7b6f402ca05c8e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c4ae7e35c4cafdd6e20835864d1caa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_627ca259f30dc36897f7fd2bcfec0f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30616e9fb3cba70b81937113fdf7dbb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bff43e7d4c8f0f9ab2da9baaee621370(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_822cf049feaf9613ff7997dbb98f6882
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6be291e756f754aa53465cf5c3059d7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69ef785ce37af44088a3317050a5bfd1
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_009d13654430ed4cbe57365382b482f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2cc54f11843edc3207164129be3d3d1
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ba1928fcd44ef39e2044c46275d0f53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f9655ce31166d4ea668936afe2d0422
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb814a40a03cf0c420616eb414c903b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 232, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fecce13bb6b7fe9ab216f68974772d50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91082e380f04b3336f3c03b0f8568c2
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea86b66b393ae773707fdeb5563ce26c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29abfe96ba2365611db70ad8b4c0732d
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdad5595773b25607c17c6e0d8e15ad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acce35679856320b2ab0b983abd6d00f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bc7672dac0ab39f764a9b8de52d707f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ac23e1f2934c09fcf7af4f4684e7858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbe056b6f75e7f5f52f1f3db8737a494(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c11326e3c118051b4a310959a21c3bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06615f427d164bd59f20c5b4c0dab205
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7010186995d36cdb0622d51c27973a68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74f9aa2251859b074dccd11de0933050
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4223c4dbc66083696a78d38c8cd2e389(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b0370a5b37c49d370dc7479dae9520(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06615f427d164bd59f20c5b4c0dab205
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3681f36cd7b1ea291b9b688f1578d4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dd8699a45bbc63727d14dd404a45578
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ff8934d4f28e26c2592445f0d0ef23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_beff53e6d2b8e0d2cb5322608d3e252f
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af61365688a1cfb8c58cffb9f2387236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d4690924f58b06838072e0033aff91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 116, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bfdcba090ea6d4be967da4bfff6bf76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f9655ce31166d4ea668936afe2d0422
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f39900f4922798de3ab6617c00ca942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bc97f66ed9de3a0fb707d623c5880d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82291521b16714ff94580db67d9d5ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e37368c6c6186d0a64a77b06ab980f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91082e380f04b3336f3c03b0f8568c2
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9916c48d25a6f8acd99fc05bc2f9fab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29abfe96ba2365611db70ad8b4c0732d
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4d67c44d89dec8a1e8fee744660229d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 24, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd7f2648812caec7639b727c334ad84b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d67c44d89dec8a1e8fee744660229d3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_25ccf9027ea5b41e9a4ffbc661d29816(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 192, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e98c7ede9c98bee452aab0204ca7b64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_25ccf9027ea5b41e9a4ffbc661d29816
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d26177b2b60a62d54c7af1def0b14ecb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 112, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa48d5195055198046a480bf8db116a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d26177b2b60a62d54c7af1def0b14ecb
        def get_inputs(self):
            return [
                paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b8d26116a632debd3566daf25b9d4bed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 128, 16, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c9a6e7bd4ca0bdb3d466e00c6cd8dba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8d26116a632debd3566daf25b9d4bed
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6665c59035be4e7209bec327f0bd8902(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6099e4f50943f062b963cd546f3dc986(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6665c59035be4e7209bec327f0bd8902
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ad1c1fa469da4a4b0f59f93a2ae2e2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 576, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4741c40f8a49742e257b10ec153f1396(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad1c1fa469da4a4b0f59f93a2ae2e2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e4b40b830281e5514c26381c64312c34(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4633553da12e6845c50f18c4cab27c06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4b40b830281e5514c26381c64312c34
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6799aed57e645ff67ddc08e49e9e34e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98caa6ea245912c29407d7a8ba8dcf49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6799aed57e645ff67ddc08e49e9e34e0
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c6a991dbaf5eef18d6d50a0f4a1ed3da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed694f25305943bc722cf0be14561294(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6a991dbaf5eef18d6d50a0f4a1ed3da
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_62205c11e6c9821e543fd042aa375066(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce25efbfb2fb492ce8c76db8e00aad01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62205c11e6c9821e543fd042aa375066
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_30089928ce1b22a230e6039d8b3bf845(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a068928ed6ee9305b320fa31be4dfbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30089928ce1b22a230e6039d8b3bf845
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7fac7c92fa935dea23e327a88ae63bfc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[128, 320, 8, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8af10535195872a50fd287d551dab74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fac7c92fa935dea23e327a88ae63bfc
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_79587a3d6cbc657793787ff07c3b06bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e7169a9a986b3b680df913e143740a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79587a3d6cbc657793787ff07c3b06bb
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57c0e3dc6d67fdc931fd97e4d74f3ef4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 7, 7, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db7576baaf9ab70c1e35759f946ee871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57c0e3dc6d67fdc931fd97e4d74f3ef4
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6b089a6a425b3ef606a7de86143ee034(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_69a99726418767aa9159f485dc2a97c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6b089a6a425b3ef606a7de86143ee034
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f763d1f6e2514461b59878aa61866d23(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 512, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68ac30ac8e7cfe3c9c2163a0bb5c1cc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f763d1f6e2514461b59878aa61866d23
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9ac7879e6fe045a2f360c4b0e78520d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 80, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87ebc1a69630199b15cc9f4ef6c1086d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ac7879e6fe045a2f360c4b0e78520d2
        def get_inputs(self):
            return [
                paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eae5407320b360193342446ee9666842(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b5a05d7fd666ffb739485109021dde1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae5407320b360193342446ee9666842
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b911e2434e58338310b82c1d265877e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[528, 4, 96, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fdff5c88561922e94f1c5288578936b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b911e2434e58338310b82c1d265877e0
        def get_inputs(self):
            return [
                paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_976c6a4a831f27913a3ecb900626dec4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40ed4420de619c947b679efa24f4b22b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_976c6a4a831f27913a3ecb900626dec4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1f0ae68adc9e51b04a633f1e24081d5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[12, 288, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4dab3bb47d210c520f81db0e55960464(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f0ae68adc9e51b04a633f1e24081d5f
        def get_inputs(self):
            return [
                paddle.uniform([12, 288, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_39321de6eea6f79e0d5f794d5a83530c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 320, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_196141d3fac2e4f18274af675028a1be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39321de6eea6f79e0d5f794d5a83530c
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9031c55248eae18ad6baf1faad3c0abb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87b0c09fe5381fac5c65f66ade9179d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9031c55248eae18ad6baf1faad3c0abb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a76c12d72d83f631baf6c5f0714aa49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 160, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71591f01d71b8c1560d65e03d7f824b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a76c12d72d83f631baf6c5f0714aa49
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_917f83110a9405438ec3ff90c32310fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a41ab33e90e82c5aca7a5aa1de98c205(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_917f83110a9405438ec3ff90c32310fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef43aef1a8fb0d67f43b03b6fe736d66(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_591697c744181846d1d8c5f722c137f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef43aef1a8fb0d67f43b03b6fe736d66
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_234f569db702339b9ea6c853951f0a01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 14, 14, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a250f2a5b48e7310cd323e4381418980(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_234f569db702339b9ea6c853951f0a01
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3cc620fc632abfa9982e98a1aa54a444(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 64, 32, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd8425c912ffaedee9197701b3ed16fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cc620fc632abfa9982e98a1aa54a444
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_77e730606d400d33f4a4fc1614b98e4d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[384, 2, 96, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34500ce4b2709a93ebcb413d2d49bb25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77e730606d400d33f4a4fc1614b98e4d
        def get_inputs(self):
            return [
                paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e3e31fb3e8899e0b9f2c460726fdfa6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a470430104eda1b8804b987f7c06fd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3e31fb3e8899e0b9f2c460726fdfa6b
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2cde8c6330264e55af28a9f770588861(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 28, 28, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f76572c93a52d67c7fb19737ceb75abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2cde8c6330264e55af28a9f770588861
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_347dc159b106c07fdbc86c60a5556bc3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 128, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c7eb7ca5c3f8dca78db096c88a0025c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_347dc159b106c07fdbc86c60a5556bc3
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_42b30f79bdf2673a5513a706b97b6e77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3dc49707c8865e84e70cb17bc3c7df71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b30f79bdf2673a5513a706b97b6e77
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e7a49a38e65dd905a8ce7ef89a639e9b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 112, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d16392b52821f57af9f16d1c93548359(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7a49a38e65dd905a8ce7ef89a639e9b
        def get_inputs(self):
            return [
                paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_44127c87d5c08541c4202e1e5a70bab9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 20, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f9255dc4fb73f4608cfc63dca41a16e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44127c87d5c08541c4202e1e5a70bab9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_53540ee6e88d79ee1da0ed7848e437bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 40, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e528dc70e8164b49c9a97d5562f1cb89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53540ee6e88d79ee1da0ed7848e437bc
        def get_inputs(self):
            return [
                paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_11b0fabca25cf2cadecbe7a98e961a69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8da049a222bee72e5f50f5cdfdb717b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11b0fabca25cf2cadecbe7a98e961a69
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_13202ffd62aadcc0a49a68d66bdcc769(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 56, 56, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_051297aeaf66327c5ecd6453907dda57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13202ffd62aadcc0a49a68d66bdcc769
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2cae298628806b99ac6f8a7500e4e7b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 24, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b75e85585da2b515795d9497de9286b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2cae298628806b99ac6f8a7500e4e7b3
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9d1fe298e2445feaabad4367db429bd4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 160, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_35a23ccdefb610a651b6ca5b5b174b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d1fe298e2445feaabad4367db429bd4
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7b56d5e5cf9fafe4715b058ac8b159c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5383685088db4f2dde209bd46b379b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b56d5e5cf9fafe4715b058ac8b159c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68ac30ac8e7cfe3c9c2163a0bb5c1cc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f763d1f6e2514461b59878aa61866d23
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5b14e06f0519c12d5411862dba27bddf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_320b3d11d67fff8a0cb8573e83853825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b14e06f0519c12d5411862dba27bddf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9f67a85597f6a4c4760e32189e16d2d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 768, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2a98c636144e38e448c78269a493c6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f67a85597f6a4c4760e32189e16d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07c9c500850f2e9081ca4954ae728c95(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04bde52da0d5c067262d727afffb23c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07c9c500850f2e9081ca4954ae728c95
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d5a504287626fc37658aa89206231c73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98855d4a7aecfe90ada740311e1e3461(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5a504287626fc37658aa89206231c73
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d33a42e4dbc212d5ab74d9df33555531(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_536e2cc40c819c159bc174b2f693f4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d33a42e4dbc212d5ab74d9df33555531
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85f8c84a44e378baaf3b775cdf9d25b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 192, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2802dfc8f8ccd327066c98e93fba04a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85f8c84a44e378baaf3b775cdf9d25b1
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5e3f303a880dfec54efd1e6ce524fb53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20, 8, 288, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf039886743551cc7544636540c62b5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e3f303a880dfec54efd1e6ce524fb53
        def get_inputs(self):
            return [
                paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a55300ceb9f4173bb4375c112935e473(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ef7a0510330f3ac45672ceae92a82b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a55300ceb9f4173bb4375c112935e473
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d9e34230382d66ed417803268136e0c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_106ed9d5ad23367bce37e2d6c9e0f790(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9e34230382d66ed417803268136e0c6
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6d8ef637127dab5ecaf5e0e2d66172b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 7, 7, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cca6058c63ea0ece1cc45ea3f6ddcc96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6d8ef637127dab5ecaf5e0e2d66172b
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f7cc88c819b5aa3c30f3083b14b998a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 1024, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7da66e4b94ea6a47f9bd309e7f245a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7cc88c819b5aa3c30f3083b14b998a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_80611239a21cd94f65af5a4d9f241df4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1e3f52a233199eb39f7ab211be374c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80611239a21cd94f65af5a4d9f241df4
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_84e38eb6b782159d4260b72ba5da3690(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[576, 2, 96, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d78ab66f7d9aa959967603f3d9324572(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84e38eb6b782159d4260b72ba5da3690
        def get_inputs(self):
            return [
                paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0515fce24ea6e285127f33af07b20560(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 144, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d1310f8a3156f0a23b1d63e6d2ccdf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0515fce24ea6e285127f33af07b20560
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d07b44590498ebd74717e902e0b0196(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[96, 4, 96, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a00dc7d0ee38dcde981f82f782f41b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d07b44590498ebd74717e902e0b0196
        def get_inputs(self):
            return [
                paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3840cd337b4a46e281a6c306829429cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[12, 8, 288, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b4a884baf38c626cbba4799cbc5f535d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3840cd337b4a46e281a6c306829429cd
        def get_inputs(self):
            return [
                paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d639b78e58276b289f9856076ad947ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e41933aa87377f19593c695d937db9e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d639b78e58276b289f9856076ad947ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98855d4a7aecfe90ada740311e1e3461(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5a504287626fc37658aa89206231c73
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e44474fc05c7bf9b97a9de530319732d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 32, 144, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc21aa8146d31fc22e7087953ae21a17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e44474fc05c7bf9b97a9de530319732d
        def get_inputs(self):
            return [
                paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3f74fd119cb80ca3ff9cd746d45c89b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[960, 2, 96, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8cdce15257541c7f623cf4f312bcfcb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f74fd119cb80ca3ff9cd746d45c89b
        def get_inputs(self):
            return [
                paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ca11cb3d1ac1f0389a5815ce09d53f47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c808f7c0a1d8f83e804dec7fa19e0359(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca11cb3d1ac1f0389a5815ce09d53f47
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e9ceab9292961d2e6f5fb38782f323af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 256, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c3416e9b4dac1f0eab2cc7bf7be8126(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9ceab9292961d2e6f5fb38782f323af
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c71daad9fd41f7fdbb2dd32fc62fd6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2112, 2, 96, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42e2225a59bc52f6803a28a6aa798a95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c71daad9fd41f7fdbb2dd32fc62fd6f
        def get_inputs(self):
            return [
                paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9d202f9c0d7c854f6a0bc64001a476c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 28, 50], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b45dd149ffe635b67ec0121f5954f139(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9d202f9c0d7c854f6a0bc64001a476c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 28, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a69f1d6a25f5a87dd9ee0eeecd022e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee93fc5dcbb94f2ddbbcfe427300719d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a69f1d6a25f5a87dd9ee0eeecd022e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a0b3f9255cbb4b8adccfdef554c0f7f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3bb92e11c89c6a3dd8718a52483b0874(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0b3f9255cbb4b8adccfdef554c0f7f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bb92e11c89c6a3dd8718a52483b0874(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0b3f9255cbb4b8adccfdef554c0f7f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68ac30ac8e7cfe3c9c2163a0bb5c1cc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f763d1f6e2514461b59878aa61866d23
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8da049a222bee72e5f50f5cdfdb717b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11b0fabca25cf2cadecbe7a98e961a69
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_051297aeaf66327c5ecd6453907dda57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13202ffd62aadcc0a49a68d66bdcc769
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7944717231d46aa12484945517f4557d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7758f64a9796063221e8f0db5e7bda64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7944717231d46aa12484945517f4557d
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87895b9c881c8c904c1213a7c4d16d99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4801aebb499a81d427bfabe315d8cfae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87895b9c881c8c904c1213a7c4d16d99
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b1bf77faff2a2f5565a873df0278f4d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ed1dfdfd5db38451284e2896830b2ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1bf77faff2a2f5565a873df0278f4d3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_faf0d9395e4d0081d27278b173e20e39(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5fa7cb8dc7946fdc3c484729b93e7bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_faf0d9395e4d0081d27278b173e20e39
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_50b784f5af30929ae24903aca1cc317b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_efc13d9c59829d76fd20f70540c21d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50b784f5af30929ae24903aca1cc317b
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7b99183024edc99ad817a58a73b13c29(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 28, 28, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fb30c84694d5342805b827c6bfa22b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b99183024edc99ad817a58a73b13c29
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_576b704420d4be7c13042b171b2ce119(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_289747c7571636543aec3bdc7b51bf87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_576b704420d4be7c13042b171b2ce119
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_41c9e48a7172595a41b3c037a889ef74(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18f7c3a5bac90076a192d055187a3cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41c9e48a7172595a41b3c037a889ef74
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ca89d6b6ea35029916d61aec56f339d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 24, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36e91b2e972f287130b022dba72919ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ca89d6b6ea35029916d61aec56f339d
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_591697c744181846d1d8c5f722c137f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef43aef1a8fb0d67f43b03b6fe736d66
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a250f2a5b48e7310cd323e4381418980(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_234f569db702339b9ea6c853951f0a01
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fd32c812e006b0ac711f5df39edb6544(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aba1675979520a7b194be2b3d4f158c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd32c812e006b0ac711f5df39edb6544
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_32ba00ac38d03f782f28422b920bbf99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 64, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3085a905084ae02bc01779981aaef483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32ba00ac38d03f782f28422b920bbf99
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f739caf2d9bd695da01e808e0adb4be2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 464, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fdda8924f5d71328c6da63849e98cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f739caf2d9bd695da01e808e0adb4be2
        def get_inputs(self):
            return [
                paddle.uniform([1, 464, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5383685088db4f2dde209bd46b379b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b56d5e5cf9fafe4715b058ac8b159c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68ac30ac8e7cfe3c9c2163a0bb5c1cc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f763d1f6e2514461b59878aa61866d23
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5383685088db4f2dde209bd46b379b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b56d5e5cf9fafe4715b058ac8b159c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a8ca37ffdf12ff3e320d38c0678b220(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 256, 8, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5738d8a381bd9c53fcb3df0173c8975e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a8ca37ffdf12ff3e320d38c0678b220
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9c8b1ad4dfd27a5931c6b2b3e52a318a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1f33e92d970f4faf73464e10bef3eccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c8b1ad4dfd27a5931c6b2b3e52a318a
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5eba2e08bea5de17e7fa7aa609c99785(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 14, 14, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32c76a063051f1a34da8f3996624c760(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eba2e08bea5de17e7fa7aa609c99785
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a04d30db8ec635013254a3f9a578452c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 8, 288, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81f5c560dbf097f5c1286ef5999bc1ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a04d30db8ec635013254a3f9a578452c
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0c9cb57f3f479d35754a94dec0382c60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 14, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a5b80b2bdde2c7cc56bbd8c1e9ba9c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c9cb57f3f479d35754a94dec0382c60
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 14, 25], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_175bc6dd4c5ebd8527cc50568de63094(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7611c736b641dd0dc4219072ec567c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_175bc6dd4c5ebd8527cc50568de63094
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4cd7a66f401aa4eb57177df8a0c88c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 56, 56, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6042276efd5b1730f5ebc0ebf188918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4cd7a66f401aa4eb57177df8a0c88c4
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7a65e6ae8001406a4691d0edb86b7905(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82ff7ae7f4d89cc55f8433c8f70f3d06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a65e6ae8001406a4691d0edb86b7905
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a6c8325ef5c78e7a67d9db9c6202efbe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[240, 4, 96, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aafa37e29d67c69cf2045daa869b5d01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6c8325ef5c78e7a67d9db9c6202efbe
        def get_inputs(self):
            return [
                paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23cd9b3d09cbf6fd6df04d978b485d2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 32, 144, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8763a8de21c13b76e82fc8f55a6eea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23cd9b3d09cbf6fd6df04d978b485d2f
        def get_inputs(self):
            return [
                paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536e2cc40c819c159bc174b2f693f4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d33a42e4dbc212d5ab74d9df33555531
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536e2cc40c819c159bc174b2f693f4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d33a42e4dbc212d5ab74d9df33555531
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536e2cc40c819c159bc174b2f693f4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d33a42e4dbc212d5ab74d9df33555531
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f29da53bde1633f847248479769dca3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33a826f5e862dccf6ffd172f1a9954ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f29da53bde1633f847248479769dca3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efc13d9c59829d76fd20f70540c21d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50b784f5af30929ae24903aca1cc317b
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fb30c84694d5342805b827c6bfa22b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b99183024edc99ad817a58a73b13c29
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a4d61c3f12475b0085ce9dce67e98ff3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d49cce27b82a923acd244346739da44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d61c3f12475b0085ce9dce67e98ff3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e9349ae45e949305b10de7309bb5bf3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06b6acf932291163c899bb6a78473627(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9349ae45e949305b10de7309bb5bf3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_634d1f16fa3baa69114d4f7b837149cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d598856a064d459b0669433e5d1578af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_634d1f16fa3baa69114d4f7b837149cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6fd5b6c407b74514752ab8c46180d613(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a57ab9929e377654fe5393d5e5069bf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fd5b6c407b74514752ab8c46180d613
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34a0c360337494c272b1977806c8f0a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 8, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a803d6459b571d0772bf09369f72f9e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34a0c360337494c272b1977806c8f0a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf0b979da857982edea39761bf665bb7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d8189a53790de5493301e52b6f00351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf0b979da857982edea39761bf665bb7
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536e2cc40c819c159bc174b2f693f4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d33a42e4dbc212d5ab74d9df33555531
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18f7c3a5bac90076a192d055187a3cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41c9e48a7172595a41b3c037a889ef74
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18f7c3a5bac90076a192d055187a3cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41c9e48a7172595a41b3c037a889ef74
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18f7c3a5bac90076a192d055187a3cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41c9e48a7172595a41b3c037a889ef74
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5382aea9faab07bcb88c4e295a948466(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95711296fd870b1ced7587d917050334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5382aea9faab07bcb88c4e295a948466
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f33e92d970f4faf73464e10bef3eccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c8b1ad4dfd27a5931c6b2b3e52a318a
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32c76a063051f1a34da8f3996624c760(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eba2e08bea5de17e7fa7aa609c99785
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bb9ed0198119e146527becc13dfce3c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 15, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_901237c79814876332a25ef77e88cef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb9ed0198119e146527becc13dfce3c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_591697c744181846d1d8c5f722c137f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef43aef1a8fb0d67f43b03b6fe736d66
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a250f2a5b48e7310cd323e4381418980(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_234f569db702339b9ea6c853951f0a01
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04bde52da0d5c067262d727afffb23c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07c9c500850f2e9081ca4954ae728c95
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8ac07604234bdd7f1f9fa191feae1760(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[44, 8, 288, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_827a75a3fd6220528b74cb70187e9594(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ac07604234bdd7f1f9fa191feae1760
        def get_inputs(self):
            return [
                paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_03c2e8ed1ed149a55fc38d4f6d0da02e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 80, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e7f159d70fd61cefe582f95c6e69535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03c2e8ed1ed149a55fc38d4f6d0da02e
        def get_inputs(self):
            return [
                paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7611c736b641dd0dc4219072ec567c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_175bc6dd4c5ebd8527cc50568de63094
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6042276efd5b1730f5ebc0ebf188918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4cd7a66f401aa4eb57177df8a0c88c4
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536e2cc40c819c159bc174b2f693f4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d33a42e4dbc212d5ab74d9df33555531
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc3b21ebcf49eacf51493a573bd9f30e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ddb36bcec8d277e466ea2009e1a492e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc3b21ebcf49eacf51493a573bd9f30e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_30174c88ccf00ad569860ac4ec575547(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62d4ab36dcc5937b7804c5668a7becf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30174c88ccf00ad569860ac4ec575547
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e7169a9a986b3b680df913e143740a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79587a3d6cbc657793787ff07c3b06bb
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db7576baaf9ab70c1e35759f946ee871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57c0e3dc6d67fdc931fd97e4d74f3ef4
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_106ed9d5ad23367bce37e2d6c9e0f790(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9e34230382d66ed417803268136e0c6
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cca6058c63ea0ece1cc45ea3f6ddcc96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6d8ef637127dab5ecaf5e0e2d66172b
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b7aed16af5932ddacbc0284c562b52e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_906e277818be8ac7a192cb62e4a31968(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7aed16af5932ddacbc0284c562b52e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06b6acf932291163c899bb6a78473627(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9349ae45e949305b10de7309bb5bf3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8b01cd976377e3fc1972f69bd28c7f77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[144, 4, 96, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53c16317e898f4cf250ea36b9729b6a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b01cd976377e3fc1972f69bd28c7f77
        def get_inputs(self):
            return [
                paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed694f25305943bc722cf0be14561294(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6a991dbaf5eef18d6d50a0f4a1ed3da
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce25efbfb2fb492ce8c76db8e00aad01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62205c11e6c9821e543fd042aa375066
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2b818cd8e6ceed47faa6ab5a3dbea2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_380457b7f0f56c0e65e573089141c86d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2b818cd8e6ceed47faa6ab5a3dbea2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e479961a049cb19f5341b38f8e3e5a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a5ac6fc1b1e71d2b02877e1a5f8b95eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e479961a049cb19f5341b38f8e3e5a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ef7a0510330f3ac45672ceae92a82b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a55300ceb9f4173bb4375c112935e473
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_685363c764d31704c5aeb6951b17d0b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_688c0a8bfaf3e825942eb2fc8c6c2583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_685363c764d31704c5aeb6951b17d0b2
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb98087f9fdddd053b5db062e3f6c495(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c000f23a4ded039ab661db6408fb3fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb98087f9fdddd053b5db062e3f6c495
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a470430104eda1b8804b987f7c06fd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3e31fb3e8899e0b9f2c460726fdfa6b
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f76572c93a52d67c7fb19737ceb75abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2cde8c6330264e55af28a9f770588861
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c089c21c85f2aaa34a132c06f4b67848(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_766d6da323661fac4fa23a77c2ec4d70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c089c21c85f2aaa34a132c06f4b67848
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2e9feb4b7e61288e3881849d9e46443(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 232, 32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9bac60ea1b2d31482cf98cb063e47872(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2e9feb4b7e61288e3881849d9e46443
        def get_inputs(self):
            return [
                paddle.uniform([1, 232, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8da049a222bee72e5f50f5cdfdb717b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11b0fabca25cf2cadecbe7a98e961a69
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_051297aeaf66327c5ecd6453907dda57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13202ffd62aadcc0a49a68d66bdcc769
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb0984e94dcbf7cfc2ab811773bebbac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 40, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b20513919b7f516f7ad2230caecd8e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb0984e94dcbf7cfc2ab811773bebbac
        def get_inputs(self):
            return [
                paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23077b2b008d58b9fbfe25ea1bb9ce4f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47621cd6327cd0107107d249ff9b3f46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23077b2b008d58b9fbfe25ea1bb9ce4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5425dd5c50c8555d48578d2619bca09(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 512, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_382638f8f31936a09e8cf5e6d11b0e95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5425dd5c50c8555d48578d2619bca09
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18f7c3a5bac90076a192d055187a3cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41c9e48a7172595a41b3c037a889ef74
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fdc056b9a7ec6d3bbae045a51c80e1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 512, 4, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f880e512886c583351913d4a8cefe42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fdc056b9a7ec6d3bbae045a51c80e1e
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f33e92d970f4faf73464e10bef3eccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c8b1ad4dfd27a5931c6b2b3e52a318a
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32c76a063051f1a34da8f3996624c760(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eba2e08bea5de17e7fa7aa609c99785
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf7daf5c6edabd33706f093b44104e73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a1ef79014ab00828a24e8cec3062385(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf7daf5c6edabd33706f093b44104e73
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f35a900f2195876fa1567e192c92bf5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_add996c1898d8a6f93b6c9f3e38ce5da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f35a900f2195876fa1567e192c92bf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18f7c3a5bac90076a192d055187a3cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41c9e48a7172595a41b3c037a889ef74
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1546b091e2e42235e2c9c1b07223bb76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 256, 8, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6500197706a7f403032e67cce9640ac8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1546b091e2e42235e2c9c1b07223bb76
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bb92e11c89c6a3dd8718a52483b0874(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0b3f9255cbb4b8adccfdef554c0f7f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_00ad02fe9c1d23a9dff35be0eb8cff8e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 116, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36d3e30b182ffe639e8cfcb22be425b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00ad02fe9c1d23a9dff35be0eb8cff8e
        def get_inputs(self):
            return [
                paddle.uniform([1, 116, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eeb128252044d15ae7adb3bbe4b755ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c42d0360964a9b569dade340ca8bb20b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eeb128252044d15ae7adb3bbe4b755ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2a98c636144e38e448c78269a493c6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f67a85597f6a4c4760e32189e16d2d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e6a58dc9a4d3335d1c84f5977dec3593(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cddd37b194512140d13c39a17362cdf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6a58dc9a4d3335d1c84f5977dec3593
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9f5aeb70777390348e9787b4fa78cc5b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1362fabe73cdfd3239043d775548eff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f5aeb70777390348e9787b4fa78cc5b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7611c736b641dd0dc4219072ec567c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_175bc6dd4c5ebd8527cc50568de63094
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6042276efd5b1730f5ebc0ebf188918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4cd7a66f401aa4eb57177df8a0c88c4
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52ea481b28be43722d7024ae155e451f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e586ccabd555c6de3e9035e85131eaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_753f32fd2d04b4713dd2d3e9aa57e0a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e3e080e2db31f977178d9aa8a68615e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9125974c873dd7e327d33dea3cfc64d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3735a25ea0bb7b4ece1dd9ac2d0b67fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efd1853b9820068883ad373bb880adf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a91fde7eacf6e7dbc7fa0d45586744d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97c46c190a182c5b72306e94d1abbcf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45d5d9dfe74280697699f2ed68ffe28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38d2df78eeac7a6f0f5c736eb276bf7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b809090694417f7d129a4f8c02c04f00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc17eb8dc5b0df0dd09961d70315e713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bc8254ad67a1b8aa39c647dc38c09f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_139726f9ad93d3792c52933c4f454c29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c022b15aeb0779c03374042a4a12875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bce0074d61f842652fa88a1b125a9850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23dac26063e0f7718091511ff4cddceb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9d94fcaf6f3a1005815205e2416633a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68007f55ff30493f4d68864fe87a6e7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8995400aae4bb8e2f603d18e87502325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([12, 288, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b21900cde4efd4b4df806c3bfb2c6160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8693d85dc5bd6f20e0f8cc0673bf82bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d0cc6d80e2fcd98c983658070880517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f63eb60cce363fae8e3667b0c500721c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_564847227ff36976a149d0a047a9e9b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fd8a9999d79d5fd6118a5fc689aee12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c48f4798d20efa9928c730f86528de6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f62111b52f3fd9fc6cf4a7c74d4d7f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe34d78f1a66083fe42e3f1c0cc0f833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72b451199fd571e9f567f82caa3eebf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2d50a64e1094895e28b5b5ab71c91a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6f6f4a771bec704459d45ed44ae388d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d335b97fa3be2e98755a1d4981bd7a99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f64840f089860f9ecead906ea50b7f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ceefde63f6d11126635804f01040154(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7677783965fa32e1c1d116aca9cb03ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5901681935408c5ac9f1663b61ace0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_124b4c75930a412c8d6d1ecf220eeca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7cb9a4c784253782205754026287fb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6b439ecb13a2df6ef9569e2ed65f74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c022b15aeb0779c03374042a4a12875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cdd0c2254946073a7616bdc1b9d8ebf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f39900f4922798de3ab6617c00ca942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5dcc20532bcedb2a475da9865adc334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a680243a70bc31c6e04b622618557506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307bd7b03fe9f08998748723c31b5231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f65b8d8dd8ca9f1a3d90d6537840086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bc9de8ea12ba79d6df9b75723e9b1fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_627ca259f30dc36897f7fd2bcfec0f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b99abab13ba25a2ba303d6d8109fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f1a957906585c64bf6a45e9876440be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b340ea0ccd75cdc449e3720377c10e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb3548f13b00eda9c37f4bb4306e7a72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee8e914a41aaed7819739d17263c564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f90ee800696d75c47b9ac37a88455d64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2681e6e0e4141639fb4c9c5a215f7ebe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85d262590c839e4ca17a78beb3d3f6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07bddfc9864b82a8c04fc8e4fc604c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a680243a70bc31c6e04b622618557506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a502a6a696ee46b162694bbe6942d936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93135260365be33e676c83dfe7394abb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0144da1bff07cf6245183c29139d317a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d53c15942e2e0cd94d9780685e51145(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cdde27ae7ff5c51ade019b56c9a8135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_337c7d2d3c77aaceb81930cb51e6135c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 28, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d838ad487e5493849a14be5b1d7f61c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af61365688a1cfb8c58cffb9f2387236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af61365688a1cfb8c58cffb9f2387236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c022b15aeb0779c03374042a4a12875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7677783965fa32e1c1d116aca9cb03ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5901681935408c5ac9f1663b61ace0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa88130b2b655e9a0e84e113ba58ae80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0862bec01e16568d837abbe17943a6bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6878052bd3b48b73ce9a3b1b528fa2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_511bb01a56a8bcac581b7dcdee0f397c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7f263871293d01065852aaa3abf24c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba9f0bd257e41b8f35b5560135c603e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad2d55375656460f31bfa34d7a52bc4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ac23e1f2934c09fcf7af4f4684e7858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23b4364dad0b39e15bc0873c019c619a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_564847227ff36976a149d0a047a9e9b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fd8a9999d79d5fd6118a5fc689aee12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beae18c044e3923f4b80e38f6372349f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_241015eef8cfb6f3411c9ff951e69cb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_121f7cc0749f52826e35ed619cfe028f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 464, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6b439ecb13a2df6ef9569e2ed65f74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c022b15aeb0779c03374042a4a12875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6b439ecb13a2df6ef9569e2ed65f74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3782f755fe2345e0bc31b4212ebf43f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b563f47208b774e7a5fd294a0c98a088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04cb7ab76e21569061579902f6b21844(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba3b704514e713a671f0c42c2573db25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_868dcc9778512f6b70bdcd09a61394d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 14, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a58a07cf188b9a1fe0a5a58cd904733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9a851ca4bd8e9cf48fa133ce89e630d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f8d750fc9d061c0c92732a7721c25a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d50d2357e8a27f72b59b4eb095b01c43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6adfe4c1b2145f874a82d8b2176cfdfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307bd7b03fe9f08998748723c31b5231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307bd7b03fe9f08998748723c31b5231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307bd7b03fe9f08998748723c31b5231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f19bc26863df8e4a0a8b3248fbe52da6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7f263871293d01065852aaa3abf24c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba9f0bd257e41b8f35b5560135c603e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3da78c7c4bc41b5866121cb284d57903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4b77f94fe0b26b063666ad049da6b32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6336f943dbc6991a7320af2ab89f7774(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_984640033ed5ac36ecb8c5c3b79722d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c21cd8bf5bacf2b0dd344f54fcceee6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5bfa59266db2a75c004b7110bca6b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307bd7b03fe9f08998748723c31b5231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ac23e1f2934c09fcf7af4f4684e7858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ac23e1f2934c09fcf7af4f4684e7858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ac23e1f2934c09fcf7af4f4684e7858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b7206ed3aedfe2769672c54d6ac88fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b563f47208b774e7a5fd294a0c98a088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04cb7ab76e21569061579902f6b21844(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0ec9152bb2e0de44562899bf0854b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_564847227ff36976a149d0a047a9e9b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fd8a9999d79d5fd6118a5fc689aee12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5dcc20532bcedb2a475da9865adc334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e1a4b286ef3b50e8c7b4da5638c29d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b40f81bdd9acca9e73a0eff9be1a3c5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a58a07cf188b9a1fe0a5a58cd904733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9a851ca4bd8e9cf48fa133ce89e630d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307bd7b03fe9f08998748723c31b5231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9f14137520382e1de5f84d0f8569d76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ae406ac0225ba346e2ab0ebec567be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc17eb8dc5b0df0dd09961d70315e713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bc8254ad67a1b8aa39c647dc38c09f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b99abab13ba25a2ba303d6d8109fc12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f1a957906585c64bf6a45e9876440be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4c45847a43bd5cc458613727279239a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4b77f94fe0b26b063666ad049da6b32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66d7266171c7201794998353bd49b934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97c46c190a182c5b72306e94d1abbcf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45d5d9dfe74280697699f2ed68ffe28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c0f62a43370e52b7b6f402ca05c8e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c4ae7e35c4cafdd6e20835864d1caa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_627ca259f30dc36897f7fd2bcfec0f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30616e9fb3cba70b81937113fdf7dbb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de8c3151727af5bcbc1f876ba24c43a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe34d78f1a66083fe42e3f1c0cc0f833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72b451199fd571e9f567f82caa3eebf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cef6a9b806a0cec5ed2dda09e275c1ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb814a40a03cf0c420616eb414c903b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 232, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7677783965fa32e1c1d116aca9cb03ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5901681935408c5ac9f1663b61ace0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdad5595773b25607c17c6e0d8e15ad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25e7b257cdf5c4e4fe956adb6b03c120(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9154f9481ffc4e4e0c75f0504e9f7b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ac23e1f2934c09fcf7af4f4684e7858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28b9d761d199c77c73dcfd6f19edbe52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b563f47208b774e7a5fd294a0c98a088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04cb7ab76e21569061579902f6b21844(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4223c4dbc66083696a78d38c8cd2e389(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29482c494e4bc39db6d7042e1a00349a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ac23e1f2934c09fcf7af4f4684e7858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_938ba5da04a4c8db056b79034751f656(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af61365688a1cfb8c58cffb9f2387236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d4690924f58b06838072e0033aff91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 116, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05ba0a9d60850fee343c24f0846f0790(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f39900f4922798de3ab6617c00ca942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bc97f66ed9de3a0fb707d623c5880d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82291521b16714ff94580db67d9d5ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a58a07cf188b9a1fe0a5a58cd904733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_092c9fe4432d39cc35d4233fa4191624
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9a851ca4bd8e9cf48fa133ce89e630d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae7efcd2c03ab3ba1629f25918fe6c3
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()