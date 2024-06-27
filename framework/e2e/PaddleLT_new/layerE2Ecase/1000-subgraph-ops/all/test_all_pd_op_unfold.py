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
    class PrimitiveOp_5fcd958d01d71fe790d8d8c4d875daa9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33cb9d1dbb35df20b4257638964b358c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fcd958d01d71fe790d8d8c4d875daa9
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_530279e80f3760a954fa7e6d51194daa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a2af3a21c4dd35db77fef4ddfb8dff1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82eafc82e89c283f64d217e747825a4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4c094239445896ea07a2715d8c96786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc8ffa823fe264e5ea2cd856c3e15499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c816eab5464213f892cd6c218aed1140(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59f9131276640ffd8a7d9a777265e5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c733085599db8831cf6aff74ec465fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b44cfc4e5cc31d4d8cb1e1a36a6ec870(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9585bbd7184b77c118177ee6738c22b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41ef279828c0f91940fffd95501e41e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02977327b9bb7f0f567ec02f7e74695f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7215f750c606f3e55f5b950c0874e3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e55f87001eb41e6e22b2a160cd6938e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fcd958d01d71fe790d8d8c4d875daa9
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4379a77ec76c743eedaabab3dc4c8ae9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59ac2827d678de74c5dd48c1c562e0b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_730e381c66f976139f2d4ebe70ee35a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_234668c53b4bf35a6365cc2d79532fd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_730e381c66f976139f2d4ebe70ee35a1
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a4381fee70217dd1d82db3446214d303(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 64, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_50cceafbc71cfaa811109c05af074d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4381fee70217dd1d82db3446214d303
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b8f722b2ed030aacb34774aa90c355e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 64, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f1e7bf606b148872f25c053001da433(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8f722b2ed030aacb34774aa90c355e1
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1273f7bf45f9e7d0f74f2a195ea2e374(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 512, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0631fe5deb3160bb70eeac6ab74ef7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1273f7bf45f9e7d0f74f2a195ea2e374
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2b717444c4fcb6b4d7eab60ac462f2a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 128, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd62a35aa37d1eef093f7215d0faf570(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b717444c4fcb6b4d7eab60ac462f2a2
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c33f9f9bec58c1a7446f2f98cc9c885(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 256, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0236ba4870b5f1a0b3282f9a33d7e4a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c33f9f9bec58c1a7446f2f98cc9c885
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e99752a750080e6e55120a588a139b57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 256, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f01e47de41dc25f7b6689d4348b99f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e99752a750080e6e55120a588a139b57
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3c6b8f6900dc4b4a05d9a02717ee4f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 128, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6dc9b055509c73b8473aa8d44529c06d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3c6b8f6900dc4b4a05d9a02717ee4f6
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fcb432eb16a08cb8d85e2e37bb36e692(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 512, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2eeffda40092e6478c4ffc45b9c76922(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcb432eb16a08cb8d85e2e37bb36e692
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d5d731b43ed5026beaecfbb3280890d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 128, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd8a935f561646784b20cf6c96d7c117(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5d731b43ed5026beaecfbb3280890d3
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_400d1e10a01141fc0c522d667a68a83d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 256, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77cd98de280ccf2b5ce80221dafaac4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_400d1e10a01141fc0c522d667a68a83d
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e43b49c33ecf15b611ebcd5a1aadc8d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 512, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_352eff40ea2d53ea5d59ac62f43d50dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e43b49c33ecf15b611ebcd5a1aadc8d1
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4fccb36af51217bc613439d0420e818a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 128, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a49535cdb1916dcbcb81250ed08e2d13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fccb36af51217bc613439d0420e818a
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_582e009fbe1b828fa6e9e68edabd470c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e60ee2decd6ac7a83a7aad3b69693667(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_582e009fbe1b828fa6e9e68edabd470c
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f46ee4aa8b3aa2292b7c018313bf4130(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 512, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b5df7cf2f3d7ab6d0306a478390a8c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f46ee4aa8b3aa2292b7c018313bf4130
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce09ce1e7233b977934e43073655c0e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 256, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_478ddb6f4cde53b576ee08398adb63de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce09ce1e7233b977934e43073655c0e4
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33cb9d1dbb35df20b4257638964b358c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fcd958d01d71fe790d8d8c4d875daa9
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a2af3a21c4dd35db77fef4ddfb8dff1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82eafc82e89c283f64d217e747825a4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4c094239445896ea07a2715d8c96786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc8ffa823fe264e5ea2cd856c3e15499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c816eab5464213f892cd6c218aed1140(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59f9131276640ffd8a7d9a777265e5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c733085599db8831cf6aff74ec465fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b44cfc4e5cc31d4d8cb1e1a36a6ec870(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9585bbd7184b77c118177ee6738c22b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41ef279828c0f91940fffd95501e41e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02977327b9bb7f0f567ec02f7e74695f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7215f750c606f3e55f5b950c0874e3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e55f87001eb41e6e22b2a160cd6938e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fcd958d01d71fe790d8d8c4d875daa9
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4379a77ec76c743eedaabab3dc4c8ae9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_530279e80f3760a954fa7e6d51194daa
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59ac2827d678de74c5dd48c1c562e0b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b3c81f666a2d0f812f954d3e65ea8f1
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()