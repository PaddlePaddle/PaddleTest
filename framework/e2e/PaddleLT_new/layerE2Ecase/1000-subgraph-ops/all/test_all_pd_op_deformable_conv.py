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
    class PrimitiveOp_b0f9e8325b03145f7e5ceefd75a2ce09(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 258, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 258, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a11fdc67f89895dd87f52c927f349b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0f9e8325b03145f7e5ceefd75a2ce09
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f068fab35668b55e3721e3030e61d6ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4126c3c7ca1aaf4b7c8558322871248f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74e5014acc1295eb0f5e535c9e509faa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8e2343f4f825815a3db8465744a6187(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3ba715543e7f6369143bece95518620(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8e2343f4f825815a3db8465744a6187
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2701751b9e3df77048471071d3a60155(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b7d6c5b6c451c62cc90f6ff42aeeba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2701751b9e3df77048471071d3a60155
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_26db86b8afdc00bc0d82f46f22b4c600(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 258, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2df86a7cbeee72c1e0cd75020ed70de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26db86b8afdc00bc0d82f46f22b4c600
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_563ca04244e6555f4bedbe9af3475739(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6257978dab60060b0bafa5d25761a3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563ca04244e6555f4bedbe9af3475739
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ec4d2f021100588e25c6499d79813ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563ca04244e6555f4bedbe9af3475739
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5bf1ca3aef8d346de02c45c00f5b943(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2701751b9e3df77048471071d3a60155
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_813fc6404cfd7e8639502ecfd97c7f56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8e2343f4f825815a3db8465744a6187
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f95e02bcbd35113022ab0c4bd52a0a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_495a6e98e21031a6967bdfd422420f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2701751b9e3df77048471071d3a60155
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00526e1220331ef4befc7abd66c7d16f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc314f2ed7d727edb259525998530038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2701751b9e3df77048471071d3a60155
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bf8154e78edc7d74fed32a0cae1a208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2701751b9e3df77048471071d3a60155
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4044ccc34fbbb99c4961d567f7e1950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee8f72470bf5dc2d0e2452e4207d24d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26db86b8afdc00bc0d82f46f22b4c600
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1e92451e6428073a5e09220cb9c3992(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8e2343f4f825815a3db8465744a6187
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b988ad833061a8b24e5c24accdbee5a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26db86b8afdc00bc0d82f46f22b4c600
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6166a67668d10c8b84a6eeff733ab40c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0f9e8325b03145f7e5ceefd75a2ce09
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b42a08cdc122978a5d3981e19f2beb97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563ca04244e6555f4bedbe9af3475739
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8e6064c1e7e05f58a7f1dbf81604ee9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8e2343f4f825815a3db8465744a6187
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e96484225d31bc76d2bb34cf957eef45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8e2343f4f825815a3db8465744a6187
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bebb3782b93d89e93150b77cb765e81e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26db86b8afdc00bc0d82f46f22b4c600
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09d5b5187b79712334742a5e2bee9851(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6f9a497482a101b3fef376fc3b3242d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8e2343f4f825815a3db8465744a6187
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_641c96135dd5f23a3204c7b26b694d6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9697f2d6529fa81599e26df010395990(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_641c96135dd5f23a3204c7b26b694d6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1fa5aa1439f789cfbd50ec742939068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26db86b8afdc00bc0d82f46f22b4c600
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b73317968700eb0e434836c58f4d2284(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e71ed40710e64d43323e00cdd558da56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da65740c55c2d57ea76a30b1fc31e9c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be883e148f1d2666f5405ccfecbc141(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563ca04244e6555f4bedbe9af3475739
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7830ff46d816f634d59e701d615955bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f068fab35668b55e3721e3030e61d6ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_376079676ac126dcc1ebfc1cb2bbd6e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24c8ff3dacbdd88dcda717b886ed9184(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_376079676ac126dcc1ebfc1cb2bbd6e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4669e2c0296717908b8f904bddacc74e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 258, 24, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 24, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 258, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 24, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_662b22db6956bb89412b894cec0c4a99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4669e2c0296717908b8f904bddacc74e
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_896b0030a2de8e08df23c9796c9c0e87(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 112, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 112, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 112, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe4824273e55540744e50b1e57e8bafb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_896b0030a2de8e08df23c9796c9c0e87
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_042a3e08a0db28e965038aef2f890176(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1fa860a88df6ba56b55c457bfd8d3ad1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_042a3e08a0db28e965038aef2f890176
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3e0b39c846578ba0883a533fd5b030f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 30, 50], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 30, 50], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 30, 50], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f84dcacd82b9263f4eddd254c7478da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e0b39c846578ba0883a533fd5b030f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4036ed6441f7a27bdf0d4a87f9ab355d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d497edef981b2eff09d0bddc7a318527(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4036ed6441f7a27bdf0d4a87f9ab355d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5220815cf330a336035f2ca7931d3f11(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 258, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c999bbeb1e726bfba74bb01f661f0181(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5220815cf330a336035f2ca7931d3f11
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3c33e37bd5ab5c90db20065fa0492974(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 48, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 48, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 48, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6aef442a0fa0922ac2a2ede2917b921(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c33e37bd5ab5c90db20065fa0492974
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87031fb7a6dc8400d7c0db0d6df79ed8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 60, 100], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 60, 100], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 60, 100], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ad9069759321491e87215b8f585774a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87031fb7a6dc8400d7c0db0d6df79ed8
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_01ba6386adca4dc3b76268a84f0a5b15(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6145a6296926b3cecb300c6040f26941(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01ba6386adca4dc3b76268a84f0a5b15
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3451b361c393f58c9c5df2d932119eb1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 60, 100], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 60, 100], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 60, 100], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a34042a7b5fe92f1e3ca522b917f7eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3451b361c393f58c9c5df2d932119eb1
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0bdeafc0c782d4f7633dfca3c2f404e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83f35580d1ded70ec3e4bcdf3e2b1a9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0bdeafc0c782d4f7633dfca3c2f404e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3d5ea5ba38be9c3748282f92225d866c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_893d06c0549fba3b9d9f152d4493db9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d5ea5ba38be9c3748282f92225d866c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f908bc98d261d5c651d9f3938878f89(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_50f1edb8ab370b95b21ed6d3de5678cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f908bc98d261d5c651d9f3938878f89
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a7d2923e32eda3344475d401aaff8e60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3805befb7b34684f4e7c0d77812343ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7d2923e32eda3344475d401aaff8e60
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8e2e95a7db97260cf8b5607ce654587f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ade29b13effaf82838405f66b1f7082(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e2e95a7db97260cf8b5607ce654587f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1171dba890707aa99a8e9d5dda9b67b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e89b216786247141332a6617ac489fd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1171dba890707aa99a8e9d5dda9b67b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fab13ae92042d1ee8ec57912034f0b72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 258, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d99b5c4ebdca37334d81b79f215aa4a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fab13ae92042d1ee8ec57912034f0b72
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_29dec230e9fffccb88af002157344c30(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 48, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 48, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a758d4939189425a63d56349d482dfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29dec230e9fffccb88af002157344c30
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e58795d13487a8e9bd86ce254bfee541(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 258, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19aa9ea4bc44aec50178682955d333a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e58795d13487a8e9bd86ce254bfee541
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_107d40d5f627fa0b1b9f2144892809ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 258, 15, 25], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 15, 25], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 258, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 15, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77ad8ba61ca1a68a3df3789996fc1eaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_107d40d5f627fa0b1b9f2144892809ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b0366c74cb36158011b50f23b47e9b84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 96, 144], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 96, 144], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 96, 144], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbfe4ba51e0dc638bcd73207ab9097d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0366c74cb36158011b50f23b47e9b84
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d04c256ef99ef0220f2e9632704c384(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 192, 288], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 192, 288], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 192, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51a270313e5e4c5ca4ae599084084818(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d04c256ef99ef0220f2e9632704c384
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2626daa5cb18c7a679eec9d0a9d36140(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 96, 144], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 96, 144], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_553cb96b57973e4ec00f344cf123f17e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2626daa5cb18c7a679eec9d0a9d36140
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_39205890bd13d9ab3f33a7c63a89040c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 258, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8064fab383e3b55a1077e2ee216e2290(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39205890bd13d9ab3f33a7c63a89040c
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_081fb89a35d3b9decd62f611f600b612(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 28, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 28, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 28, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8fb7d21a1db1462b628224959c9f2b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_081fb89a35d3b9decd62f611f600b612
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc953c981481a792e69e20dee8c41293(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 120, 200], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 120, 200], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 120, 200], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_617090f4a113c9ca5f1c7adc3f2842b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc953c981481a792e69e20dee8c41293
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19f9b8fcf9628a8d1b37b6f07247d407(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 120, 200], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 120, 200], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 120, 200], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6393c640c1b4568c816b98312454255(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19f9b8fcf9628a8d1b37b6f07247d407
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f26abcca3dca83ca9e1b4011575a070d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 258, 36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84e871b7965e5a6ae638506abe11c5aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f26abcca3dca83ca9e1b4011575a070d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_547de7d7b2d731a856a92ba9c5813b68(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 12, 12], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ffcc6d2193f61623d4191742f6d49a5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547de7d7b2d731a856a92ba9c5813b68
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d6ca5e045660dd674baf7579b24eb7ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 56, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 56, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 56, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9619b67cf9b749306c13d0c8d5304a69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ca5e045660dd674baf7579b24eb7ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07a2d2dff5b3bbf06a229d3778b61851(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2a552cc0603b08ecb1efee165984dea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07a2d2dff5b3bbf06a229d3778b61851
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa7f4e3e16d88feb529bab31f2cf9e41(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 30, 50], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 30, 50], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 30, 50], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_439a127cc426f198c065fc1b4629c944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa7f4e3e16d88feb529bab31f2cf9e41
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_44e39a046e315492e9e026824f999060(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 14, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 18, 14, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9, 14, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02b142ce31b057efbc599c03bfc1dc42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44e39a046e315492e9e026824f999060
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac2d3ae00efec16ea029a55747dad882(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 192, 288], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 192, 288], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 192, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f23a8377949e9b0f487a879cff023db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac2d3ae00efec16ea029a55747dad882
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a4794a4326af3c2585bd9978b0c4873(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d2a7eed1cb70ca491434146f311fca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55d8aeb8f4645843ed809a768a228197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2b5249ba78c7c94e0a73a9ca24330de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c6bf4f880c03ff2bd76c71f7999bab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_177992dfb655456d3c9ff99d8e3844c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3774215a3fd55193e6b0ea0f61a04dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538043b61b1775314f362f3c73ef9653(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c7a8e4439824076b25b6c74d1c22377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36bcbafb8ef4302c00f4f4751c68c6a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5420f8d8a90afd800f025cfaf91d882(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_012f0a30142553178813e039b5a2bcde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_409587858df96d22f006dab726f8baeb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39ba248c7c5dbcd9ea03eed6b40110a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52c35b1bbea66e6752c685258fb672ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eafff04e425c06303fc61e15cc5de518(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afe514727813ba536d88cb12cf3b7a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9388131fbdbfd9f722175b3f51b0917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e747f2a76c3caa4a8a6b0aee4b3433c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_766e247086c98e0347692c9a45c49418(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43d400c37322f402afa5a552516dd0a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cabe5d4c2fec2a0311f67da28bd8b65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4518a0ad14fe5d2c2f04860f4fc36a7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1310ab1f529eb1932511117329a5e4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ebd7f6233b125745be42b2c6e72896e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_592a50401c7808621c3b9feed2009e23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2773285a21f516d05d44b69d57cbd6fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2, input_3):
            return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ad71b55575d8a0796842ed1f0819d85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2773285a21f516d05d44b69d57cbd6fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e6b1a8833fce67a74b3fd32c99ae54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8a265603db6ae4059c164ecc0aa63f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb6d99c082013ad0ef47709b0cb27320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44843a1985055b629347868e82cc04bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1344a4640e7fcdeba1145fff1d8d23ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b65a69bb572781f85f8b485e5db8a50b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d833af30dd9d21e6c2b142f3e485bd20
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c10f7c46e1dfe263b64718dd36666616(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2773285a21f516d05d44b69d57cbd6fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()