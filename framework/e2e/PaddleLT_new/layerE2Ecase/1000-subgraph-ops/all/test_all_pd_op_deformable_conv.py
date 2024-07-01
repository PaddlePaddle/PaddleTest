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
        paddle.seed(2024)
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
    class PrimitiveOp_5e894c1350189a2b1b7ddfd52bfe942d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_69072308237c8df7fa309aa870895137(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e894c1350189a2b1b7ddfd52bfe942d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_77c3b1595d10748356344273ee7e79cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_6560e7ee468d7bfbf9b63d1e3554b902(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05dabccb07a7f663808c6c1a28552876(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a027bfda2ff72be4f8a3b66ae1086d4f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_13c59d2a86369aa571b2066c59bfc590(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a027bfda2ff72be4f8a3b66ae1086d4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7320b589bc1dd8425cdac976c1e7429a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_1dcee9e95b8c50ecd821bd7432aa17b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b589bc1dd8425cdac976c1e7429a
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_158a38e09838b14342da4e3890daf181(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_22cdc86a303f7fe64403da629ac73147(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_158a38e09838b14342da4e3890daf181
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86cf252a0a9e69d0c550adff42b38ffd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_50a51db3fd4af2499bc430041c828d89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86cf252a0a9e69d0c550adff42b38ffd
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cff413ae2342026d3397a538fa86d1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86cf252a0a9e69d0c550adff42b38ffd
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96593cd68a75388a357922176741f751(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b589bc1dd8425cdac976c1e7429a
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d37ca96876178585cfe51c6a7bd39715(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a027bfda2ff72be4f8a3b66ae1086d4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5f6de024941b81f48eff375e9a73c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a01839a4dcfb1362d417cf9515cd0ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b589bc1dd8425cdac976c1e7429a
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d988edd2978b2a09abef90bfe7ff690(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8db3ed4b0ce93ac09115a4125f272722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b589bc1dd8425cdac976c1e7429a
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ef4a345e07b0012773c05e7ba123e49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7320b589bc1dd8425cdac976c1e7429a
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea385269ed4d1cd8f7d6cf881375de67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51ac9744a525fda3fd747832744b8ad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_158a38e09838b14342da4e3890daf181
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd1a9f105de0d7a15f64fd6ec367bcc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a027bfda2ff72be4f8a3b66ae1086d4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_160db1c6014f13ddf7e671723571fcf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_158a38e09838b14342da4e3890daf181
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abf5f7a5936a4c989680783c08fff066(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e894c1350189a2b1b7ddfd52bfe942d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58b02239df20c527e95259af68e1ae9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86cf252a0a9e69d0c550adff42b38ffd
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d18c25669903fb1d63ec617364cd248(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a027bfda2ff72be4f8a3b66ae1086d4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_803b192ebdcb9ea2428ae06b88f12b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a027bfda2ff72be4f8a3b66ae1086d4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e319f2c3829b24de9661b07b3bac605(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_158a38e09838b14342da4e3890daf181
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1c3657fdb0fd253a8437387e5e1d3b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c2cb911010d32bfe8a6c059d384269d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a027bfda2ff72be4f8a3b66ae1086d4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_961e8b436e0ed73349ce806388dd9512(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_28b1dcdefaf1d6ac8d29edde1ddb9537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_961e8b436e0ed73349ce806388dd9512
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5a2b89e9ff7190d32be4350d1c73343(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_158a38e09838b14342da4e3890daf181
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_517af0e569454db7734563130831a5ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccf4a753694eef3ff59c7ccdbeda9993(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11dc0a6913139738986b39b4b4490f46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ada121ff74868734b1a5dd5e76da9f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86cf252a0a9e69d0c550adff42b38ffd
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba08b328ac6b01581d053c1ed47f648e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77c3b1595d10748356344273ee7e79cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4af585cae15f38c2657f298562165744(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_163afcdb8d29eb1794b27bd4d29a5902(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4af585cae15f38c2657f298562165744
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_5ac644b029961b737cb2c60bf7d54d9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9e89bbf3de2a7c40faa3ba044d0d52d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16ea6664c050c78a98b3c4058ef65b9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b6a079d1ff632fd64239128ee3d8d28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8002f78f910f922a6074a1461ac095c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12bd51b5df87b176c655d6f3bfc5f2d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8937b93d9d50215c0a930f75d1405449(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b9ad526ad1b362fb9a5d94fe4d6951a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d5819090f6a67bc676ce296f4e1a260(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_091a04c40b4f8862a082edd8da90baea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e20ee4ea680eb37e0b57204fa65109a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_343d2212d926d91be5a021a57b00c55e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e89acf8a0909bae50d5070b902a4aa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa5d7a609ec421bbab692c2846d658cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d461492360a7ac0c09033bced72c3eb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7baefb2ab5536bb38ec739109afbbe77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c2c7061b16c3d399b26b03c220b79e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccd98dd2d457a7bff0cb6ed4c0f02b01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5fd2eca704777d043c9682c372ecf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2378c1df13ba8cc63de013ebfd5cecfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28d85982d913a6db189ade5510b5d276(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_581dba2772992e010815f47eb67a4270(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2250394e29e859dfde0c972a4078530(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e1b148356dce28d83c509a0145ca4c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06d3b379fed6b690e206d7af4d03028d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27a643a1c0f958dbf96c4b64b4a0fe50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c39133c18d6074692627a7e8b2aab673(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2, arg_3):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            input_3 = arg_3
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


    class TestPrimitiveOp_0b6b2bac68371796d840989d46e9de5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39133c18d6074692627a7e8b2aab673
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fb1c9e5d1dbcc94a539b576bbf8f0b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43d4cc9f22f55ee46e2d11784dc01e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0e9ae7ce60ced77af3135869e4654af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85e9fc958e1b559e561450beca8a05c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b8d8922c077e0959ed78e45dfe5b075(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_498a2b971edb7452b51f39cecb48ffa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85accbcb935cfe1d6658ca26731db2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2b00b4e6f5aff139773a4223c951873(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39133c18d6074692627a7e8b2aab673
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()