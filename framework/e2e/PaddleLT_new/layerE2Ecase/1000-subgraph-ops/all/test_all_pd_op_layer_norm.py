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
    class PrimitiveOp_74e2f68c5ea845f8bf3cf8a3022ca21e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d0ab105d434e8bbb5558974637f220d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74e2f68c5ea845f8bf3cf8a3022ca21e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_94cd5033d92bda9dd6b39bb9f74153c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d26164145eecfc67a9909e5e02947014(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94cd5033d92bda9dd6b39bb9f74153c2
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_89a597b58767a4884dfaef890ace6422(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff6140d90c37c3148f8ed6a69e575c37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89a597b58767a4884dfaef890ace6422
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bd4466c2ef869b4132ad5c0eede72ae7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17d27910bfe5274098d6a4334e2f7e30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd4466c2ef869b4132ad5c0eede72ae7
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9b905495881efff550ccaf7dac60cb9b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1018f22d10b7c576dc88abb374da97c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b905495881efff550ccaf7dac60cb9b
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ad76a681d67da4a0058b1ec8958ddbfc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_064c1b45450b1409299bb74af23d3e66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad76a681d67da4a0058b1ec8958ddbfc
        def get_inputs(self):
            return [
                paddle.uniform([16, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c338b6d457240937b61aae18eeae763b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_584ce295e2dd398ae820ef207bd423b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c338b6d457240937b61aae18eeae763b
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a04ac451a0bd4ec0b54c6f68fab11912(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_26ffb64c054310de1a0169def8ad2eae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a04ac451a0bd4ec0b54c6f68fab11912
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c37df4ddef12e9cf7732e119e9799d51(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_178e774cf32d47b6bee97dbb767f7c38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c37df4ddef12e9cf7732e119e9799d51
        def get_inputs(self):
            return [
                paddle.uniform([128, 32, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9b2faf6f06e6754109763dbac119bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9b5b52d10cd0835ac152f5a9ece4d7fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6800c7fd8c7d1014f41b72b74954f251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b5b52d10cd0835ac152f5a9ece4d7fb
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e5c97065954cad3b3e78cd1750d74cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94cd5033d92bda9dd6b39bb9f74153c2
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_485ce525749cea333c650a419d2bfa4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1068ce541a857db8788d4a366d7794f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_485ce525749cea333c650a419d2bfa4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_425bf9849af322e86dd380ca2b82c09e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c37df4ddef12e9cf7732e119e9799d51
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_84b860e28cff1e9148ce77c03eea718e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f10aab0633bad5b1e3dd61c191356cfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84b860e28cff1e9148ce77c03eea718e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8dce8064510e88914873b954139fbed8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45b33f663f4b0dcc791dbc0e275ee44d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dce8064510e88914873b954139fbed8
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_82f3735d65ac96e5fd167a76d984cc63(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9bb23a91ba3842162e49ebeade170beb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82f3735d65ac96e5fd167a76d984cc63
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9e2b118b3f6d9cb0c2082c3cbef8087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4118129924eb5ac279c99fcaf51aaaed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48b01fe608c9e6ebca6df306259c907b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4118129924eb5ac279c99fcaf51aaaed
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8924f585da3de1666a0656bd464c0234(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e9b7016e7e235e64595c40867de7b00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8924f585da3de1666a0656bd464c0234
        def get_inputs(self):
            return [
                paddle.uniform([64, 256, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ab9809b7d82561045b1be59ae49ca373(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f2be68d90d9b93efc2b3d515bcb7c43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab9809b7d82561045b1be59ae49ca373
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26ffb64c054310de1a0169def8ad2eae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a04ac451a0bd4ec0b54c6f68fab11912
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db46801601c4ac6e6e4a5db9773b96e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f36219243492cb05d9155af28c47eb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db46801601c4ac6e6e4a5db9773b96e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6800c7fd8c7d1014f41b72b74954f251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b5b52d10cd0835ac152f5a9ece4d7fb
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3029a11f786923ee1a87a87183b56b62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67d400f30d617698b3cdf48f2d10e973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3029a11f786923ee1a87a87183b56b62
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e4cc896bc4965bdc2c6aafae64b3601(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad76a681d67da4a0058b1ec8958ddbfc
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9f393274a41e13acb590aae116539a58(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29d4ce653c7c982eb137ab7b371d21b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4d20cde61d051cc2423c26128a694dd9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a513dea8ad8b4dc04216c3a5aed0f664(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d20cde61d051cc2423c26128a694dd9
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_969e9099f0a810c148ea7fa891d0ef23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8dce8064510e88914873b954139fbed8
        def get_inputs(self):
            return [
                paddle.uniform([8, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9b2faf6f06e6754109763dbac119bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f70b8f21fb9eb37255ab90580f968528(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f72e1138c5f17d4d5f7af9647e50e1c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f9a2db4e9041cb698f58a5e425657a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f72e1138c5f17d4d5f7af9647e50e1c5
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc6eee252b315ac9c4f2d8a871ab297a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89a597b58767a4884dfaef890ace6422
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a6cc4eeb9aaed173dfcf98efcddb5131(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_306001c3f403015f6f3408eb93620688(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6cc4eeb9aaed173dfcf98efcddb5131
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08f052af702eecfbbe33149a634c8f42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7c05ff1b386841ce556ce9ba0fc4f76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08f052af702eecfbbe33149a634c8f42
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7c05ff1b386841ce556ce9ba0fc4f76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08f052af702eecfbbe33149a634c8f42
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2297415f0f1db17cda56d9ac7354b93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_311c254eb866a6794f02269416c4dd5e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d683c5f22e7fffe03612c578e7fb272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_311c254eb866a6794f02269416c4dd5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b0b96103750669ca788e2d67719fb01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_130a37196d0d0bf6ba5da2d5cf1ef62f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e07be89e7ae4e8fdea21d548a25bb7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_130a37196d0d0bf6ba5da2d5cf1ef62f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b17e44dfe76c40495166f5bd80130fa6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae68819c1feea50ccb00297b1ea5aafe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b17e44dfe76c40495166f5bd80130fa6
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d26164145eecfc67a9909e5e02947014(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94cd5033d92bda9dd6b39bb9f74153c2
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fa4d134ce81749f5d6fd3d324a5ab31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29d4ce653c7c982eb137ab7b371d21b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6b71200d1199262a30c23ad7123fd6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab9809b7d82561045b1be59ae49ca373
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6b71200d1199262a30c23ad7123fd6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab9809b7d82561045b1be59ae49ca373
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93e7b522bc2bb9b9ad69a8322dc22b05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29d4ce653c7c982eb137ab7b371d21b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9032165ba59dbcf95374cff048d1896d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9e2b118b3f6d9cb0c2082c3cbef8087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15ff684a4613b884201ba61e87d6cdda(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d20c478f680ab629177c41283697a937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff684a4613b884201ba61e87d6cdda
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_05cdfe5aaa4b4a7d1b5ce8df892aef3c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5a906eda75aaf971661df97380520ff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05cdfe5aaa4b4a7d1b5ce8df892aef3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_078b1144a40a6aad9477ca3d92797697(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5cd0f48182c1ed48456d0a455bcbf23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_078b1144a40a6aad9477ca3d92797697
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8ecb7fe42036731f4126b636a379f05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_130a37196d0d0bf6ba5da2d5cf1ef62f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5a9de25a966e424746fa4dc23cc8bde6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e846fcf9c4f897ea29b8755aed6d733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a9de25a966e424746fa4dc23cc8bde6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7d4ff74e847df64e065f3a144dffc32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8924f585da3de1666a0656bd464c0234
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d7894a7dfc65bb43517f239a2ae336b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a061febac9f6bbe5ec329cb908fc5a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7894a7dfc65bb43517f239a2ae336b7
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d0ab105d434e8bbb5558974637f220d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74e2f68c5ea845f8bf3cf8a3022ca21e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0de5f059812eea7332f7084d0d38e349(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89a597b58767a4884dfaef890ace6422
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_686abd800a4d2c1f35684a38eaa4c60d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_311c254eb866a6794f02269416c4dd5e
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48b01fe608c9e6ebca6df306259c907b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4118129924eb5ac279c99fcaf51aaaed
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f32ac17919eba6b1bcb7603116156ba8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a00333749ec26a2d035d3aee5d69b85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f32ac17919eba6b1bcb7603116156ba8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7283c92c41ba8ee4767e46de7e6df006(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b0dc8d53cad02868d90b3e1e2e6c6bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7283c92c41ba8ee4767e46de7e6df006
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93f8011bfa8492328f3078e4187dd74e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab9809b7d82561045b1be59ae49ca373
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2297415f0f1db17cda56d9ac7354b93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_72b48fb322962c154afa005887053a84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8abd8c6cd2e13c1b610eea31a359cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72b48fb322962c154afa005887053a84
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c898cf4b5d11bc16fda71d9f11dad8c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb504b365f12deb6830bfc9b9c6b2bb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c898cf4b5d11bc16fda71d9f11dad8c0
        def get_inputs(self):
            return [
                paddle.uniform([8, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee3662caa8a6391444feb66a21bb632c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_306001c3f403015f6f3408eb93620688(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6cc4eeb9aaed173dfcf98efcddb5131
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77b08b4e5d8642706bcdb4f9d7ecfe81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e5c97065954cad3b3e78cd1750d74cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94cd5033d92bda9dd6b39bb9f74153c2
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67d400f30d617698b3cdf48f2d10e973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3029a11f786923ee1a87a87183b56b62
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e9edb91a54fd452cb4dc4d927718b484(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cfbd0dae33636a942a34d4156f34e678(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9edb91a54fd452cb4dc4d927718b484
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d34e512d77d240b91cd72687a529356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f72e1138c5f17d4d5f7af9647e50e1c5
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2961404366d2f7b1c47465ee3be09ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2297415f0f1db17cda56d9ac7354b93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c61ab739e233dda5a65103218e44f496(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 50, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_447227e3f5a6f96121a80e2ce297038e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c61ab739e233dda5a65103218e44f496
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b4032432f4a5247404cc2f443b3505d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6b71200d1199262a30c23ad7123fd6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab9809b7d82561045b1be59ae49ca373
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1068ce541a857db8788d4a366d7794f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_485ce525749cea333c650a419d2bfa4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8dd60e243945dd7e49a013c15089625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db46801601c4ac6e6e4a5db9773b96e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c4de7aa40d52cc2d3bdec2ca48a7f855(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b38e0cca2fa1135406007196ec03cf1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4de7aa40d52cc2d3bdec2ca48a7f855
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a3b0da19b7a948697db0e0452d6616c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ce21a85b03424e398de4a1486fe42d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3b0da19b7a948697db0e0452d6616c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_250c820dab738ddd08e1070a8a58a107(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee3662caa8a6391444feb66a21bb632c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9e2b118b3f6d9cb0c2082c3cbef8087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_250c820dab738ddd08e1070a8a58a107(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a128f2fb376ddb9c17800ee30202d41a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9b2faf6f06e6754109763dbac119bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77b08b4e5d8642706bcdb4f9d7ecfe81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9e2b118b3f6d9cb0c2082c3cbef8087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2411250e8a63ef584e1cbe756f6e80b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_450913fe39704b4b0424ba049617c75a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2411250e8a63ef584e1cbe756f6e80b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9b2faf6f06e6754109763dbac119bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b0b96103750669ca788e2d67719fb01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2297415f0f1db17cda56d9ac7354b93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9efe84b5064927d2cf8da3d8d277e33c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57138edf104d9dd33dc84dc82213b8f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9efe84b5064927d2cf8da3d8d277e33c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bec15f8b17035ae2a961a5176acfa10c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3b0da19b7a948697db0e0452d6616c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8abd8c6cd2e13c1b610eea31a359cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72b48fb322962c154afa005887053a84
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26866393983e56676948e00d04bd260a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_078b1144a40a6aad9477ca3d92797697
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_584ce295e2dd398ae820ef207bd423b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c338b6d457240937b61aae18eeae763b
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afb86535aa13abc34ff62985cf8d4bc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2411250e8a63ef584e1cbe756f6e80b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f2be68d90d9b93efc2b3d515bcb7c43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab9809b7d82561045b1be59ae49ca373
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da3b3d4fb1889faf1557781af5c5067f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab9809b7d82561045b1be59ae49ca373
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17d27910bfe5274098d6a4334e2f7e30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd4466c2ef869b4132ad5c0eede72ae7
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee3662caa8a6391444feb66a21bb632c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae68819c1feea50ccb00297b1ea5aafe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b17e44dfe76c40495166f5bd80130fa6
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f2be68d90d9b93efc2b3d515bcb7c43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab9809b7d82561045b1be59ae49ca373
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29d4ce653c7c982eb137ab7b371d21b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b150aa50d33d40fc4b15d2f67d8dc4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a513dea8ad8b4dc04216c3a5aed0f664(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d20cde61d051cc2423c26128a694dd9
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb8144f9f771f2a3b23e1870774343d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9edb91a54fd452cb4dc4d927718b484
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f10aab0633bad5b1e3dd61c191356cfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84b860e28cff1e9148ce77c03eea718e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfbd0dae33636a942a34d4156f34e678(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9edb91a54fd452cb4dc4d927718b484
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57138edf104d9dd33dc84dc82213b8f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9efe84b5064927d2cf8da3d8d277e33c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fc93a3dac917cfb5da5b4815d13b98d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7789bd25d1a1bc214a9235b8066f44f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc93a3dac917cfb5da5b4815d13b98d
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8d44da85882e33fad3f40a38c75516a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89a597b58767a4884dfaef890ace6422
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e846fcf9c4f897ea29b8755aed6d733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a9de25a966e424746fa4dc23cc8bde6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b723c9c4aea804fdf74721a95831f5a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f32ac17919eba6b1bcb7603116156ba8
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b38e0cca2fa1135406007196ec03cf1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4de7aa40d52cc2d3bdec2ca48a7f855
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_821c351ee87c7b6a1266443f3eb28944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc93a3dac917cfb5da5b4815d13b98d
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee3662caa8a6391444feb66a21bb632c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3de8a14b096a1fca040c3677ec9fa2b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_130a37196d0d0bf6ba5da2d5cf1ef62f
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28d288a689083c12d471c7fcfd06afc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbbe46404355b8c7abe8b8d59fa7fb6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a906eda75aaf971661df97380520ff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05cdfe5aaa4b4a7d1b5ce8df892aef3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cd154df6bb8d1b7a70104eee34852f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c898cf4b5d11bc16fda71d9f11dad8c0
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f51a2f5184a8588aac96e7ceb5d5ef2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e6fbfb88ed4f412948e8f298553651
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1018f22d10b7c576dc88abb374da97c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b905495881efff550ccaf7dac60cb9b
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb8144f9f771f2a3b23e1870774343d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9edb91a54fd452cb4dc4d927718b484
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77b08b4e5d8642706bcdb4f9d7ecfe81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77b08b4e5d8642706bcdb4f9d7ecfe81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f393274a41e13acb590aae116539a58
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dde2bd2d7119c29704e2e24fd045ca5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_130a37196d0d0bf6ba5da2d5cf1ef62f
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72b64b4cf4f89c2a40295489c167236d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1, arg_2):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = arg_2
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56b46fad56da343ecc69b0f8f7c2fc2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c08b461ae493339aa7bf47cd47500b80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff4de58b9a45f839cfa9ce3567dbb94f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_760881e4cf29fae810ea5ea343dd366e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4f73199a1ef7445e943b1a0c5d14851(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([16, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_546ff02faf61adb06226a7814345690a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fb8dcb34a36bec70ef5b66cc70b420d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_300126b41055c71e93dc6ac8042950b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([128, 32, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7bafcf4d5d2b361f6be9c5466c639ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0baa9215ac922afcdcfc5721be960d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_265fedb0f03d02e6a225e640dde8b3ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de43d7b2e0037e6f138d54de75cf9363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e70d1fda0a8e9ac7d01fc25356798a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_433d2df2959e88fa292461f316ea1648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbedd156ea743e18624e646128bec389(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6a6df28d88d2a41908a9d410ef99fd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0dd8d1dbcdc1edde2a7e6d630ad5c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f38a08631a49ad048bf7739180ce138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9246be490b932f1044911df4dfb89d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([64, 256, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d76812610604f201c5c270877f55da9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fb8dcb34a36bec70ef5b66cc70b420d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ba6e04c65c9e88f49ce50c2d25cf1dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0baa9215ac922afcdcfc5721be960d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b91c0c6fe0c2e13cdb805544add8090(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4c86b43d9c1d5d5fa880178d074f33d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f4e907fea58740c836dc9acd5756f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35a6a170faf168abd57bdf0545e5ceab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_019f0b245e25cc589cf17ccb8ecf71fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([8, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7bafcf4d5d2b361f6be9c5466c639ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7b1b3a55de24c4e55040b0699a2808f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a46eed3c577ca6322310158aaca9a500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_659fbd3cba66f778482c4b620ed57547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e37da4be81668dd543bcb8a16dbda8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4122cf2b7543f16ed06a37e7f4e893dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4122cf2b7543f16ed06a37e7f4e893dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca146a1965ecbf313076a9136f5af586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd0e65cab30bdc92f04f41e6a77fe9f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dfaae3576152f4daae32721888d97f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc2e3aa34643b489c3da298ee25eb13e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98d4cb3c0bdd398551e1357518b2f4bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56b46fad56da343ecc69b0f8f7c2fc2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3660be295e90a28690e5babdddbedcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f4e907fea58740c836dc9acd5756f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb4c4462ae8542d701158b5c9e7a40e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb4c4462ae8542d701158b5c9e7a40e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa8c70af47db20fda2368af6004e309b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f4e907fea58740c836dc9acd5756f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ad86a5ccf094907dcd8ba40f9e21a37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0dd8d1dbcdc1edde2a7e6d630ad5c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea83ec070cea03851f8caa10ac30e489(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02256b8ac243ff2ae7dd905167c3323c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f383a25c02fa951373dbb6a4b52fbbf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b552b53ca3942e6c18dfa514b28bf6ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1af27ded857bf42ffae463c0b75d6397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c8c1fe1a82ca3aab0197ab0af06da5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91496b989a7c9fe6cb68e0d5a27c8c6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72b64b4cf4f89c2a40295489c167236d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c267d8cf654de82c87adf8ed7554d47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32dd60d918820cdcfa0eefe0b4dca246(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f38a08631a49ad048bf7739180ce138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5f3458212b484f38bf9e17c7c15ae9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e4e58ea56f9788b0e9b77a027f3131c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4dda559d376d0f026e68a1e5b7ab5dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca146a1965ecbf313076a9136f5af586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f12378f559e32b0e094acf15387d1e28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b852188c2f9e5b5713c3497c718545a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([8, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d55d7190619531da02b718ac598f5b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e37da4be81668dd543bcb8a16dbda8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60283b9e9edf5b8c36b4cd6aa26dfbc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_265fedb0f03d02e6a225e640dde8b3ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b91c0c6fe0c2e13cdb805544add8090(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cad92b0d711be8c8cc68f1674430e0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57e63f21d52631cd81f3afcae7c7076b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67ba3034b100ee135e0ec102118747ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca146a1965ecbf313076a9136f5af586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c03240132c156217b31f60074953f9ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1e3733e2f2304b8d8c89775980a1025(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb4c4462ae8542d701158b5c9e7a40e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de43d7b2e0037e6f138d54de75cf9363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4b3847317346ce6c7a5700f37473bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34a25407d7020ad88ab48c71ccd3d76c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46440533a7fda9b11c45afc30a7cead5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84cdafed12a9b182cf24959e1f7b9984(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d55d7190619531da02b718ac598f5b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0dd8d1dbcdc1edde2a7e6d630ad5c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84cdafed12a9b182cf24959e1f7b9984(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0939a8cf0e35dc30e4a5e39ff990b7df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7bafcf4d5d2b361f6be9c5466c639ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60283b9e9edf5b8c36b4cd6aa26dfbc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0dd8d1dbcdc1edde2a7e6d630ad5c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_466bef8a42b7027c7c7a9f2dcd3bed0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7bafcf4d5d2b361f6be9c5466c639ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dfaae3576152f4daae32721888d97f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca146a1965ecbf313076a9136f5af586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b70089b730b471146a999a3fd76db0d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d07d6fe3e08d9b52be6f28c21b14e8d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f12378f559e32b0e094acf15387d1e28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c392f1520f999448acd031e972fe3b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_546ff02faf61adb06226a7814345690a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88b926175eadf507ea42c8face80d8b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d76812610604f201c5c270877f55da9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5dcb98b439f6a7fe4ac938aa8b567fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff4de58b9a45f839cfa9ce3567dbb94f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d55d7190619531da02b718ac598f5b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98d4cb3c0bdd398551e1357518b2f4bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d76812610604f201c5c270877f55da9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f4e907fea58740c836dc9acd5756f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ddbc6cb37e4371191ee3765a1b5e26f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35a6a170faf168abd57bdf0545e5ceab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca01dd70199dea9b012a8d7c06021daf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_433d2df2959e88fa292461f316ea1648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cad92b0d711be8c8cc68f1674430e0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b70089b730b471146a999a3fd76db0d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ba79727637e40044499017f850576d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_738ccad899212bf1bc2d5838df5f3de9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1af27ded857bf42ffae463c0b75d6397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f536ff51f015d2a1e422e39301bd3bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34a25407d7020ad88ab48c71ccd3d76c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bf6a2b3e6b46477b93fa40e3e1ec3f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d55d7190619531da02b718ac598f5b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ede3d2d2052389b41badc7338ec2c8dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c99821ae56f51fae1fa4dfafc4b95bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02256b8ac243ff2ae7dd905167c3323c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17469aed3693f3d491095e078d85f1f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8beb985b6aaa883002276e2b5c73850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_760881e4cf29fae810ea5ea343dd366e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca01dd70199dea9b012a8d7c06021daf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60283b9e9edf5b8c36b4cd6aa26dfbc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60283b9e9edf5b8c36b4cd6aa26dfbc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6ac8cb716e3299e21675483df1bbb1e
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08f059f8bb0956040fbcb308d8e1467f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9ff65c8d80af706a48c1b861e8a7654
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()