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
    class PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0868a68cd0ac3074c935e697ce6005b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 91, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de4cd2222df07ea0180ce78f1bedbf12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b300e7afe771dd0fff85588d4398f2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 784, 6, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ce24587b314453406b807e2d466c291(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b300e7afe771dd0fff85588d4398f2a
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_664326ea73be78ceae355590da0dd2c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8ef886adf43cd44b0659c800029e0cb5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 192, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff82ee39581ed1311024b8276fe86cfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef886adf43cd44b0659c800029e0cb5
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_36dca69bb6baec8bc7ac96325010cf6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 2, 6, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08042e139f87f09cbfdc8cf9cac95548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36dca69bb6baec8bc7ac96325010cf6b
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6295bd17f2f1f862fa6bb214dfbcffac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 6, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc03fcdf22fcfbfc1739254f1d833b80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6295bd17f2f1f862fa6bb214dfbcffac
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 3, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1ce1d7e3bebc552f081f469fd58d89a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 168, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_392afd7de94d490b1912cf5ca46a7e15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75a7ee3c65ed4046a38e4983ad477d91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_896a842ea2842dd59572bf08c9c8f4fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbe516d61324dc99d34ecf0485409552(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3475e3a9f3b0e03dcd4c4cff3545fde9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 168, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39fc8d86bd746f820ae5183aba0dff46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 84, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24793c99e90bed8c228761defcc7e448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 42, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa1879c46fc1ceabd2c3a15ef4953dc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 21, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b946a689d3d713a431f6b85366a798d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_851dd8d68402e39c7d99d0427ee34f67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 256, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2dac23aa74736b9dc0e41d5f607278ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_851dd8d68402e39c7d99d0427ee34f67
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_134007ac0b4b40ffe9be2c35f77eb4d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 8, 7, 8, 7, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b46691b7998bc7f907679a785736ad1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_134007ac0b4b40ffe9be2c35f77eb4d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98745fcdd24bc0f23605e28777f0a02a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 64, 49, 3, 3, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55edb4f2f55f291f124d5d726f20f46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98745fcdd24bc0f23605e28777f0a02a
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_432c4659e272e917c6114753489a3b0a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 64, 3, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2fbf8b610cea4ec488f9cff7933a300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_432c4659e272e917c6114753489a3b0a
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c367e65b02327aeb34a63577c8ebd3f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 3, 4, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf18bf323c14d0c143e70a60f9a17833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c367e65b02327aeb34a63577c8ebd3f
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_062fd9c719698e68a30b7dff9f1f66d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e112b62995069639771ff875b61ae592(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_062fd9c719698e68a30b7dff9f1f66d0
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dce1a972cfa7b6c9ef936b3dedf72038(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67a01ee01c28ce8d4ce1d9309880aa34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dce1a972cfa7b6c9ef936b3dedf72038
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_99e37bcebca30a551d56bbcfd4e8b644(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 198, 3, 3, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_578a32ffe8ec092e26b31fe6e70e3696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e37bcebca30a551d56bbcfd4e8b644
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c511d70ca6dcc15f6cd8ddf8705f8b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24c86ffc6e5f08ea78ff90e658d376ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c511d70ca6dcc15f6cd8ddf8705f8b6
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1cd980a87d6f29be401ec45588b35624(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_927919abfc0f5509f313a07c673cf339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cd980a87d6f29be401ec45588b35624
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_639ff6d4c1f55996735c0c3c887bb4fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1960, 16, 2, 4, 6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e1668ffcd01aaa48105848266464230(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_639ff6d4c1f55996735c0c3c887bb4fc
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b671a18a5c91c31f0710e879af95c032(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1960, 16, 4, 6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d5a2d470ecd43bc9f5e6457b0f6d6a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b671a18a5c91c31f0710e879af95c032
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 4, 6], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_83de2a4ee8f7831b902633ab33bfc26d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1960, 4, 16, 6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8429d3aa6df5001d2ec2e5debd2e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83de2a4ee8f7831b902633ab33bfc26d
        def get_inputs(self):
            return [
                paddle.uniform([1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3fa368e2fb4cf39f3cba1bbd8b5f743d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 784, 6, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c428f29e5c8800441b4679b4bf92c4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fa368e2fb4cf39f3cba1bbd8b5f743d
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4950a2033c869b6ef473e7ac34ecf2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dfb8470becfc2912eb208ef3f9ad9bc5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 192, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f79c2c2471aaa110fa38fa06f54c8912(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfb8470becfc2912eb208ef3f9ad9bc5
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2efc28d303926d154df6bde5aaba7d9c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 2, 6, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5c5a89e383099304839f806bbb71485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2efc28d303926d154df6bde5aaba7d9c
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_189f73412e2897689bdb8d16c93806d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 6, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1fe6c81ae4837092a8e4890253af7448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_189f73412e2897689bdb8d16c93806d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be57b71420c4b85fa13e5fb9769bac7d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ab4d84f440118c2b4499c5fcc98aaa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be57b71420c4b85fa13e5fb9769bac7d
        def get_inputs(self):
            return [
                paddle.uniform([16, 32, 128, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e08a760b38bea9ed39d58bcfd388692c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40e138dde135c06464d8b233202fe145(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e08a760b38bea9ed39d58bcfd388692c
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c03cede99185f7d22033c089602bae7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 7056], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 68, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e2767897e6b80452a7b5f70fafa1ae5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 7056], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a59e95d62605f02037245de6b35cd8ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 8, 7, 8, 7, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_420acb663cfb0b2607ffe5e525bd428e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a59e95d62605f02037245de6b35cd8ad
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1bfafa881f4341cb8cb170f6029ec437(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 64, 49, 3, 3, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c90643598dc91e9b239459046987ac2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bfafa881f4341cb8cb170f6029ec437
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe0337e8f6be410432a7e21bb7c6bac0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 64, 3, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da0d63df44036f56ddd4c87816b8d49f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe0337e8f6be410432a7e21bb7c6bac0
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0595143d39a313d317200341f229c286(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 4, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c126058036cd3accb09179310e06b2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0595143d39a313d317200341f229c286
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec2b82be0322aeea623f96f3bc852262(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c10fdaf44a75392a287258dda3c78aea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0945d30a988d3f958eb336bc4205017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63d03c57ac6704700c29b5dc48f5a982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e10c263d31e9006123f7c727fdee7161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_114eb5c8104b6e0340d9158b8f03fe12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8718448b220b1f4c48dbc86e6ab3649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46278c09246dcc73144185d3f44f943d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab67b8d82312044aa13402dda9725b8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ceec03e8fda1d1c2f0a657553f115bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed3fda0b608a55eeffed89c6b27f9726(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f81b0c43b60818caedc57e5d419d321e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c235557da1baf5b1560f28f6e062dd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c31ea4e41dcec0f675eff66a5b2255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d73ac1446d2fab41e10c65400c7be6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33b385c519350d83efd3e0648705f856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_788b5b4b234e8afe6d801e8cb5ba8c81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_574ea32336baf2655a271aee5b8497b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 225], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_247e593723ba6e8a1b2d6a357a964734(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 225], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_946d06ba9961149e2c5ca46fa0750be5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 256, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b642405f7b7d312e2f0e1d7b8ffccd31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_946d06ba9961149e2c5ca46fa0750be5
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e7399885c688bbdd0c29bbbd02f57a53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 20, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_58b961c1957d447b28f7e244dcf6f7e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7399885c688bbdd0c29bbbd02f57a53
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b9940e77d7fa08cf03c2135114384fa5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 40, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc433c7e27836f15a71eee5d532a0cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9940e77d7fa08cf03c2135114384fa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d50053d40f51c7fe0d7687ea0e19ef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 152, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33deb4d18984899a2a2bd5c5779646a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13ebcabdf69eaf57a2d2124dc655ad09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6ffbf9437342d7ee023a8506a0bbc6c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 196, 12, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f33dbfc01c1b58f66776c1326aa18ab7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ffbf9437342d7ee023a8506a0bbc6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48582d7f9636bc1813000808f66d9bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f1c254262e2eb9012276e827058ad72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 384, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b34c02e8a23cb378c1aadc48f8613a7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f1c254262e2eb9012276e827058ad72
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3666fbd96f0f6178a27f37a1fdd1ed4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 2, 12, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8576614c0f55ba4f7145fa3256838cbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3666fbd96f0f6178a27f37a1fdd1ed4
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2652c1e9626e535a8eb78611b3f75b7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 12, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_922f809e72012acfd65e181ee6b6c728(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2652c1e9626e535a8eb78611b3f75b7c
        def get_inputs(self):
            return [
                paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f1771564966d83107a861b1215e800fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_16f249f84c79b4cd51e07576f7faa40c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1771564966d83107a861b1215e800fd
        def get_inputs(self):
            return [
                paddle.uniform([128, 16, 8, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0ec192f6737a747c0d9497fe034ebf8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0a39484427c2d16ff27a2804c85e0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ec192f6737a747c0d9497fe034ebf8f
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475bcd04fb9af52baa73a6f559185088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ae3209ca54bbaa6c577d9025cfde049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf97d1375c4a23dba07836821d9af42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6273796a71bd4ee4fd80effbf27088c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 76, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb190a4622f81b26fd6b0c35ed226e65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6273796a71bd4ee4fd80effbf27088c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 676], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6ec84711734f3ac9a5d66433f784d495(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8e636973403ba964864e503c0ade1745(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ec84711734f3ac9a5d66433f784d495
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8887bb5e97c8da5daf4db4a2a5dd6b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 120, 216], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0611cd5653f5c27fc0dcc779ae2beb84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 4, 5, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, 8, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_563fa57a8f5afd34d030603b4142ad92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0611cd5653f5c27fc0dcc779ae2beb84
        def get_inputs(self):
            return [
                paddle.uniform([4, 8, 8, 128, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4d4973f7da5a54dcabcb68bf6fb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 900], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f27fa0852dce373c9585d2751663bce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 900], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06e6750dbc9b761b47ad990270ba92e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a327f9e4cb2c5b3d2f1c8c6b365611f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cade68cb1ef28de7c4361b50268e27aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 3136, 3, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5537dffbfbece2c8deec146435dd24dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cade68cb1ef28de7c4361b50268e27aa
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbe4f7783d832e3c5986d79e39aa3857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5704a8b88535adf763e8a7b980b87c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09d071e77c7a00a93ef8c5faab34db87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5704a8b88535adf763e8a7b980b87c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8290dda6720668f7da9f1c2860e4f6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 2, 3, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d49729b0f19bbde1e018e1fc5c8e4f00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8290dda6720668f7da9f1c2860e4f6f
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_25f4570e6a06a0767522d81396171560(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 3, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54994a9f48f2e473eec6fc6100445e38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_25f4570e6a06a0767522d81396171560
        def get_inputs(self):
            return [
                paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b803a4abdc784aa7744dacca9d6bab4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c367e65b02327aeb34a63577c8ebd3f
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02a83931fd60d3c6f83d45807d61671f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_062fd9c719698e68a30b7dff9f1f66d0
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66f8cdb4360a7fce0d7b12e747edd539(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dce1a972cfa7b6c9ef936b3dedf72038
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_047df291e8f273aa8b13600b19a42e9c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 15, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80b65a8f2c5d1ebae52cba44fa7ff5e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_047df291e8f273aa8b13600b19a42e9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4d34fe4a588b3bebc2cd7f6303b2bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49a36435ad7e5e96f463c2dc0585df7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6273796a71bd4ee4fd80effbf27088c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 169], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7c638f63aa1da0ffa16a7a202a958c13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_418f9b5fc7b09152902208c4716487b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c638f63aa1da0ffa16a7a202a958c13
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_490f0470ab1d9cd0f86021905c25d600(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 19, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8624c1e42db457557185fc1c803d06cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_490f0470ab1d9cd0f86021905c25d600
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0bbdb98c086adf63912707b3f789dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57f54a63ee668ae6289b4c5245c2710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9740085dfcc65c6f3797367471116a8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0e3af87b82a4ad418cb482e76a6e556(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8f14286b85e4b14637b547c24902a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1156], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ecd821db397f12c941a15ff022374077(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3690da291200edbc447e6cec4bfa227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8792ff653fd56a19d5cfdf896987bafc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 24, 48, 2, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_730dcd57fcc9d1c3b9572517410fc4e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8792ff653fd56a19d5cfdf896987bafc
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fc550174bdc7418304cf55c035db74f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 5776], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6370b2bea0da96264a56a2401a8889f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 5776], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe535e7eee578c19a040132351472630(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 4, 7, 4, 7, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8aec75da2717427a263e8c7b1f47ba24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe535e7eee578c19a040132351472630
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ce6d418dbcc515306858260ab40cc94(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 16, 49, 3, 6, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b9cb305ed002c7531e26f09f96b7ec1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ce6d418dbcc515306858260ab40cc94
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f37942e61d2574ac641badb08253d2f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 16, 6, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20d49b6527ee5918d8dee15402b82247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f37942e61d2574ac641badb08253d2f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abc8718f0d2f08a181bc53b9aec33888(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c638f63aa1da0ffa16a7a202a958c13
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 16384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d607b6aead53f3c5b1c867d0bb06932a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 21, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed944dc62ab77c0bfc78365aab134dfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d607b6aead53f3c5b1c867d0bb06932a
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c31ea4e41dcec0f675eff66a5b2255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_424897228944f97e370c62a02d1ae457(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_428d276a4ce7ed0ba9a3b7e1ee616f52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8268f5f12efd185c7a4f11802e269011(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 1, 12, 24, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e85c1f9b3205c0051c3012bad30f516(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8268f5f12efd185c7a4f11802e269011
        def get_inputs(self):
            return [
                paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b9c0db1d5a850caec00628b7eb67209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1771564966d83107a861b1215e800fd
        def get_inputs(self):
            return [
                paddle.uniform([8, 16, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_955fdafdd90accb7b9d08798d050a265(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ec192f6737a747c0d9497fe034ebf8f
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f87c5935d0b3a288a9c63642b6ddfc6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7a2a4656a35cd3998a1ca23264f8d5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be57b71420c4b85fa13e5fb9769bac7d
        def get_inputs(self):
            return [
                paddle.uniform([8, 16, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a86c0a31170c06e4558a6fba78930554(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4030aefb9e5a73c6c6f5676d5581d927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a86c0a31170c06e4558a6fba78930554
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ff1cce43ec99a84958a525ac094f098(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5e4595c729963cc67e7e38658803396(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 3, 12, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9a2401b2846ec95c183e9eb685e0ae2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5e4595c729963cc67e7e38658803396
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 3, 12, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e363412347a60abab7ad07040d437303(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d288eec7446269f4c9a6a0f78300ed2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e363412347a60abab7ad07040d437303
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_109532bfc0f354b1a5f8393154f672e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_164f091f3e4051d9f77581324d6f5f3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_109532bfc0f354b1a5f8393154f672e2
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_840e5de81925216549526620520be216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1296], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98d67c89f75b153078c1a725d66dcd26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1296], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d403e454876fb7d5cdad9cee1e750561(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1296], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_844787bffdd8157a814b863514416014(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc42323883c5ad8ea14ab1b1ca9219c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_844787bffdd8157a814b863514416014
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_84708b69cfa914b04bb4ff3a22a49942(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ca962336070caf5166adcca0dc2a80e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84708b69cfa914b04bb4ff3a22a49942
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fabef0f4d710a0f15b48c4a2481c01c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 197, 2, 6, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d807c09a115c070a145fdce3ea22a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fabef0f4d710a0f15b48c4a2481c01c
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2016f8c90138c08e1015e0eafa49d9c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 197, 6, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_921b86cb66b67782116bcbb24fbe4f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2016f8c90138c08e1015e0eafa49d9c1
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b1d4c24e57e2d9c67e41c9f35c70b710(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 6, 197, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e74c9cbb4f3f040931235d20fad242e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1d4c24e57e2d9c67e41c9f35c70b710
        def get_inputs(self):
            return [
                paddle.uniform([10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8de0b5896b1c997563b4987a8293131e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_047df291e8f273aa8b13600b19a42e9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1cc452e8f60ea078d38e5c5449180ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0863b06275110cbcdd8c8855b93c88ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 96, 96, 1, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9f20356d851bf637f9f1841243bae28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0863b06275110cbcdd8c8855b93c88ee
        def get_inputs(self):
            return [
                paddle.uniform([4, 1, 96, 96, 1, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eb583cbd79393cd09d052060e5c946a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 4, 7, 4, 7, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b3c3bfc3eb2fc46492da12a9e7093d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb583cbd79393cd09d052060e5c946a5
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8b8437af92decbaec0e9f33e04ebb827(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 16, 49, 3, 6, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_528dc9e22f91d4b490df299c280aea75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b8437af92decbaec0e9f33e04ebb827
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9dccb62b585eb9a09ee70125ea12b51c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 16, 6, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff33aea4ef5d15db2ec4e8c9456f0e34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9dccb62b585eb9a09ee70125ea12b51c
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_801d4d89f710509b3f1cdfd800e43ad6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 2, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4623ce81ba239fdc442ef95376040f76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_801d4d89f710509b3f1cdfd800e43ad6
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 2, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6d3153c786e372ab8466f533947b420(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c760f37d8b8309e94ab84d3e25e4c64c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6d3153c786e372ab8466f533947b420
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_915a2d52bd49338934168550d95ac8e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f1481f49f74da06ae12d6d1a87684c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915a2d52bd49338934168550d95ac8e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_540f0c9db3b78134acb42074df3155c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 2, 2, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2d53e5d725942e797a834b1ab4b1869(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_540f0c9db3b78134acb42074df3155c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 2, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b01fa7fd5ede57bc99df39fe5b24be99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7019ad909511c8d879c548b3f6edbe8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b01fa7fd5ede57bc99df39fe5b24be99
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a6d24b809d5cccfefd462f43d2dacff9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3e806933710322660951e12e8a0c35b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6d24b809d5cccfefd462f43d2dacff9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2bb92477d6f6a08afa0f4e49d04d162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_047df291e8f273aa8b13600b19a42e9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5537dffbfbece2c8deec146435dd24dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cade68cb1ef28de7c4361b50268e27aa
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbe4f7783d832e3c5986d79e39aa3857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09d071e77c7a00a93ef8c5faab34db87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5704a8b88535adf763e8a7b980b87c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d49729b0f19bbde1e018e1fc5c8e4f00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8290dda6720668f7da9f1c2860e4f6f
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54994a9f48f2e473eec6fc6100445e38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_25f4570e6a06a0767522d81396171560
        def get_inputs(self):
            return [
                paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3dd91925169d051e5bcfb5bea44bb31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be57b71420c4b85fa13e5fb9769bac7d
        def get_inputs(self):
            return [
                paddle.uniform([16, 32, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60c13fbbc34fb9656344b4abc27378f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e08a760b38bea9ed39d58bcfd388692c
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4d34fe4a588b3bebc2cd7f6303b2bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de3bf14822669b64d3dbd6f6eec6043(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_771c44f9dfd7ef70b13141d8ce5d04d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 24, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_990295f3fc6a067a6c06d9021a9c04f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_771c44f9dfd7ef70b13141d8ce5d04d0
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19703cd3f09143b3c0a437ebd77ef7ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 2, 24, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5a42369f982101cf528d5038be90ff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19703cd3f09143b3c0a437ebd77ef7ad
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd4c08243a55dc78a22358f07b697164(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 24, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d2d37e00efb8f341d4ee33206faf384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4c08243a55dc78a22358f07b697164
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f8eccd4c2403fd72bdb8799f8df2ca4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_257c1d49d02dfed02e069b536319591e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_864dcf7b7ccdf907be7e2df2bc4985d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff6bc93e35961e54a41f81ea52411982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3f2bf36d87837df3b630fbf6314cda9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be57b71420c4b85fa13e5fb9769bac7d
        def get_inputs(self):
            return [
                paddle.uniform([8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a8320b99fb43a984c6e4012fab487ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a86c0a31170c06e4558a6fba78930554
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc7a6311d52c24acb0fb0e62f3858cb9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22c906ed446446b5bacfd056970ffc80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7a6311d52c24acb0fb0e62f3858cb9
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cafbd55fec581e8a89f52f74d8d3c5a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ce7086acf02e3e9390b4958d25ff43f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8c4dd0dea949e41463caa2e7db59b9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ce7086acf02e3e9390b4958d25ff43f
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_24f259006e30a07a2a8765caa0bcac1a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a5501ca03312aceed7e3d4880af1ab7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24f259006e30a07a2a8765caa0bcac1a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d5705b3db3a030bb5819ea08b65df22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84ab273389bf9dfaa0d9b3bd0658156c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d5705b3db3a030bb5819ea08b65df22
        def get_inputs(self):
            return [
                paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7eafa921f8c0b573c851ec3648281807(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 60800], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c604ce1bb6c098b84d9ecf238eff94e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7eafa921f8c0b573c851ec3648281807
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23d4ded88eb2357e63e5ffd5cc1fe7d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 3, 2, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e80ecd1f30c574cdbcbb493016ec8830(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d4ded88eb2357e63e5ffd5cc1fe7d3
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 3, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f67962df975b9c256c2fb8f58baf4ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c068d0a1cf43783addde3c5f8887a55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f67962df975b9c256c2fb8f58baf4ed
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15cc6b18714f96f66112b2facd7b03ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_313a7f9b7803f81a111675026f378e8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15cc6b18714f96f66112b2facd7b03ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_203598c93addf4624033be691238501d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 5, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56c1815f60a8ea30f78b0b6d36739bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_203598c93addf4624033be691238501d
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b527f13fda65c40fe6def0714912336(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e37bcebca30a551d56bbcfd4e8b644
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08903d6b423905f8f736b5ff6a78f823(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c511d70ca6dcc15f6cd8ddf8705f8b6
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4d2b534401192979b0b2b4943b2e0d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cd980a87d6f29be401ec45588b35624
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ade30624e4d96243ec6569be0a92856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4b29d239249802735bdfd1798db9003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_271f22cf21bc4e4d4938780943d50e8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 3136, 3, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d60d89b1d47796c9e3c836e54fef145(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_271f22cf21bc4e4d4938780943d50e8a
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ab59f8f6ce4247a3e744ecf132045d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c2bd64a3246c680977b397a1d73ca601(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d274bd34e3181a560f4c82e9fd296091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2bd64a3246c680977b397a1d73ca601
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d7519500da3b2d605c9793b320d45ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 2, 3, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccd735b44799bfa4cfb8344734e89a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d7519500da3b2d605c9793b320d45ac
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3c89d96c28e3154f52e2335c187f0838(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 3, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d0a4dd10033a20bf602a110a40b0feb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c89d96c28e3154f52e2335c187f0838
        def get_inputs(self):
            return [
                paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c964419aa7b9a001ed8beee4910d1064(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fc46fddaaa142cfda605b3bb653040e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c964419aa7b9a001ed8beee4910d1064
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de96c842679f316b7ef1324bfc6ea0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a40dcf1f13e525400e75c3718ae8531(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 2, 24, 12, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1449fa5727769f2d68879a97b296a258(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a40dcf1f13e525400e75c3718ae8531
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_702a6591e53c62d363035511a3db54a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_576d1fd19f124aeccf27417af6e97305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_411551885ba39ef4413237552f27c80d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1, 7, 1, 7, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61c58844a02e9e2f970038740bb95ffc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_411551885ba39ef4413237552f27c80d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ebfe4951d147abec7054270921106da4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1, 49, 3, 24, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c512a6a9a3bd0e9b3f473d1f03ce245d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebfe4951d147abec7054270921106da4
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d04760efbcd9493e308cadff249a5c9c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1, 24, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e9b3f236c26dad76c8f78ee2be5f032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d04760efbcd9493e308cadff249a5c9c
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86ac92e03106841e9e4bdd776a78efa8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4312, 16, 2, 4, 6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ed146dc49278c9d06cc4b40edbd404e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86ac92e03106841e9e4bdd776a78efa8
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_109ce41a7912d41048087363a4f43868(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4312, 16, 4, 6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eb7bab6fb2dfd284d7405b0c72baddad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_109ce41a7912d41048087363a4f43868
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 4, 6], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a8271d8cb9e23d819da9f7c9def64f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4312, 4, 16, 6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41b3e74f65062d1caf0d21f834e7a314(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a8271d8cb9e23d819da9f7c9def64f8
        def get_inputs(self):
            return [
                paddle.uniform([4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33b385c519350d83efd3e0648705f856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_788b5b4b234e8afe6d801e8cb5ba8c81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_561fa710f2f3e5ac2d9ffe9efbd447fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 441], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12904267eb4fd9745116959e50348216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 441], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_420acb663cfb0b2607ffe5e525bd428e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a59e95d62605f02037245de6b35cd8ad
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c90643598dc91e9b239459046987ac2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bfafa881f4341cb8cb170f6029ec437
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da0d63df44036f56ddd4c87816b8d49f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe0337e8f6be410432a7e21bb7c6bac0
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0e3af87b82a4ad418cb482e76a6e556(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_592c2af1dd665e7b82fecd7e8e8e3830(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_973176490dfd08e5faf08a5de976c08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1156], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_927880bdfaa41731e4bebfb017bce0d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 1280], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c0a062993b20fd04bdc9ad5a6c7ef33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_927880bdfaa41731e4bebfb017bce0d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 1280], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_22660f49b7d4b2e7e36e2b94755c032c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1280, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca0e446fe4f1ad1d673be36d5ff07e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22660f49b7d4b2e7e36e2b94755c032c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcf513b26f182befc0f6b89116c9482b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5dd57ac345a297a772afeaad86e2a59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e26d9bd8805f31c3ddda52f71a3b8e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33fca6910bab54f261b06eed17d78e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbe516d61324dc99d34ecf0485409552(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56a6d6a48e9504b5dae65fa85b1f0ce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82a0cf1c7f5036d6dec2b8bce490c2c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c09e172beee0e44b71c4ace533478105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5c27a67301bf6f35b5ca98ac5ef52d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b946a689d3d713a431f6b85366a798d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_542a0f5d490f6739119409be6662dcfb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf1e47be3f45c28254abd6738a7c1ef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_542a0f5d490f6739119409be6662dcfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 65536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_676de93280bc2840afecebf4b93f0b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_142fd9c0d1cd694e7bb80fd0d881e95b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 1, 1, 96, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94fe85f82c7778ebb21ed85e16877b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_142fd9c0d1cd694e7bb80fd0d881e95b
        def get_inputs(self):
            return [
                paddle.uniform([6, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88aee3da86aae97f4a58f453aa38bbb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 324], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_baaec74da1a8ac0fb603feab3d439791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 324], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ac544fcb1fa35297e39c0dbd4fd8e8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 324], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebb4ee664f4ed8ed839e767913b894e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ec84711734f3ac9a5d66433f784d495
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93ec05d4b9bc72b811ce4cbbcdb89a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_172546e0a1615f561386acbbe1eda2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 289], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_568c0a6a2be823935db6ef8d206d9691(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 289], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f30351c67ab75db136dcaa684f39482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24f259006e30a07a2a8765caa0bcac1a
        def get_inputs(self):
            return [
                paddle.uniform([6, 96, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_499abf921ff037f1d869fe3605856eab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_78de80e32c80e8fecfaac58860b0568a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 12, 12, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60d1df5b85efe5eee27c6a65cf558e52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78de80e32c80e8fecfaac58860b0568a
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7acb6fd7bce6303b674ee0275d122111(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecdbd89a4f795ca2aeff2f4e7ef6b054(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8792ff653fd56a19d5cfdf896987bafc
        def get_inputs(self):
            return [
                paddle.uniform([4, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12c43878f79552cd08650f155bd915df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e85c1f9b3205c0051c3012bad30f516(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8268f5f12efd185c7a4f11802e269011
        def get_inputs(self):
            return [
                paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_efd6e6798aecc93aea16b27c72f61af8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_93b67b22075e3ec321526fb35e51cdce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_efd6e6798aecc93aea16b27c72f61af8
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_33035f6b7a66bcfc24f832181bc1d508(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da714a7eabfaf23b7405469620638cdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33035f6b7a66bcfc24f832181bc1d508
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0850d3fccf64417751bc26c3c7bf4e9b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b11cdff3565b8a6f84407d308fc968cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0850d3fccf64417751bc26c3c7bf4e9b
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a60a1eaabd7e591aeca45ab9c46d9f55(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0471806c67cfa1ddfa5d7baaae84e4d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a60a1eaabd7e591aeca45ab9c46d9f55
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c680aca0f495dba78e0efb03f0a5ab2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_353574e733805f56462f66a99d4c0df4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14d994308eeb588951f8900f2d77bc69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9e5ab55aa7d4b4e95a5a5d1dfc184e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78de80e32c80e8fecfaac58860b0568a
        def get_inputs(self):
            return [
                paddle.uniform([6, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475bcd04fb9af52baa73a6f559185088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ae3209ca54bbaa6c577d9025cfde049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_885a1b8711587a1989966975fbb2198c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_576d1fd19f124aeccf27417af6e97305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f6b84477dbe7ceb7eadeccb81d02ab9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 196, 12, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1546e977c89818f65e56f1343841c7f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f6b84477dbe7ceb7eadeccb81d02ab9
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7772d840062dfcb0baeb9cecada0165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7838c7e55f6a099c91084528b6dd7a37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 384, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4cbb482ca01da568ac8fdd8b6695d50b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7838c7e55f6a099c91084528b6dd7a37
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc460daf7b6852ab898531efcefdf840(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 2, 12, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f42f86c8edfc4ae2d01ee8f997650a81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc460daf7b6852ab898531efcefdf840
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_61b492a74c8b8e95992a37ac4a5bbc97(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 12, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76790b6960c38393b4a03dc63a5a75d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61b492a74c8b8e95992a37ac4a5bbc97
        def get_inputs(self):
            return [
                paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_272926c2c9b07d9be34ab3b82afe5d04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d5fe8fbd1d77a023da74d17bfb9e985(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_142fd9c0d1cd694e7bb80fd0d881e95b
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf18bf323c14d0c143e70a60f9a17833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c367e65b02327aeb34a63577c8ebd3f
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e112b62995069639771ff875b61ae592(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_062fd9c719698e68a30b7dff9f1f66d0
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67a01ee01c28ce8d4ce1d9309880aa34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dce1a972cfa7b6c9ef936b3dedf72038
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b55bffdf74ef82ab402bd59741a043ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 3, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_015d725b9870c011aeda650b4ab06434(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b55bffdf74ef82ab402bd59741a043ef
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fe02ffbae0448ff245a6e30f5937b32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 3, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 84, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54ac9e9b4fc0379bf9388a84370aaffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fe02ffbae0448ff245a6e30f5937b32
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_043e9c81faffd266c10ec320959ce15b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 3, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f6391b50d26fffe415a9989501ccc51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_043e9c81faffd266c10ec320959ce15b
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_acf8a96f06b76f8722d0a482a1982092(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 3, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 126, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6f6cc8cabc4cac3c8986489e85fcd834(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acf8a96f06b76f8722d0a482a1982092
        def get_inputs(self):
            return [
                paddle.uniform([1, 126, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e595f80a6a913f87d3dbf8780bd7cf7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_043e9c81faffd266c10ec320959ce15b
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a755ec862d0720cdab6fafff17b84e42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acf8a96f06b76f8722d0a482a1982092
        def get_inputs(self):
            return [
                paddle.uniform([1, 126, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_861a16addf3825fc281dd1ac564ca086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_043e9c81faffd266c10ec320959ce15b
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dc5af8af89c0ac10721704e6552f986(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acf8a96f06b76f8722d0a482a1982092
        def get_inputs(self):
            return [
                paddle.uniform([1, 126, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8339dfa91f9eff50d43caaa6e56cc7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b55bffdf74ef82ab402bd59741a043ef
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee9bee474b8a2f55f2f5445b4f99db97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fe02ffbae0448ff245a6e30f5937b32
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2237d90ccdb64c6398895857162fb10a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b55bffdf74ef82ab402bd59741a043ef
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[16.121261596679688]], [[18.27570152282715]], [[16.624515533447266]], [[17.452688217163086]], [[17.60372543334961]], [[18.380887985229492]], [[17.626935958862305]], [[16.684293746948242]], [[18.099266052246094]], [[17.77437400817871]], [[17.67827606201172]], [[16.585798263549805]], [[17.15077018737793]], [[17.22074317932129]], [[15.508727073669434]], [[17.932458877563477]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_8c4e57990fa84fc720e555e77347df19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fe02ffbae0448ff245a6e30f5937b32
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d23d5cfffe8c985ea837bc7828e0738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5f99ccec5fdf5c711f487a4834b64e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_142fd9c0d1cd694e7bb80fd0d881e95b
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7620dcb7aa3cf6d0c17e0d2fdb5c1ae2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 36, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e4ce67722a7b89d37546cfde7ef9713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7620dcb7aa3cf6d0c17e0d2fdb5c1ae2
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 36, 28, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0386b0666c515aa186852f2d7ff4052(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93ec05d4b9bc72b811ce4cbbcdb89a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c023fd65ba6b158975e06240fac7873(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 289], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5169b08426d34942072794a7c0e564d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d3bdb7bafd129f457fb825a1fd2e5a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5169b08426d34942072794a7c0e564d6
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d3bdb7bafd129f457fb825a1fd2e5a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5169b08426d34942072794a7c0e564d6
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_99625f004ccfc1232ecaa52737316f03(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 49, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e8406588545b657e12462d9addd9dd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99625f004ccfc1232ecaa52737316f03
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b760f1109f1d0d39398cc8e1947a174c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 8, 49, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7eaab23d64b723d5a30b68920e3ea620(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b760f1109f1d0d39398cc8e1947a174c
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9bd0661a12b2807d324972e17006593(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_188313fa257412f4fe8941dfd795718e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9bd0661a12b2807d324972e17006593
        def get_inputs(self):
            return [
                paddle.uniform([8, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b46d8d177d5f181219b2f1d88b314d10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2401, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a04d20d790e5efcf0fbfd05df9daf64a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b46d8d177d5f181219b2f1d88b314d10
        def get_inputs(self):
            return [
                paddle.uniform([2401, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4950a2033c869b6ef473e7ac34ecf2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_54af4e1d18abedd62427a4de6bae4631(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 192, 784], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_897138345e1af8c6aabdee842be8f368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54af4e1d18abedd62427a4de6bae4631
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48582d7f9636bc1813000808f66d9bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f8d1fe2794c13b213d744b47e76eafb6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 384, 196], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9fcf1a35f433b4101f7393f20eaf918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8d1fe2794c13b213d744b47e76eafb6
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aff551abf45aef0f2bcc9b28aa043a77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8dbf01453852af8f5870f8405da7523a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff551abf45aef0f2bcc9b28aa043a77
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0bea335ccbe22114637cc2db9079d55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_14e646df43b2bc9418c13ebd29072a10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 5, 1, 2, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None, 8, None, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28fe151a85218445b2cc3c0f777c90f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14e646df43b2bc9418c13ebd29072a10
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4, 8, 16, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b3c3bfc3eb2fc46492da12a9e7093d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb583cbd79393cd09d052060e5c946a5
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_528dc9e22f91d4b490df299c280aea75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b8437af92decbaec0e9f33e04ebb827
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff33aea4ef5d15db2ec4e8c9456f0e34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9dccb62b585eb9a09ee70125ea12b51c
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0726032a54353d145f752be2987fdf27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14e646df43b2bc9418c13ebd29072a10
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 52, 8, 202, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92c36308895c0e975f4d288c73501e71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2e8b101b26f480e9a73ee2ad76c6c9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dffd92a98c1d8bdc197d71faea63a958(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3b98283841bc48e63e29b0801064de3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59938c6d67dde93dc09bc08d60797ae4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60e20b6b312b8183dbe31469f17e705e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f01f925b51941429ccabb69fa3627cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103a0a41ba6d2d2ebccab48aa152dacb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9094dbfcaa0cc27c866e2f4788eb61c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_734b3b7285d9b7254dbacf2596169acb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60d32c01b4a670c9be8b574733b0b044(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820c111c8c0ee4f97e00fdb2b2e71a0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0868a68cd0ac3074c935e697ce6005b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de4cd2222df07ea0180ce78f1bedbf12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5090d14f5beaa8426df2cabee1c1e036(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 3, 6, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5354de4035f30f3c35988a14fade3f73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5090d14f5beaa8426df2cabee1c1e036
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 3, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_37bb9e7e11bcc50a14f7e52987810e22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ef1d13bb4cdd6543345d2b44784fd4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37bb9e7e11bcc50a14f7e52987810e22
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9d2fc58e0cba07dd62d479c226ec9ade(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e2434aa20aa24604345f7355a0ffcca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d2fc58e0cba07dd62d479c226ec9ade
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68650482a9ecb052130975fd9b3a5e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84708b69cfa914b04bb4ff3a22a49942
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0fae9efed4f3c287fb7459c1e8a28fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f7c23c4267dd40040b1e4a1bf498495b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 196, 8, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de5996cb3ff26f33f9372017686f38c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7c23c4267dd40040b1e4a1bf498495b
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8de57a0e58b1949141161805fbfb93d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7c23c4267dd40040b1e4a1bf498495b
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_67dade50eedc387589f442135c926257(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 24, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4ea4408e48f370e8eea277cbe14bcab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67dade50eedc387589f442135c926257
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ed21bfe9ce7db8f8b65c8cf13eea3985(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 2, 24, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1a669da9d2c828929b876904603730e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed21bfe9ce7db8f8b65c8cf13eea3985
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ed2acc1641c245abf7383e6a55a658f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 24, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_754a6e04a3f48b9cc88147c08e15c8dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ed2acc1641c245abf7383e6a55a658f
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_560a554ff14e9d4ad575292576eb7192(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None, 150], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e875dcebe2c79e7d07dfebc1e941d646(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560a554ff14e9d4ad575292576eb7192
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bd611e16eadb581f610efa945e46125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24f259006e30a07a2a8765caa0bcac1a
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_678dc06ad4a4ac50b81922d65f52afea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5372481bdbb3841b3d59eff0861851a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 68, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f71ce75f2c563a744c0507f8a0ae3ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 34, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfad1adea851a467ee64c4df9327d7e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 17, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5df96edcee71a05c806a89b3b74b3789(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 9, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bab52226c617e53ebe6d796349c6af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_494272ac1fdbb352c73888482d7efda2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 68, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd576835e548b6608d0a48d9d1356d2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 34, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7a4b21b05b5279fd7facaea7565cf91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 17, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9831b1dae0e409df4540450bb1811f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 9, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_271531e666d3e6e242b53d2708daa4ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, 8, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_078c0e6b845760c2639f9043b9266613(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_271531e666d3e6e242b53d2708daa4ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16, 512, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_904b1c0bc69c223e1d018b85a3e95975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74fb00c31b27b1b61074361fee12bfb0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4538911feb641de23acf2516305d8ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74fb00c31b27b1b61074361fee12bfb0
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0868a68cd0ac3074c935e697ce6005b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de4cd2222df07ea0180ce78f1bedbf12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0774175e489686deaea841961ae4c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_271531e666d3e6e242b53d2708daa4ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 13, 13, 512, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ab59f8f6ce4247a3e744ecf132045d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce74003ca70cc20ccb7bba9a2d623335(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 3136], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a0400e7d63b9df54fd420b69205eb45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce74003ca70cc20ccb7bba9a2d623335
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_762a4a15126b581ecfd59c82279412c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 5, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef2325b5b4993f358d8de35e5c3998c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_762a4a15126b581ecfd59c82279412c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 5, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a20a4f14f67e7f63629ce99215b23c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_607dbb1eacadf6a3b094b833505f9e14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a20a4f14f67e7f63629ce99215b23c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08d9e56e59bf482f9daf586373bd242a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_58f266ae2ef89f17c8299a2796bd8783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08d9e56e59bf482f9daf586373bd242a
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81b36a588514f388e176c4d0c6b6e33c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 2, 5, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c985adc861871ff11202bc9e45a507ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81b36a588514f388e176c4d0c6b6e33c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 5, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6dfa8a6e871e066f689f08f36a4cd8ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05184c8d0bdde02e6e7dc2e0abaf8842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dfa8a6e871e066f689f08f36a4cd8ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_55facffaa0e371aa9858caa3c4426f82(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c93c691dcdd74067db4ba55804248024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55facffaa0e371aa9858caa3c4426f82
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b210632a18163b1e549e1ac3fd37ca49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_efd6e6798aecc93aea16b27c72f61af8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f22df97e3b9b7b048741a0e83289c4f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33035f6b7a66bcfc24f832181bc1d508
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f84a2d1cb12f8caf04bdf49bf279cb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0850d3fccf64417751bc26c3c7bf4e9b
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62450de1605e69bcadbe4c9cfe125b83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a60a1eaabd7e591aeca45ab9c46d9f55
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb938f55c2d062d04fd64a606bb5412c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdec3c32a90e8b3a0409358d408c3e31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_301aac15824cfe47edbe92be2a538804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 3600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_840df4e3eafcc3e203c1e25dd051e01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 3600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70b484a926673fcc8e4528e16efeaab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_844787bffdd8157a814b863514416014
        def get_inputs(self):
            return [
                paddle.uniform([16, 32, 64, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf6fc3219650b6092611ead1a25c78ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84708b69cfa914b04bb4ff3a22a49942
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9a6c3aa852387ba17fdccd0ee2ae841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8f6377a8955be4f864c5e085294dd45b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, 100], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0b7b1b836652f89550ddd49999167d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f6377a8955be4f864c5e085294dd45b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ce24587b314453406b807e2d466c291(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b300e7afe771dd0fff85588d4398f2a
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_664326ea73be78ceae355590da0dd2c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff82ee39581ed1311024b8276fe86cfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef886adf43cd44b0659c800029e0cb5
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08042e139f87f09cbfdc8cf9cac95548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36dca69bb6baec8bc7ac96325010cf6b
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc03fcdf22fcfbfc1739254f1d833b80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6295bd17f2f1f862fa6bb214dfbcffac
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75c5d559ca2522cd9fddd14350894e85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_353574e733805f56462f66a99d4c0df4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07b2959d3dbb462b9c08d667267deba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b2611b52b122e071e5b4d54a62dd28b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70cf539137f411504869b98ef2ceaa65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06e6750dbc9b761b47ad990270ba92e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d2b3a62f59b248a229ab3fbb9fc18f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6273796a71bd4ee4fd80effbf27088c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 2704], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dced495ae88854a88675bcac36372ff6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 232, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e82f85334e501674982b574455cad6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dced495ae88854a88675bcac36372ff6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 232, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61c58844a02e9e2f970038740bb95ffc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_411551885ba39ef4413237552f27c80d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c512a6a9a3bd0e9b3f473d1f03ce245d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebfe4951d147abec7054270921106da4
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e9b3f236c26dad76c8f78ee2be5f032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d04760efbcd9493e308cadff249a5c9c
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9fb0c217657a37dcbd089952c7c40af7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 197, 3, 3, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48b6c6191bd0ff07bc8e8703af28e493(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fb0c217657a37dcbd089952c7c40af7
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a06933ffaedd701b9bb61a73eccfac8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6798fe7719e93a7771282d905fe156db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a06933ffaedd701b9bb61a73eccfac8a
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7e55914b5a0aba7d7d8f3d3ce70eee1a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8e5f9e188b7b45763d197ea3acbd68d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e55914b5a0aba7d7d8f3d3ce70eee1a
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_64bebb238464fef4ec8979c35ca54ad0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 16, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_197164e8c3d4a7367f3d4983a8a7fe04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64bebb238464fef4ec8979c35ca54ad0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e29cd4639821bc1f328abd51cfdcb1f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_542a0f5d490f6739119409be6662dcfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 32768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_44b9d53138728d10d2c342457d33e1a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 1, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6483813b99d231db3c1fb19ff53ed1f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44b9d53138728d10d2c342457d33e1a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 1, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4bd1d941d8cdaec02f9bbee763b9ea6c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdcf3a8a864452dac7446a4715318c13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bd1d941d8cdaec02f9bbee763b9ea6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2eb012f837a4b7d686dcd0ad8fb669bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b4c0a6e853286c3b86071ef9a5bcfedd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2eb012f837a4b7d686dcd0ad8fb669bc
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bae23c6afbdcbbd3f474a082fdf154fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 2, 1, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a54b01118b91e5ad8f3c2b0322abc024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bae23c6afbdcbbd3f474a082fdf154fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 1, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f565caf07259e0270b6f61ee57777acc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c813d51096364eb27af7098ab97b4e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f565caf07259e0270b6f61ee57777acc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2e422802271a267c916848b5cb859f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_684bfab9eb37b3888028983fe4cdcfef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2e422802271a267c916848b5cb859f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2dac23aa74736b9dc0e41d5f607278ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_851dd8d68402e39c7d99d0427ee34f67
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22501ab89dfeb6c065f0655f47c7a3e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc1112b8883f14ef85ede21fcab7e8bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73b155e9c726c7fb94cdc4b703e7cdec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1112b8883f14ef85ede21fcab7e8bf
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d33552defb8b5a2ab9ba739e258ac69d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_047df291e8f273aa8b13600b19a42e9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb938f55c2d062d04fd64a606bb5412c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63b243766c263ccf81386fd48370c8c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_770ed9e46645ad15f65e0cab54377441(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4d34fe4a588b3bebc2cd7f6303b2bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de3bf14822669b64d3dbd6f6eec6043(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e35fbe21a10573f9a1a7d16481e108f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ce7086acf02e3e9390b4958d25ff43f
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf97d1375c4a23dba07836821d9af42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7e9a9a2553b5bd304f1e31dab8c589f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d4c2aa48887e0228a0d639ce7fc6708(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 529], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68d1bd65aa61b217ea61b03eacf3fa2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 529], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aec75da2717427a263e8c7b1f47ba24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe535e7eee578c19a040132351472630
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b9cb305ed002c7531e26f09f96b7ec1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ce6d418dbcc515306858260ab40cc94
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20d49b6527ee5918d8dee15402b82247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f37942e61d2574ac641badb08253d2f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475bcd04fb9af52baa73a6f559185088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ae3209ca54bbaa6c577d9025cfde049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0478ea92a9fe4dc44ae854e1a0c4b052(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b0fe0e670970c97967d676b0826d359d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0478ea92a9fe4dc44ae854e1a0c4b052
        def get_inputs(self):
            return [
                paddle.uniform([8, 16, 32, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_313cf6931e4be5ab5909cd328a8f15c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df8802d835a18bfdb7c625658fb4dbba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_313cf6931e4be5ab5909cd328a8f15c5
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57c645b06c7d1fc0e825c390b7435bd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bcad61fd40a8faabd4efb06cd0d74cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a40dcf1f13e525400e75c3718ae8531
        def get_inputs(self):
            return [
                paddle.uniform([4, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0bbdb98c086adf63912707b3f789dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57f54a63ee668ae6289b4c5245c2710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9740085dfcc65c6f3797367471116a8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0bbdb98c086adf63912707b3f789dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57f54a63ee668ae6289b4c5245c2710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9740085dfcc65c6f3797367471116a8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ade30624e4d96243ec6569be0a92856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad13f1f895f650a8943e408fe4483b04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c534a1df8d5f73b0aa31dd6779c34ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1600], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f825118e6cb4571b8d39c9e11ee8149c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 72, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d313fc5ef5ab60f84aa48550d601386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f825118e6cb4571b8d39c9e11ee8149c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 72, 14, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d60d89b1d47796c9e3c836e54fef145(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_271f22cf21bc4e4d4938780943d50e8a
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ab59f8f6ce4247a3e744ecf132045d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d274bd34e3181a560f4c82e9fd296091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2bd64a3246c680977b397a1d73ca601
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccd735b44799bfa4cfb8344734e89a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d7519500da3b2d605c9793b320d45ac
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d0a4dd10033a20bf602a110a40b0feb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c89d96c28e3154f52e2335c187f0838
        def get_inputs(self):
            return [
                paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7772d840062dfcb0baeb9cecada0165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_945c54a32fe72bef9a9e25ea14b3c4dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 384, 196], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d741a901db9d28b1541c0beb1bdfea3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_945c54a32fe72bef9a9e25ea14b3c4dd
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81bcfdfa9d09aff6ffafa8fceb583ff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0611cd5653f5c27fc0dcc779ae2beb84
        def get_inputs(self):
            return [
                paddle.uniform([4, 8, 8, 128, 4, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_898691882b3639c369f3914adb0adfcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_203598c93addf4624033be691238501d
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42a797a5f63dd15726e4b1e8bfe44de8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b803a4abdc784aa7744dacca9d6bab4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c367e65b02327aeb34a63577c8ebd3f
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02a83931fd60d3c6f83d45807d61671f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_062fd9c719698e68a30b7dff9f1f66d0
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66f8cdb4360a7fce0d7b12e747edd539(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dce1a972cfa7b6c9ef936b3dedf72038
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e71ab47eaf526b327c4ec72061f88721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 361], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9b9ce10951b1add7f9c975b7aa49332(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 361], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0868a68cd0ac3074c935e697ce6005b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de4cd2222df07ea0180ce78f1bedbf12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_02283ceb5f36527311a6ed64a1cb52f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 1, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbbe9469f68e518dbcd3c2c7991fe21d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02283ceb5f36527311a6ed64a1cb52f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 1, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f6add3c34ef0ec0d02f0198a8bfb2cab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3b03dc6e0d7b0f7ba8e51b037f6993f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6add3c34ef0ec0d02f0198a8bfb2cab
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57984f93996f568afd51f228f65a0ec4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22bcfd6447ea0eeea32dd7d0ae35c6fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57984f93996f568afd51f228f65a0ec4
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d0af790fc4e87e0d9a643b406b7fe2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 2, 1, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7cd441f610d8dfb4ae1d25dfc6ccee7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d0af790fc4e87e0d9a643b406b7fe2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 1, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4c8ee220756ddb5b17897827df1a83b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40206851019b418c2dcd7fac8285ca18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4c8ee220756ddb5b17897827df1a83b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c82aa40e1072af3ef7e4f3c20dd3cfd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7caf58a3ed54fc7e61fa7805dc2225f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c82aa40e1072af3ef7e4f3c20dd3cfd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4ea4408e48f370e8eea277cbe14bcab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67dade50eedc387589f442135c926257
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1a669da9d2c828929b876904603730e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed21bfe9ce7db8f8b65c8cf13eea3985
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_754a6e04a3f48b9cc88147c08e15c8dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ed2acc1641c245abf7383e6a55a658f
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e875dcebe2c79e7d07dfebc1e941d646(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560a554ff14e9d4ad575292576eb7192
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6874d243110276b48eabb1f15615804c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bb44f35b9f84b2b6629d5cb7fcf52e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8543598ca79fe743b607c268cc8101d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d4ded88eb2357e63e5ffd5cc1fe7d3
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 3, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0798a7146dc37b7c205fd771c1fb8e54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f67962df975b9c256c2fb8f58baf4ed
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d959ba59db3f714855abd1a47c39bab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15cc6b18714f96f66112b2facd7b03ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7caf8eebf86dbad543370bd1a4384818(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_32f1d7626b9ed79423d7126d80ad8fb8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a4c6a3e2604eb832ffd8c6a64b043c8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32f1d7626b9ed79423d7126d80ad8fb8
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_52064260e707e49fbc4010415a587dc8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04b32cdb1f708f7dbb96b2aa6fffd35b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52064260e707e49fbc4010415a587dc8
        def get_inputs(self):
            return [
                paddle.uniform([16, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_365b4844e8a70dbef12e7cbc6f3837fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52d9109188d49663466b766fb80834b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_365b4844e8a70dbef12e7cbc6f3837fb
        def get_inputs(self):
            return [
                paddle.uniform([784, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1155dd8cbb02b802c308bb1d99d2e638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_327f258fc8101bdb3854768236c39a14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7a6311d52c24acb0fb0e62f3858cb9
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81b0aa0b221d2424c01a67a71e8ed9af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 197, 2, 6, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54e140c489b63cfede17a3958c4ffabc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81b0aa0b221d2424c01a67a71e8ed9af
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e330f0c06e987586ea15a3502a4fdd7b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 197, 6, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbc4a842e55801f9a84b5df520a77b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e330f0c06e987586ea15a3502a4fdd7b
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_350d9f33c632e8a5f35bbd22d55905ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 6, 197, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d0fc24d6dabaeec03e067b05a689e86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_350d9f33c632e8a5f35bbd22d55905ed
        def get_inputs(self):
            return [
                paddle.uniform([22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4c3ba60d927271a8e6ba6827540e26c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ad3bc28c6c66c76915aba867398e0a47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 50], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4101fef762df03967b029803427e07ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad3bc28c6c66c76915aba867398e0a47
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88e277d0104da039cfa98079223e5330(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24f259006e30a07a2a8765caa0bcac1a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76f08e15d18cc312caa9bbe5c50c40b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d5705b3db3a030bb5819ea08b65df22
        def get_inputs(self):
            return [
                paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0bfc64f44b3654c7442875e9175e5ec8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 21760], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aca3e464b161974c63fc085d0b32198c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bfc64f44b3654c7442875e9175e5ec8
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84ece65e1f713c83ac9c45c965e6676c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b021c4a69f3f746d8ecf411c3b96f6a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8792ff653fd56a19d5cfdf896987bafc
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5751e9ca71fe06a82e71fd365ae563df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_818763e8ba84bde064a7d044c7ea27f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78de80e32c80e8fecfaac58860b0568a
        def get_inputs(self):
            return [
                paddle.uniform([4, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adb6765245039e57bd5928c846b982d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 136, 208], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22e82c7e9f59a2c17834390e3366fb26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 68, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e12ee050ac9c618f5039256bed11e72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 34, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2aad59492911f32b27ba1f94e650de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 17, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_127d48fd0c93cd9ff77a8281731ae615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 9, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f5c512baed0872e66b75e8ae0e541fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 136, 208], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33d7da82f9dcab441ae37e4435665121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 68, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19601e3f3719c9a24825ace02f148d8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 34, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8d908caf7ff52ea296c3ac34718ea9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 17, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4c8b4df33bd9c1a812dfc4dc1f0c3b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 9, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf97d1375c4a23dba07836821d9af42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7e9a9a2553b5bd304f1e31dab8c589f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f89eb2a59a21e86eb74e4c37fe0312b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 2, 7, 2, 7, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_85d21cd33ea6c14faefb25f6cd856ea3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f89eb2a59a21e86eb74e4c37fe0312b
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c4b5a5d6021e49a7a021e4eccd93af96(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 4, 49, 3, 12, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f1a593254d2c6808be06dd2b41d6e9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4b5a5d6021e49a7a021e4eccd93af96
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3a49df72dbc01f82712a1d8a1cc05fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 4, 12, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_627f108fd199ea35ae56c92b176950f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3a49df72dbc01f82712a1d8a1cc05fd
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c3fe0ea2f0d96ec0d8690aa253da18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8ba7bbe017000ba22529beb152e4e3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3e63d896263b116f97fd5ea632607f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4624], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d7f3945f0ce570f3878660d80c0b099(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 2, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_00c1268a3d59a68677b41f8dd8a2b83e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d7f3945f0ce570f3878660d80c0b099
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 2, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9b6d7e235dc77c7134ca688f734e6f4a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d65aa00ce036972ef6c7c0aac7f88e67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b6d7e235dc77c7134ca688f734e6f4a
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f93d96bd8e151f3e8922bae49f187d28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915a2d52bd49338934168550d95ac8e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c57fa6b697450eb5e413c4a081cb6b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_540f0c9db3b78134acb42074df3155c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 2, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aec0b39b3aebddd04c73ea7e34653538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b01fa7fd5ede57bc99df39fe5b24be99
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b5cc7e4feb60a654d81cd267f2153db1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e50142a5fe22f922f520d4b5250bea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5cc7e4feb60a654d81cd267f2153db1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5b806c7f31b54b1373ab5f620d2da277(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 5, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb830a4115c50785c83cffdb94eebc0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b806c7f31b54b1373ab5f620d2da277
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ffd8278ad6286ca1b3e007eafde963a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cca8c9ca5df94d6e35eab15136237a6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ffd8278ad6286ca1b3e007eafde963a4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8f25fc9e02e5346dc05935e1f1b08bdb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 320, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c29d56b4da68e56fc6ca7ce2fc08aa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f25fc9e02e5346dc05935e1f1b08bdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_99967150eebc620e755e5446513bade6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 2, 5, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6ce887fdd37c99247876089d5936c55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99967150eebc620e755e5446513bade6
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8db7a72cf3ae7643552b3b86f2aab6e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f38c16daf4eb02ab8cd08d4ea970204c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8db7a72cf3ae7643552b3b86f2aab6e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5f6efbe8affc7e00eedd9eb80bb074d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a35d6c635cdc2fa510adf7b555a98fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f6efbe8affc7e00eedd9eb80bb074d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10a8e20cc1455147b74860f4de3bca86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4b29d239249802735bdfd1798db9003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f3c0768a8359b9cfe5e7da09d94590(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 5184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcba15a7b110e9a02159729fd6a9cad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 5184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46a249261662e7f0b263ddfcdf180f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 5184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afcaf972b162a394b01b8146c8802e77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11efe6253cd2a07301087dd474ad4f6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84708b69cfa914b04bb4ff3a22a49942
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce0d0dd0275daf669e6eeed184c7b1b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03a1b64877810aceac7c5c4c6eff57b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_203598c93addf4624033be691238501d
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_055e1ac3b9b339476d58fb811388dfa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b895ba7630fde30667fe9ed4bf92db41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afdc384dc787fed9f8ab2d78de0c10d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c71feb927176e50a4da4143c5c360d0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1, 7, 1, 7, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bad367ee49cb79fbc7acf96634d12ad8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c71feb927176e50a4da4143c5c360d0b
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95b75055052e7adcc2df5f496a3a41f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1, 49, 3, 24, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a784dac776ad3f667d682f9bd5f6be4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95b75055052e7adcc2df5f496a3a41f1
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2df408d3057187104a82ea9da2c391b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1, 24, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6bf0fe4ec6d3f10e3bfd520559689b72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2df408d3057187104a82ea9da2c391b8
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c8420a5ce931c22b119a22491878033(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_62752d7b40db116425d8a500092b3909(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 768, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b4909794315f4f51e0225c1d90ddeb76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62752d7b40db116425d8a500092b3909
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63caa662627a2a71fed9f0faa81e6177(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24f259006e30a07a2a8765caa0bcac1a
        def get_inputs(self):
            return [
                paddle.uniform([4, 96, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fee3781a568cca093c1df6a1c4f0a08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff6bc93e35961e54a41f81ea52411982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b591602d40721aa4ba65f24d07ed08bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 2, 7, 2, 7, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1bbbbddd72fc61d1d5ba5fc959545ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b591602d40721aa4ba65f24d07ed08bb
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81233471ef6f74dcccf0ff794638594d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 4, 49, 3, 12, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ce3ae17df9eed288eb2afeb60b9d6b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81233471ef6f74dcccf0ff794638594d
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5009c1db75df88de47f6b4dbb5dfc7a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 4, 12, 49, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea08cdb32c6419ad5f32fdd2bdcec831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5009c1db75df88de47f6b4dbb5dfc7a0
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bad367ee49cb79fbc7acf96634d12ad8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c71feb927176e50a4da4143c5c360d0b
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a784dac776ad3f667d682f9bd5f6be4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95b75055052e7adcc2df5f496a3a41f1
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bf0fe4ec6d3f10e3bfd520559689b72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2df408d3057187104a82ea9da2c391b8
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32ce7d3b4b6f2dc499296e60a97cfe12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5e4595c729963cc67e7e38658803396
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 3, 12, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e4aaabc63dd24cff9a559ff8cd5bf68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e363412347a60abab7ad07040d437303
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a9ceb130f19831c083cbc73e97c9ab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_109532bfc0f354b1a5f8393154f672e2
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cad40db7be44a99afa1c8a68757fe777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ce7086acf02e3e9390b4958d25ff43f
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156d314ab3780196420474ad4e7bf51b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78143bad50c1dbbd6a6c5a61a812f12f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a40dcf1f13e525400e75c3718ae8531
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1bbbbddd72fc61d1d5ba5fc959545ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b591602d40721aa4ba65f24d07ed08bb
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ce3ae17df9eed288eb2afeb60b9d6b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81233471ef6f74dcccf0ff794638594d
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea08cdb32c6419ad5f32fdd2bdcec831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5009c1db75df88de47f6b4dbb5dfc7a0
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c31ea4e41dcec0f675eff66a5b2255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d73ac1446d2fab41e10c65400c7be6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc42daa531c72c13ecdf09dedb760db7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_203598c93addf4624033be691238501d
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_289f95d24e5907d75051fe404d24b801(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 20, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a15f3ddf412a0d7dbe1929d820d7fea4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_289f95d24e5907d75051fe404d24b801
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dacecabc464be88f5572d87e84a7a424(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 40, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95e8f5615e8756ced70b9498a47d63ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dacecabc464be88f5572d87e84a7a424
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_567ff6d1f5bde7f3076d0dbe7a8d753e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 80, 32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1cd626344ddd8166787acf53c3798f5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_567ff6d1f5bde7f3076d0dbe7a8d753e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e0c7055b308912d98cbe34ecd5ab33e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 160, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ecabdd85eedc1cf0a5278bbd0db1c80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0c7055b308912d98cbe34ecd5ab33e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcf513b26f182befc0f6b89116c9482b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5dd57ac345a297a772afeaad86e2a59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e26d9bd8805f31c3ddda52f71a3b8e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33fca6910bab54f261b06eed17d78e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc84df865742a3624d9ee8741d1d141(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56a6d6a48e9504b5dae65fa85b1f0ce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82a0cf1c7f5036d6dec2b8bce490c2c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c09e172beee0e44b71c4ace533478105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5c27a67301bf6f35b5ca98ac5ef52d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b2609d319b033ffeddd956fae2ff832(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b642405f7b7d312e2f0e1d7b8ffccd31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_946d06ba9961149e2c5ca46fa0750be5
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8beb61c967d2652f10b75ee308feb39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c638f63aa1da0ffa16a7a202a958c13
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_66e116ac140c25a7b8f0aeafea52f2ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81d01447cdf6a51b4ff13753af52a5a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66e116ac140c25a7b8f0aeafea52f2ca
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6cfc98fae4f2aece3be7c459058e279f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 8, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3ac884e4866d13b8f14d0765c14859c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cfc98fae4f2aece3be7c459058e279f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df814d6f0929efc19353a2aadd424637(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2, 8, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae0869bbac98f59886838394415245a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df814d6f0929efc19353a2aadd424637
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c92a3922d909d106baed4453da2f57e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b732259a9407b4c0536bfec9edd266b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c92a3922d909d106baed4453da2f57e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c6be5cb3fe68154ec016f1bc31b29055(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2889fdd42cdcd86cea604bc1469b066d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6be5cb3fe68154ec016f1bc31b29055
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42557d544e289e4f0474787808d1e55e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b46691b7998bc7f907679a785736ad1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_134007ac0b4b40ffe9be2c35f77eb4d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55edb4f2f55f291f124d5d726f20f46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98745fcdd24bc0f23605e28777f0a02a
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2fbf8b610cea4ec488f9cff7933a300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_432c4659e272e917c6114753489a3b0a
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b8865cd2cf78820686b784ce11b57c23(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 1280], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09f2bbc3d41e00ee4059b64af5ea9db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8865cd2cf78820686b784ce11b57c23
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1280], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdac3f23a1ef305d2a201eea6aac56b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22660f49b7d4b2e7e36e2b94755c032c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90c33943123ff7476491ecbc207bd47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6efc156a1bf93b6fa843f6107f98d40b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 768, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8cbe52f7bfe986f92374658e2113a606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6efc156a1bf93b6fa843f6107f98d40b
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85d21cd33ea6c14faefb25f6cd856ea3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f89eb2a59a21e86eb74e4c37fe0312b
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1a593254d2c6808be06dd2b41d6e9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4b5a5d6021e49a7a021e4eccd93af96
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_627f108fd199ea35ae56c92b176950f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3a49df72dbc01f82712a1d8a1cc05fd
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06e6750dbc9b761b47ad990270ba92e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a327f9e4cb2c5b3d2f1c8c6b365611f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a98dfb09fb480eafbe8e82ef3f881e26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_047df291e8f273aa8b13600b19a42e9c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_adbfa492f647686670686398734ab641(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, 7, 7, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d182ce7a2c4fed1dc73b1ff9b182610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adbfa492f647686670686398734ab641
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 11, 7, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_990295f3fc6a067a6c06d9021a9c04f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_771c44f9dfd7ef70b13141d8ce5d04d0
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5a42369f982101cf528d5038be90ff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19703cd3f09143b3c0a437ebd77ef7ad
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d2d37e00efb8f341d4ee33206faf384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4c08243a55dc78a22358f07b697164
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2cbe61257b5b5f85a58d3202b5d6071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_864dcf7b7ccdf907be7e2df2bc4985d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4e3014e13fe03f8357c83dfd236c68d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c06da9caa3facc877ed274204421c64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cad110a083939271ec8df0395d164f6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8464], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f088fd110ff30f16826d21868d582c9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 8464], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c3a6aeda83033db5dc12084a19ec0f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efaa3abc4b8b6b64985ff13490fd9cd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8792ff653fd56a19d5cfdf896987bafc
        def get_inputs(self):
            return [
                paddle.uniform([6, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c6c86a5f0c96714c59d388f90c0e820(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 5, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_075bdadfd6df3a144f52398a432c9e9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c6c86a5f0c96714c59d388f90c0e820
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ff1ac347450b61ba550590796094b13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a06435fdfaff58f846397f7cd992233(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ff1ac347450b61ba550590796094b13
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_512698d58e8c1f5c77fdebe01b429403(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f25fc9e02e5346dc05935e1f1b08bdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98c2aa7b5ff1d235b14bbde7d6d08259(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99967150eebc620e755e5446513bade6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7511e553158ea2e59c025168110ddb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8db7a72cf3ae7643552b3b86f2aab6e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_641a24620b96030f7f689292e8a798a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e6f0449f99b648396fe89d672e535277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_641a24620b96030f7f689292e8a798a7
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65a5e76238c02007ad99ad43504aa342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 200, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1339429fb68d4d118fe564404450ef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 100, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73b3a7d944adc766361c49e7c421e158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 50, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70d3a2c994601c4b3b612ba45d36d850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 25, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a17775b31165b28d2c006c4c8004c01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 13, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45451236b884efe8dcb6e7fd89d4892(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 200, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dedf1aecaec86dc61852e687843adb46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 100, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82a3bc8580630b0d8b08bd6909718f72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 50, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51ff1789685b4dd8cf292d4015ca358a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 25, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30d9898ef43060bc8ffb2ecce0a29651(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 13, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f605efd0499b8f92a71b2a64b3c1f2f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 5, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc41b3ed99482b74c67e40e9e72f4a12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f605efd0499b8f92a71b2a64b3c1f2f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 5, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_777414814acf47186aae0f819b47135c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2fe384fe0856807d1fab5f87d6862f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_777414814acf47186aae0f819b47135c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab24c3a5b39dced20df77767bae57617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08d9e56e59bf482f9daf586373bd242a
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e1121ccc7a3638841c6ff5db2f84563(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81b36a588514f388e176c4d0c6b6e33c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 5, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5f835e2c89cb81711006904c5c6ccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dfa8a6e871e066f689f08f36a4cd8ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_482302fdfc66820786704fde2256b8eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_096ccb2395a09f7ef4ff33fe5bc2fd0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_482302fdfc66820786704fde2256b8eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0bbdb98c086adf63912707b3f789dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57f54a63ee668ae6289b4c5245c2710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9740085dfcc65c6f3797367471116a8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58b961c1957d447b28f7e244dcf6f7e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7399885c688bbdd0c29bbbd02f57a53
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc433c7e27836f15a71eee5d532a0cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9940e77d7fa08cf03c2135114384fa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a52c72fda834a1d426c47f6c46bc7a86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 80, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_91cef9348816e2f9344548c5c443070d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a52c72fda834a1d426c47f6c46bc7a86
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f33dbfc01c1b58f66776c1326aa18ab7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ffbf9437342d7ee023a8506a0bbc6c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48582d7f9636bc1813000808f66d9bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b34c02e8a23cb378c1aadc48f8613a7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f1c254262e2eb9012276e827058ad72
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8576614c0f55ba4f7145fa3256838cbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3666fbd96f0f6178a27f37a1fdd1ed4
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_922f809e72012acfd65e181ee6b6c728(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2652c1e9626e535a8eb78611b3f75b7c
        def get_inputs(self):
            return [
                paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62340932d6b405717d20640d3ea367f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cfc98fae4f2aece3be7c459058e279f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7523279f5603e4ed154fc12f638454f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df814d6f0929efc19353a2aadd424637
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e937c700bb434a9dbad5c187f756478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c92a3922d909d106baed4453da2f57e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a280015554f467e440dfea388cbe66b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6be5cb3fe68154ec016f1bc31b29055
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2d1a2d8c2d190f3dac383c5ab0ade3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 176, 176], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36cadbc6d69f8c8afd64b359abb49376(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_158554f7019acdac8b9138e55919755f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2b9669dd827995c1854dc22c8edce50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d05b5ea7f644d59471fe1d81de8fd3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95cec26075f65b00c78a318f50f03d76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 176, 176], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9549683f407f97db400ee968dcf9331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fbbcc4f494586e62dd8c16b57bd54d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a59fff257dd10d32847f931e44695975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e0d8de6e4938d594306d03960fc9ebc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475bcd04fb9af52baa73a6f559185088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ae3209ca54bbaa6c577d9025cfde049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bd4b0d6a537c6f26183096783d0b3c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c428f29e5c8800441b4679b4bf92c4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fa368e2fb4cf39f3cba1bbd8b5f743d
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4950a2033c869b6ef473e7ac34ecf2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f79c2c2471aaa110fa38fa06f54c8912(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfb8470becfc2912eb208ef3f9ad9bc5
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5c5a89e383099304839f806bbb71485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2efc28d303926d154df6bde5aaba7d9c
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fe6c81ae4837092a8e4890253af7448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_189f73412e2897689bdb8d16c93806d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac50058365ea3a2b74bf9bb8037c7149(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cdfc6485070feaa4251c1670cdaeb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be7fdac4266094e79afd322bab5411ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cdfc6485070feaa4251c1670cdaeb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b886cebde68d764d5c942f556120a5e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66e116ac140c25a7b8f0aeafea52f2ca
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c78fc790a2a4133a73e348acaa5559b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1444], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03915086912f909046188fad4ddcc1c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1444], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0c740901c1ffdcd681fa3abaf6694f0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 116, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6f25eb4c1a83746f20a43fb2980f51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c740901c1ffdcd681fa3abaf6694f0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 116, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1546e977c89818f65e56f1343841c7f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f6b84477dbe7ceb7eadeccb81d02ab9
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7772d840062dfcb0baeb9cecada0165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cbb482ca01da568ac8fdd8b6695d50b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7838c7e55f6a099c91084528b6dd7a37
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f42f86c8edfc4ae2d01ee8f997650a81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc460daf7b6852ab898531efcefdf840
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76790b6960c38393b4a03dc63a5a75d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61b492a74c8b8e95992a37ac4a5bbc97
        def get_inputs(self):
            return [
                paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef014f0774a1ef0cb11dd6aa1b15ca85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff551abf45aef0f2bcc9b28aa043a77
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32f873a401b25ba0e2dd96f767070dce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1764], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_155e8fd9a43d9383cd6eca2eac0f5014(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1764], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_664326ea73be78ceae355590da0dd2c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_46f00ec38eab3b407e7546581d8f69ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 192, 784], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b525c0911e031df574a1291e02448ed6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46f00ec38eab3b407e7546581d8f69ff
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc54844d93516e808fbbb690adf6dddf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78d7702ad0461edd3a27b1a0c242d551(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 144], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d29a79725faf88f2a165c5780948fd6e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 2, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2005405816434bec2e74bd72ac795b11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d29a79725faf88f2a165c5780948fd6e
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e009abbcc9f2a7f654f5b3cce7e6e7c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96d40c1e096795bac995c0fd4f64b188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e009abbcc9f2a7f654f5b3cce7e6e7c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a955a801859922f3b1d76d75b567e9a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57984f93996f568afd51f228f65a0ec4
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0811e7639fae3f2588724f110a1a627(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 2, 2, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf2638c1a1fcbfebd682dcee360e36df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0811e7639fae3f2588724f110a1a627
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9ff6235922a4aa12bb2265e8e3970d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_507a2500b58d99ecb04b9b4f68282718(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9ff6235922a4aa12bb2265e8e3970d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eedf90e16f4c6f93b79114bc66f437f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f27103fcecfd08e99340391c11acb465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eedf90e16f4c6f93b79114bc66f437f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ffde9945cda08e70d6ec2f262a629af0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 2, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc722ddde1e60fb53b69631cf47c8437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ffde9945cda08e70d6ec2f262a629af0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_de626f57c2ad35af1ee02416fa7d9d36(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08d0582d6ef6b8e18587b75aa7c97e0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de626f57c2ad35af1ee02416fa7d9d36
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22bcfd6447ea0eeea32dd7d0ae35c6fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57984f93996f568afd51f228f65a0ec4
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa28062c17995893b0c2bdd45a906158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0811e7639fae3f2588724f110a1a627
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82126654511d1a0a7fda1ef36a4977bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9ff6235922a4aa12bb2265e8e3970d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ee3682fb6a8b543725769a55cb841ff4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2459d9df18a6ba7f10ac25bef29d274f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee3682fb6a8b543725769a55cb841ff4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_904386c7444165f51823128d173e660e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 184, 280], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1fafdc26fa02adc964dd497fb12ed50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71cb22d0adcd31dc6379cf0917fab3bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc7a4c85917bf2b080239953d96aec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac7fe5124d262f86950b2b18814afb08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7527c58a46c004e07faca7adacf924cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 184, 280], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c151b5f48f40bbb523163299bffd534e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_482a69dfd3f143e7640c2ad3833815a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a12def8f4d3edd88841952da73dcca55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3a801f16acb30a4ccf198bb6ddbaf57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_27e42c2de2c78a1e0672d2b43110e6ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1f8ed9a47ae88354b12574ee85be5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27e42c2de2c78a1e0672d2b43110e6ee
        def get_inputs(self):
            return [
                paddle.uniform([4, 16, 64, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90a8f76a34f29330e8c0a224905eaaf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c638f63aa1da0ffa16a7a202a958c13
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d79d320a8ea4583644601ec43d4d421f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 80, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a48691f3601c5758b02b6fe71c98fedc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fb0c217657a37dcbd089952c7c40af7
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82220076688c06dc663f95e3f490e501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a06933ffaedd701b9bb61a73eccfac8a
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8563394f5f3041cd0dbc325ab7df7116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e55914b5a0aba7d7d8f3d3ce70eee1a
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6be63ab6557b2efc271c3c683bdc5a8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 1, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45102475625c6a07281c83c4553b81b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6be63ab6557b2efc271c3c683bdc5a8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 1, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7ef873260c754bee8146721fca70a507(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be291a755cd3d7a8897a4b9db7087cbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ef873260c754bee8146721fca70a507
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_855ec8d2ff03f0b8e8e68b6b5055cea1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2eb012f837a4b7d686dcd0ad8fb669bc
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17f6f0f6241740277dec6722ce9ebafc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bae23c6afbdcbbd3f474a082fdf154fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 1, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20fbb88a8c5fc8e1e48050e65f9c7c49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f565caf07259e0270b6f61ee57777acc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48c2618cb468103ccb05750d7c3c4872(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05ec7f40e502871f332de752c07f97a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48c2618cb468103ccb05750d7c3c4872
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59324dec315ba4f7120e1224758b9fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c638f63aa1da0ffa16a7a202a958c13
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b88b9eb3868e171398431ff5be57f878(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27e42c2de2c78a1e0672d2b43110e6ee
        def get_inputs(self):
            return [
                paddle.uniform([4, 8, 64, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d503807294de6615024b89a17f80f341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c638f63aa1da0ffa16a7a202a958c13
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_672d23e9c9b5855b2997fc8872efdff5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_203598c93addf4624033be691238501d
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d54099d7fd65f3cb1658dee3cc667b2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb67a78affdfc42fc0be7fb9b7438a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e40711f0ec46ff01ef7b8a720f95473c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07bdc080600e2a746d1787aba597bd0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e40711f0ec46ff01ef7b8a720f95473c
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07bdc080600e2a746d1787aba597bd0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e40711f0ec46ff01ef7b8a720f95473c
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1c34fa51122eba1b37145f852a5b2bf3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 196, 4, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7dd8f2dc91136505470aa999abf60c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c34fa51122eba1b37145f852a5b2bf3
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_748806d054696df8c110094618df4bc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 4, 196, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd656c46bb02d3e8ea1ca60cab3547c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_748806d054696df8c110094618df4bc9
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 196, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fcb73ced7ffcbb6ffcc23c766b8c8c43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 196], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8cf695d6dd4f6229fb5f2a2f89fa18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcb73ced7ffcbb6ffcc23c766b8c8c43
        def get_inputs(self):
            return [
                paddle.uniform([4, 196], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4d5bcc0b4b57dc76979249599b2cff53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[38416, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d11ea34eca00a233efac7ef59638fa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d5bcc0b4b57dc76979249599b2cff53
        def get_inputs(self):
            return [
                paddle.uniform([38416, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b60553ad4badd6b25f6d1dec3afbe57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020494f0f46240c214d3680a64cf4d1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74a02ee76af605dd61d4b1943fe78448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c17080b043103323c28cc8f5cb7afc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac7fe5124d262f86950b2b18814afb08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54be1e58b7c452cb029b681c1d044ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12c82b12dc9e023b35c8d1d957f96472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26e47a3db5d0c75e4ab244054dde4b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_167d463c747ebf5e6d774a8f74a5e225(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3a801f16acb30a4ccf198bb6ddbaf57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d45516b13caa90a5e083f2cd5e5b6dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 3, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05a77222ff3580f44fe258a618cbe128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d45516b13caa90a5e083f2cd5e5b6dc
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 3, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11298f67e8aec9b62e4d6eeb1c66e188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0850d3fccf64417751bc26c3c7bf4e9b
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9355d17082153d9568d07e7d9ff640e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a60a1eaabd7e591aeca45ab9c46d9f55
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d3bdb7bafd129f457fb825a1fd2e5a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5169b08426d34942072794a7c0e564d6
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f44db5043472924af0a5f19269f19b31(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 196], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c51ada9bf6cc1b470ee514629602a786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f44db5043472924af0a5f19269f19b31
        def get_inputs(self):
            return [
                paddle.uniform([8, 196], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c54bbb401a2ae7522ce8722a9bb1a7df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[9604, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc5cc06959353b5acfbd26a6335f4b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c54bbb401a2ae7522ce8722a9bb1a7df
        def get_inputs(self):
            return [
                paddle.uniform([9604, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a0639a7dca3bc42370f9933008d0dc6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 8, 196, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_393603a7942a18fb54da0c96887622d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a0639a7dca3bc42370f9933008d0dc6
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 196, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ec9717a45d7eb3f63895790068c6fe37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_908df69ff704f0ebffb2822eb704b5f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec9717a45d7eb3f63895790068c6fe37
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_908df69ff704f0ebffb2822eb704b5f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec9717a45d7eb3f63895790068c6fe37
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8595e3aa41b07841045a183fe3b4199f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 12, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0662b23029fa526be97a51de78c0c114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8595e3aa41b07841045a183fe3b4199f
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f61359a869ea5097acbe49951af60534(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 12, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c7b43196b79266107521de32db8f2fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f61359a869ea5097acbe49951af60534
        def get_inputs(self):
            return [
                paddle.uniform([22, 12, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0fe633a8c814ea47d9d285255aaa7cf3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[12, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5bb638be44f274a451c028c621115add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fe633a8c814ea47d9d285255aaa7cf3
        def get_inputs(self):
            return [
                paddle.uniform([12, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_efc15e7881135d6dd456c6eee5a06f3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33e34a9c24534d09b2b279c4d400e1a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_efc15e7881135d6dd456c6eee5a06f3e
        def get_inputs(self):
            return [
                paddle.uniform([256, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2be226fb8e48116ffcc955115fb5944f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5090d14f5beaa8426df2cabee1c1e036
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 3, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d36726743deb20bf228564013ed43090(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37bb9e7e11bcc50a14f7e52987810e22
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9785e48663250205407676da2e842f16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d2fc58e0cba07dd62d479c226ec9ade
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e1d881ebf5f4942a6b596cf6a55c829(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ec84711734f3ac9a5d66433f784d495
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0328eb3d08cddd47f62e1599b8f09b5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0478ea92a9fe4dc44ae854e1a0c4b052
        def get_inputs(self):
            return [
                paddle.uniform([4, 16, 32, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeef69d424bcd9d5e5ee83e3ec37aa4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_313cf6931e4be5ab5909cd328a8f15c5
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fd26088fd7b278376c06a48e7c1ad793(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, 58, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bce0260ef9b8d8de4087efb758454fa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd26088fd7b278376c06a48e7c1ad793
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 58, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c3fe0ea2f0d96ec0d8690aa253da18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86890c4abfa53e37d505fdde59a021de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a85cce777f2c80a27b7a927956ac6d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 4624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33b385c519350d83efd3e0648705f856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a1650e059736a99e337e4867e1e33c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6832bb4c5ba5c17ff4642eb074d99808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee61bc3fc60b0d382e045749de96a56f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5e4595c729963cc67e7e38658803396
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 3, 12, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e91ede3b752b151aa3ef8c9eb0d1ac2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e363412347a60abab7ad07040d437303
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ac1fd1df3a560ca621eb63e21d0e2ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_109532bfc0f354b1a5f8393154f672e2
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5b1dfbe9a0ae6721ef8d8d3eb184ba6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 1, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bdbae182390cfa843be0b44d6f396291(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1dfbe9a0ae6721ef8d8d3eb184ba6f
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 1, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74e4942fa3364adfcbda0a748f14ee0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1cbad3931dbe57d402a4938d1b891aa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74e4942fa3364adfcbda0a748f14ee0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a955a801859922f3b1d76d75b567e9a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57984f93996f568afd51f228f65a0ec4
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0605ac6fd306d024033c2c281dd21127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d0af790fc4e87e0d9a643b406b7fe2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 1, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_650baf70d4a8eafff400636454842a65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4c8ee220756ddb5b17897827df1a83b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ea7004b75733702551618203aaea0e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_711c1b2ffc7111a88594daee735862f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ea7004b75733702551618203aaea0e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2370b1df07a7ca84268b63c025b7fe82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24f259006e30a07a2a8765caa0bcac1a
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42674d518393eee019c4c2f9e61bb102(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d45516b13caa90a5e083f2cd5e5b6dc
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 3, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06a65818e4cb134cd96360046615583a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0850d3fccf64417751bc26c3c7bf4e9b
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1fb02f29209a967f3b4740d7176e6da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a60a1eaabd7e591aeca45ab9c46d9f55
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbe4f7783d832e3c5986d79e39aa3857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d0483db87367e7285e07852671e8276(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 3136], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8cac99c104f0410a6af7a876a3b91000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d0483db87367e7285e07852671e8276
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0ffa1bfb3c090fca8e937edbee0dfb5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 49, 16, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d818f4f745f766a25bba832975d3e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0ffa1bfb3c090fca8e937edbee0dfb5
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8c0c0243b541c8aa1480b7f4ec791cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0ffa1bfb3c090fca8e937edbee0dfb5
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0868a68cd0ac3074c935e697ce6005b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f06f37c09087bddf10892f4edd6bfd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfbf729c6e85c239a283b05d9d4a8b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_664326ea73be78ceae355590da0dd2c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e90c03b362e5ded7b1c62c7e5a81958(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5a31565c137412fe5084594e73488a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1b41392063425c86651a8257b0531f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b33dd24936edf2563626c0b643509bb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1ce1d7e3bebc552f081f469fd58d89a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 168, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_392afd7de94d490b1912cf5ca46a7e15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75a7ee3c65ed4046a38e4983ad477d91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_896a842ea2842dd59572bf08c9c8f4fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbe516d61324dc99d34ecf0485409552(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3475e3a9f3b0e03dcd4c4cff3545fde9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 168, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39fc8d86bd746f820ae5183aba0dff46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 84, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24793c99e90bed8c228761defcc7e448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 42, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa1879c46fc1ceabd2c3a15ef4953dc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 21, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b946a689d3d713a431f6b85366a798d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_264da08812db7a9725864a27d21d1d6c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[2, 0, 1])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e164491d4d643ca9b4b0e67fdf526fec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_264da08812db7a9725864a27d21d1d6c
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e77a10c358ab764b9b8dbea6163f167e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e8e9f42979f7dfa562970eee17cb253(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_70030537caa7c657235fc920381d248c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9500b5944be53b8c8ce937cb64b12f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3055f42117ab7258d7a356220d5f32cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ea3672e5de6cc05470e8578e00abf1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11c162ed88a353d2de8afcbef9d80559(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e0f4599216cdf6796f0e197fd9fe657(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5733b284aab15c5841f656e516b026ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d42bf3c4e411144db21dce93b0ea1f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_685999a82baa61a20e25e973d4e1a3f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7661c0fbb1526bd3b02ab5e49f1830a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 4, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecac26a635db01995498abe99e71a02a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8fb4cb728926f887452b160fdd2a18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4950a2033c869b6ef473e7ac34ecf2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_040db168f6e5d52ca046334af8321ce0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa25bfc778fbdc0e7ae8ddc95b799b87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a63dac8ca41b3e7867cc20afc359926c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6284eca213c92077041f218b4aa84dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([16, 32, 128, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_853404789c83ca55f1a2b53a4cce26e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c03cede99185f7d22033c089602bae7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 7056], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a81742819b0705f30e86ef09225387f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 7056], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5bc1c84983d114663dcdd5dff8041ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b2585a5a8ee1c25792b19e7ee9d1334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9853acbd9394e3ed1acd23f02a0adb57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59a6102e832a48ed85976acbad13b8df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec2b82be0322aeea623f96f3bc852262(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c10fdaf44a75392a287258dda3c78aea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0945d30a988d3f958eb336bc4205017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63d03c57ac6704700c29b5dc48f5a982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e10c263d31e9006123f7c727fdee7161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_114eb5c8104b6e0340d9158b8f03fe12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8718448b220b1f4c48dbc86e6ab3649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46278c09246dcc73144185d3f44f943d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab67b8d82312044aa13402dda9725b8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ceec03e8fda1d1c2f0a657553f115bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed3fda0b608a55eeffed89c6b27f9726(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f81b0c43b60818caedc57e5d419d321e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c235557da1baf5b1560f28f6e062dd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c31ea4e41dcec0f675eff66a5b2255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b17de755b7740c784acc67a144dc933(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33b385c519350d83efd3e0648705f856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8bc86aa8392efb995f4a4637a4d8db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_574ea32336baf2655a271aee5b8497b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 225], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_541e1212bb8330e9d323e9da7a07cc53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 225], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1a66a5648525411351ec26ddad5545a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_264da08812db7a9725864a27d21d1d6c
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_117e9f59fdced30f11fad526b2f3fd04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6119bbe5a851bcb8954ada87fcbf9e4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d50053d40f51c7fe0d7687ea0e19ef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 152, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33deb4d18984899a2a2bd5c5779646a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13ebcabdf69eaf57a2d2124dc655ad09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c325e8019a6605a0da84af3c755bfb83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48582d7f9636bc1813000808f66d9bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19045dc804a7f7ecbf48527fcdf0c76d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a8c9678be75a507d0f62ae69cb2db7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ac2d9c95fcfafb327e9707823f0584c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ae33a98288b0e2d2e381cc075637a46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([128, 16, 8, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1a64493214d08d50558c9ff4b259b00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475bcd04fb9af52baa73a6f559185088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4feb71362b97bbf81b8d76ab8578369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf97d1375c4a23dba07836821d9af42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cf4a3d0fa3610de3d01b4d970c5bfe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 676], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c987e8636fa43892181ee9b826df762e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8887bb5e97c8da5daf4db4a2a5dd6b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 120, 216], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_55e9a6c1f9bdced9f4a54172897bb952(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 4, 5, 3, 1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd9d488ca93a5b9864c860cff5d16f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55e9a6c1f9bdced9f4a54172897bb952
        def get_inputs(self):
            return [
                paddle.uniform([4, 8, 8, 128, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4d4973f7da5a54dcabcb68bf6fb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 900], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17ef6182a339449fb02a2bafcd55c09a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 900], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06e6750dbc9b761b47ad990270ba92e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d189612c3ff490afb7177f44332778(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e474712b2e85e93fc2a90e2e83427b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbe4f7783d832e3c5986d79e39aa3857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_968f567d5d44888360043c7f8a2fac56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49aa8845319b50cbbd0930902e5867d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7da77d8af3785ab49553e41612ae2038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619a294726cf1edd7d8b0b7487c09960(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c00ef963c5e87861bfceacea770c5171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51b9041963a32ce55b13b1c4fb23d591(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f00d95c436ec4c5b0eba15b2b33b12d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4d34fe4a588b3bebc2cd7f6303b2bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc53a19bcccedfc816494fe61530c809(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3b48a90f61ed8329230338e5e5ba4fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47f20f5d98d0bd9bcb8c6ef1e9b1d25e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0bbdb98c086adf63912707b3f789dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57f54a63ee668ae6289b4c5245c2710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0553b86d36c42f67a1aa4741f49a0f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0e3af87b82a4ad418cb482e76a6e556(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d3a432ebc9274425a6f0499e37a81c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_146ea0c862891eac9fd7d46a080026b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3690da291200edbc447e6cec4bfa227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a91f0e792df92cc958af3c2bc71f6fd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fc550174bdc7418304cf55c035db74f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 5776], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b96e56ff29088864d7cbd5ba09974ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 5776], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7edb72a7c1281324140475138cfffe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_258d2dec3cd4238a112975829309af62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd58ac4fa6e13dcf8d8cacb6cc12caf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c9e9ecb2369c7493a795b87cd874957(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c57882c0c8e090c36d4c7b29129851f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c31ea4e41dcec0f675eff66a5b2255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_424897228944f97e370c62a02d1ae457(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_428d276a4ce7ed0ba9a3b7e1ee616f52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_298e3a19282eda2d7b3455ff31099d00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6c9f9a21cf8b1382cf14ee62f278784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([8, 16, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbb710a05fca31545c8a62a08a87e992(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e5f2c6d527e5f4b96b998e023d30212(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_053a4c8b364084366374d725915dcf6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([8, 16, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa751d891f4e9be7211d54e9b29e1989(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ff1cce43ec99a84958a525ac094f098(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4176ee945c8c09ad82f55d3fb27b5389(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 3, 12, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7e92d13584a30dda55c1776c7e8167f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72b1e9d699e051aaf120dfab53f975fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_840e5de81925216549526620520be216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1296], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98d67c89f75b153078c1a725d66dcd26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1296], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d403e454876fb7d5cdad9cee1e750561(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1296], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff4b81608143e5f746bb8f80a07a1012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a772731ca3dd522378a42adc69469261(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbea5665ea99b269f3d2cb80bde66295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9958a46a1d1e0905088209e4ba28019(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88f860685786f056034320a908277f16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([10, 6, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f54a48feeec32a0342bc36e43716eaf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1cc452e8f60ea078d38e5c5449180ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57194d0e14bbd05e45366ba85f6847ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([4, 1, 96, 96, 1, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1932038d7e05c39874eebdcef885b09d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae630b67933913a19a0f4fa11c252899(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d305ee4b2f302909700aca4ac002793d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c469f80c0a1a7020c63be1740b46b47d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 2, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3aa19e077bcb88f28f8f21d190cbed67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8cd0a693b12148a111c7613c6cd5bc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7e9c04379dffb5490e8814fa1d3c280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 2, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f56a78c78127712bdef43b6bf988d00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4854cdcbaac2f797e5b2b6b4c0d8df03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475bcd04fb9af52baa73a6f559185088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e474712b2e85e93fc2a90e2e83427b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbe4f7783d832e3c5986d79e39aa3857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_968f567d5d44888360043c7f8a2fac56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49aa8845319b50cbbd0930902e5867d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7da77d8af3785ab49553e41612ae2038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c261475df3014070027a8b6f4c165c12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([16, 32, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10f708bcf7c0068e7ff95afb71bfc149(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4d34fe4a588b3bebc2cd7f6303b2bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da0345643f051cf9bc9614e6c1b1005a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6d2e076df5e0612f36ab419e2467064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41df78616de1fae5772758399963dd58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71a429b8dc2d19f94b091d50ed10767a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f8eccd4c2403fd72bdb8799f8df2ca4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f58b4d8fd57f49c9366f4f30bc00a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_864dcf7b7ccdf907be7e2df2bc4985d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e525d39c87cfe3a200bac5bba27fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a5f21bb6f878cbccad29ae4456fd290(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([8, 32, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc220f47ea8558ef83beab396fe1618c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90c33943123ff7476491ecbc207bd47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffd48c91f34058e1e1cd66147f1892ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88ea020362d36e11c514138dbce30957(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d093e7b0a4a5d3952516267aecc76783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25b2e8d889ff8afefd676ac47656cafa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d093e7b0a4a5d3952516267aecc76783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7779030bac24925053f270aef1d0fe6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 3, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_984de2cfed24d4e83a85cdca49ccec08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf04f4797eadabca5d72a9635b8e6f8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01e3cdbb8f8fa11b33fe5a7b505baaa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc4d62ba5f7d0ca119dfaf069af6f83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4c424414885a09b369acbd6a6bbd6ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be661829ccc29b25e0d9183bcee58aea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ade30624e4d96243ec6569be0a92856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d88315cd56fa2b0870f5ae4693f2eec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b56b992f800d302e0d786417db8d576(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ab59f8f6ce4247a3e744ecf132045d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e810c095a5eb3f9dc8473e452da758a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ad6f04313218688c4c02edf52845ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd3173daef0d2301454144e0eb437e05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb0af856cd6e2de0161bbc4ef938a872(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6de96c842679f316b7ef1324bfc6ea0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79a7b1af7974839cf66c4825c5fae361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_702a6591e53c62d363035511a3db54a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f169e42e116c35c7bae629f6ca6133bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e3d8d9f3a63e6c27415d587d836f02e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ade4a87551346042e67da2daba76a99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcdf02cb3b28889cc8faec771d644f19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3598fbc725415f993435f512f43274d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2335b209a428d0f4945632076d154303(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 4, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f50d05533cf54aa750a601b9af93e133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33b385c519350d83efd3e0648705f856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8bc86aa8392efb995f4a4637a4d8db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_561fa710f2f3e5ac2d9ffe9efbd447fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 441], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25d58bd824fd72d0008dd2c4b641ccf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 441], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5bc1c84983d114663dcdd5dff8041ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b2585a5a8ee1c25792b19e7ee9d1334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9853acbd9394e3ed1acd23f02a0adb57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0e3af87b82a4ad418cb482e76a6e556(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_592c2af1dd665e7b82fecd7e8e8e3830(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_973176490dfd08e5faf08a5de976c08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da331f6b805d0392f7833eeeedafa88b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 1280], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcffaab1d13c0913dee15b1609ca2aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcf513b26f182befc0f6b89116c9482b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5dd57ac345a297a772afeaad86e2a59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e26d9bd8805f31c3ddda52f71a3b8e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33fca6910bab54f261b06eed17d78e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbe516d61324dc99d34ecf0485409552(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56a6d6a48e9504b5dae65fa85b1f0ce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82a0cf1c7f5036d6dec2b8bce490c2c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c09e172beee0e44b71c4ace533478105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5c27a67301bf6f35b5ca98ac5ef52d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b946a689d3d713a431f6b85366a798d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a85acc1eb28d8b9cb3d93fa8ae7716fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 65536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_676de93280bc2840afecebf4b93f0b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_447c67acbfcf245689402259b2d9aa2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([6, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88aee3da86aae97f4a58f453aa38bbb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 324], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_baaec74da1a8ac0fb603feab3d439791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 324], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ac544fcb1fa35297e39c0dbd4fd8e8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 324], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e479d42699604074a16db61ff8545f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93ec05d4b9bc72b811ce4cbbcdb89a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_172546e0a1615f561386acbbe1eda2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 289], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_568c0a6a2be823935db6ef8d206d9691(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 289], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a384b4822b4a398bb05831d47a53ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([6, 96, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_499abf921ff037f1d869fe3605856eab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4dc9c33d68171583ba6d05d83bb4b059(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7acb6fd7bce6303b674ee0275d122111(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d3ba0e4b253bd95cc01057c6bc7a3ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([4, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12c43878f79552cd08650f155bd915df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_298e3a19282eda2d7b3455ff31099d00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7c1732f3d4200dbf2c515a6e584d9c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_163d829b2719052d09eb278f9f0eb1e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81f1e2fffafc4f6a02052d715f6a3624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0367f266a7cdaf4c61b0adb45a0b927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c680aca0f495dba78e0efb03f0a5ab2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d6275c2fcebe648ed095b59c8c28683(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14d994308eeb588951f8900f2d77bc69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df795372f3ec29359062320c98b99853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([6, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475bcd04fb9af52baa73a6f559185088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4feb71362b97bbf81b8d76ab8578369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_885a1b8711587a1989966975fbb2198c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f169e42e116c35c7bae629f6ca6133bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01d749ef5d4303b48e5b3fb44052f1c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7772d840062dfcb0baeb9cecada0165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd5b8641f1b4e6dbd945d5210bedbf56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0501fb950daace01af6ecf3d12f57f00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e65b23fcb406fe0472eb17abcab3ea1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_272926c2c9b07d9be34ab3b82afe5d04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f62c4577fa1fda453c02bd4f120f1db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3055f42117ab7258d7a356220d5f32cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ea3672e5de6cc05470e8578e00abf1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11c162ed88a353d2de8afcbef9d80559(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40863bfc9f811d3212c61823ad801d6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_579f5b8ddee10e381ef0d842d71952d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37267ed6dfd907443a669d42bb8210bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac8a32d60348fbde1cf8550713c4e637(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 126, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_339cef1fd94f999541324a753cd61e31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_612bf84bbf25d3e377e3e50cd802d39f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 126, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_999432698b9823376d59d93d645a12cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39ed29cac91da14860612141c28048bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 126, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_188d845f9a4ffcfce46bceec5535e6ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33242f6a093c2dee962cc0faf49ffc16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9db3bdcbd11290a095bf1b0fd83c573(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[16.121261596679688]], [[18.27570152282715]], [[16.624515533447266]], [[17.452688217163086]], [[17.60372543334961]], [[18.380887985229492]], [[17.626935958862305]], [[16.684293746948242]], [[18.099266052246094]], [[17.77437400817871]], [[17.67827606201172]], [[16.585798263549805]], [[17.15077018737793]], [[17.22074317932129]], [[15.508727073669434]], [[17.932458877563477]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_3a04df86f27aac2f49c670c1873c7488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d23d5cfffe8c985ea837bc7828e0738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe4659e37926f59b5407f0f1030d481b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12ecc10c10db10a11002603bf1c3be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 36, 28, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5be8d551afd8a425f7de817b52821097(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93ec05d4b9bc72b811ce4cbbcdb89a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86b990f34f67a6ebd7c7b1d0474c62f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 289], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cde89065b1f4e82062ec441ca1d8d694(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cde89065b1f4e82062ec441ca1d8d694(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f3c7c7acb0bf43a8d24a5ed6313a98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49c9a7748292b78b9d66f25792ec1efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[1, 0])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dac269b0cde4ca4d59ed229feb5ccfa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([8, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0ad7c1702c718b3bdadfb25ac2a57d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([2401, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4950a2033c869b6ef473e7ac34ecf2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_111142167bd18c6708bb9c3494cf3d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48582d7f9636bc1813000808f66d9bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e576057a26875d2f90bc162cfe5e6586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c62ee5a60be858951a7938ad7e85973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5c49f6a5f1b7010edda12da2287df2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a739b0d5d6239e39eddd54a9f648a5aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 5, 1, 2, 4])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42e600a5018c47f86b04c71c3e92454a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a739b0d5d6239e39eddd54a9f648a5aa
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4, 8, 16, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1932038d7e05c39874eebdcef885b09d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae630b67933913a19a0f4fa11c252899(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d305ee4b2f302909700aca4ac002793d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da9733b89ce0bfd6e50c4f378d3b64c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a739b0d5d6239e39eddd54a9f648a5aa
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 52, 8, 202, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92c36308895c0e975f4d288c73501e71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2e8b101b26f480e9a73ee2ad76c6c9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dffd92a98c1d8bdc197d71faea63a958(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3b98283841bc48e63e29b0801064de3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59938c6d67dde93dc09bc08d60797ae4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60e20b6b312b8183dbe31469f17e705e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f01f925b51941429ccabb69fa3627cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103a0a41ba6d2d2ebccab48aa152dacb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9094dbfcaa0cc27c866e2f4788eb61c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_734b3b7285d9b7254dbacf2596169acb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60d32c01b4a670c9be8b574733b0b044(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f57c9aa882cc9c672b7a5372256374f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0868a68cd0ac3074c935e697ce6005b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f06f37c09087bddf10892f4edd6bfd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e218b3e3a5dbee49b039628586df30c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 3, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eaa8b3902c750b85403c0f3a3268cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac11405c9fa20dbfbc0995ff89afc36b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a4e911a1ccd8b1072ecb810dc51d840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0fae9efed4f3c287fb7459c1e8a28fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34a03ccdbf76c2d80bc01028c4620c6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_306a9c1e96b1769ac68bb353825a8c06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d8bdd5cf2d47c43e370ff90dca7f2e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cfe00bce015b39ef6e0c77a1ac253fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d58b2a19b18f0b9c8f2b38b626d8e319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c1a8d9f7561ef70da21d2413832af70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcc4c8da0129b4b532b5823156df71ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_678dc06ad4a4ac50b81922d65f52afea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5372481bdbb3841b3d59eff0861851a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 68, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f71ce75f2c563a744c0507f8a0ae3ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 34, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfad1adea851a467ee64c4df9327d7e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 17, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5df96edcee71a05c806a89b3b74b3789(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 9, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bab52226c617e53ebe6d796349c6af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_494272ac1fdbb352c73888482d7efda2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 68, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd576835e548b6608d0a48d9d1356d2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 34, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7a4b21b05b5279fd7facaea7565cf91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 17, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9831b1dae0e409df4540450bb1811f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 9, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74e027a27b45bb93a2eb5d77b6cdd92d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.transpose(input_0, perm=[0, 3, 1, 4, 2, 5])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bda1bd221f77a12fddf3bd1f0af91c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74e027a27b45bb93a2eb5d77b6cdd92d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16, 512, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_904b1c0bc69c223e1d018b85a3e95975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3c7362513dfd1d5d9254e26843e12f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0868a68cd0ac3074c935e697ce6005b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f06f37c09087bddf10892f4edd6bfd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19b2394528a0d506ac0ebf358e66ae3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74e027a27b45bb93a2eb5d77b6cdd92d
        def get_inputs(self):
            return [
                paddle.uniform([1, 13, 13, 512, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ab59f8f6ce4247a3e744ecf132045d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf15623d36f739d61c8248b9c71d91d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2b586bd3299be7c50fcd2e1c09e5bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 5, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_739e730a80b41197a799bea3f187a8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_155e9b1e65668cc63ee9587222b37c0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4518ee8b2dfae74730c08d23e8c02ce8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 5, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05a994e802ef296d73aa7a6aca7821ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c155c1eda6a8e6a249b476e93469acea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67238fb21b4050e920e5589ada908519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_592cdc4531604d34756a10de0136b37c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51ae430ce5233051822fba86666c2bde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84ac7d528cb21b5a45f1a62e401ed320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb938f55c2d062d04fd64a606bb5412c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1a7d6299ff8a4a398c4a32f463a4346(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_301aac15824cfe47edbe92be2a538804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 3600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_813451655ab79f48244682b4ad9e5184(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 3600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea553bfdc90ed4e022cd9079abaa9506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([16, 32, 64, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4a69f6b66f8d67e76a81306ea033d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9a6c3aa852387ba17fdccd0ee2ae841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6367e6e053f181f027d447117c6dfb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfbf729c6e85c239a283b05d9d4a8b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_664326ea73be78ceae355590da0dd2c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e90c03b362e5ded7b1c62c7e5a81958(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1b41392063425c86651a8257b0531f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b33dd24936edf2563626c0b643509bb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75c5d559ca2522cd9fddd14350894e85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d6275c2fcebe648ed095b59c8c28683(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07b2959d3dbb462b9c08d667267deba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b2611b52b122e071e5b4d54a62dd28b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70cf539137f411504869b98ef2ceaa65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06e6750dbc9b761b47ad990270ba92e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6756aa339faec85c28473aae64a976c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dbc59cc1b63a325af27d2d3b6839186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 232, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e3d8d9f3a63e6c27415d587d836f02e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ade4a87551346042e67da2daba76a99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcdf02cb3b28889cc8faec771d644f19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81c1e0c4cbf533870caeb1354fa777bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68ccf7c0b3f590ab1088d45acdc6ee54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18ae9274f50ccfebf575801366b8edd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd4ad12ef0056a7f6995b3a2b2e0eb46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f81c36c6c6250973af4d59a47685779e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 32768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46bdadfb857ec03cf3e60ed8278c86bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 1, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d8126a1cc29f8fdb5a906e090314aba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_380eb14b2c13afdcbf0025e8e5409622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abf66c335bf5d66caf2ab8b501cd5528(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 1, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a3bfc467bcbc17ef637b8788541098b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0e4e72a9f3e0422227537ee73a82132(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e164491d4d643ca9b4b0e67fdf526fec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_264da08812db7a9725864a27d21d1d6c
        def get_inputs(self):
            return [
                paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22501ab89dfeb6c065f0655f47c7a3e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd3640ac1e0370bf986451ecd9e8d111(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0868a68cd0ac3074c935e697ce6005b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb938f55c2d062d04fd64a606bb5412c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63b243766c263ccf81386fd48370c8c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_770ed9e46645ad15f65e0cab54377441(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 6400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4d34fe4a588b3bebc2cd7f6303b2bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da0345643f051cf9bc9614e6c1b1005a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a4616f2df88d57bf26764b7ea49b10d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf97d1375c4a23dba07836821d9af42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8156229e09b9e095783a06afc82130e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d4c2aa48887e0228a0d639ce7fc6708(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 529], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b33a3a3667f2567171ee56a97560d358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 529], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7edb72a7c1281324140475138cfffe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_258d2dec3cd4238a112975829309af62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd58ac4fa6e13dcf8d8cacb6cc12caf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475bcd04fb9af52baa73a6f559185088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4feb71362b97bbf81b8d76ab8578369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9a466cd85e999fd194d75243c7f8f23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([8, 16, 32, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7534aecd0c7a5f0c15d7f20b77db9b9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57c645b06c7d1fc0e825c390b7435bd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bd7b7183105a8ad32c950341554fa23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([4, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0bbdb98c086adf63912707b3f789dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57f54a63ee668ae6289b4c5245c2710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0553b86d36c42f67a1aa4741f49a0f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0bbdb98c086adf63912707b3f789dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57f54a63ee668ae6289b4c5245c2710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0553b86d36c42f67a1aa4741f49a0f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ade30624e4d96243ec6569be0a92856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad13f1f895f650a8943e408fe4483b04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c534a1df8d5f73b0aa31dd6779c34ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39b3872d362248642412eba97aa117(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 72, 14, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b56b992f800d302e0d786417db8d576(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ab59f8f6ce4247a3e744ecf132045d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e810c095a5eb3f9dc8473e452da758a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ad6f04313218688c4c02edf52845ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd3173daef0d2301454144e0eb437e05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7772d840062dfcb0baeb9cecada0165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b58dbc51f0ae3ad3aa3dd21587cf2341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe5d1a66304c225dfc525e4f015217d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55e9a6c1f9bdced9f4a54172897bb952
        def get_inputs(self):
            return [
                paddle.uniform([4, 8, 8, 128, 4, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f369d4d33a512627d1794147bc78aaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42a797a5f63dd15726e4b1e8bfe44de8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619a294726cf1edd7d8b0b7487c09960(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c00ef963c5e87861bfceacea770c5171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51b9041963a32ce55b13b1c4fb23d591(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e71ab47eaf526b327c4ec72061f88721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 361], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8431f15fbbb2e4ec30e5821ec05101af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 361], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0868a68cd0ac3074c935e697ce6005b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f06f37c09087bddf10892f4edd6bfd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88ec356002f8ad018ce701f360950bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 1, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9459fe72e2476a7b29bb1688504bcd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d990aa303a431f7f2cd461b913a92afc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19963d3f5d5822fd7e2280edaa7bc2f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 1, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fbe9850d3092324a651a4bae7aa96ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3499b2feb43485e500dbbf64f2829dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d8bdd5cf2d47c43e370ff90dca7f2e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cfe00bce015b39ef6e0c77a1ac253fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d58b2a19b18f0b9c8f2b38b626d8e319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c1a8d9f7561ef70da21d2413832af70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6874d243110276b48eabb1f15615804c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c8b85e07b1c82b85ec69a467230996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bb44f35b9f84b2b6629d5cb7fcf52e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_610a249d363c7fada43395f87d0b19b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 3, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97e31b31a104eec474f650298d2b35db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b97f8a46f325f84fb7c711d41dfe91c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fecf9d1944d628388662669dcfb2e95b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4165d9a24f40bbd50e9f17aebddf9fb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f30ab12f21d6f41da6be7006b5d61b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([16, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac9678e99bd40f292aca33e18dfd0913(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([784, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1155dd8cbb02b802c308bb1d99d2e638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c8420a5ce931c22b119a22491878033(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ec667f3e11de1c015428df6205734b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4027dad80669a4272bdfd09407e0ddbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f67fa20c4ce4e340c04a605ad659f0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([22, 6, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4c3ba60d927271a8e6ba6827540e26c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88206feaa147fbe039339fdb340d29c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4ff74eed09dd66a74326180c11f7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf078faa520dd450eb1e71078e0ab3fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4ff74eed09dd66a74326180c11f7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84ece65e1f713c83ac9c45c965e6676c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4836ec3380e09567baf10c2e9ea05b56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5751e9ca71fe06a82e71fd365ae563df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_838759682c906ade4e246e6ec8db1c6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([4, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adb6765245039e57bd5928c846b982d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 136, 208], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22e82c7e9f59a2c17834390e3366fb26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 68, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e12ee050ac9c618f5039256bed11e72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 34, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2aad59492911f32b27ba1f94e650de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 17, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_127d48fd0c93cd9ff77a8281731ae615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 9, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f5c512baed0872e66b75e8ae0e541fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 136, 208], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33d7da82f9dcab441ae37e4435665121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 68, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19601e3f3719c9a24825ace02f148d8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 34, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8d908caf7ff52ea296c3ac34718ea9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 17, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4c8b4df33bd9c1a812dfc4dc1f0c3b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 9, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf97d1375c4a23dba07836821d9af42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8156229e09b9e095783a06afc82130e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb2707b3db5b34968f46f7aa471fe46c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e50838ba8bc09a848e8553ea67f1b1c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70ecf3d5e0105f28c767819eb986f617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c3fe0ea2f0d96ec0d8690aa253da18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8ba7bbe017000ba22529beb152e4e3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3e63d896263b116f97fd5ea632607f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a53da641536086cd1248690596f9c7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 2, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd52fd8dd948778cb99c9bada0408042(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22fb523cfdc4529d584412afbab7c0cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a99846b474abf02b8dd65c8bd5b1a372(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 2, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ac7821a61bfa4101cff7a35e25c0e4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2e4d606a09c67165687aa36e29a777e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f4c6dd2c799851e422fba2732c1a720(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54e1ff6e9bf31fa0052cec63e9a66dae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c538e04ea86f3810f6f76de9547d2b6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1475d8261a90000a815a4ee4975d8614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7dc084d87f8041af77da0c55b0864dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92a3401ff088abac090e1d6019a60ed8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10a8e20cc1455147b74860f4de3bca86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d88315cd56fa2b0870f5ae4693f2eec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f3c0768a8359b9cfe5e7da09d94590(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 5184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcba15a7b110e9a02159729fd6a9cad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 5184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46a249261662e7f0b263ddfcdf180f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 5184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5abb11dbcf4bec5e79d48b0a1dd1712d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3112a35eb7b4196f05fdcf97dceac9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce0d0dd0275daf669e6eeed184c7b1b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbe5b930e43045b20ee805eaa4dda682(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_055e1ac3b9b339476d58fb811388dfa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b895ba7630fde30667fe9ed4bf92db41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61821050f124ee9828093891d449c037(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71054f631fdadf7e7931a4554b4ffb1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a05f035faa12cbcf92344e0fd600798b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c24200f2b04387be71de94886191a557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c8420a5ce931c22b119a22491878033(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a4616f2df88d57bf26764b7ea49b10d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df60fd7ccb149cc38c718ae26aaedd73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([4, 96, 9216], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fee3781a568cca093c1df6a1c4f0a08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e525d39c87cfe3a200bac5bba27fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e91e2cf6e27584d59e77f9dbcb9c51ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_542fd27e5d4091a6d062ac6ef0c5e543(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ddc34454e40f5be4f4de934bc419d2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71054f631fdadf7e7931a4554b4ffb1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a05f035faa12cbcf92344e0fd600798b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c24200f2b04387be71de94886191a557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f70f777bf895e41a382290bbe9ca2e0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 3, 12, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d22be252545b951da331d51b55e667ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aba4dd047af6b156b75d6dc6a9e768eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7339c3cb72e0de040cab10f85690dce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_156d314ab3780196420474ad4e7bf51b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a008598ae349ddf559e587568a6cdbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e91e2cf6e27584d59e77f9dbcb9c51ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_542fd27e5d4091a6d062ac6ef0c5e543(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ddc34454e40f5be4f4de934bc419d2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c31ea4e41dcec0f675eff66a5b2255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b17de755b7740c784acc67a144dc933(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bfcc70dd0284fb364c198d700236440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_117e9f59fdced30f11fad526b2f3fd04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6119bbe5a851bcb8954ada87fcbf9e4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be22e81a3f110b5e0521ee4cdace383a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6db0b29db88d5bff1f02804e43416a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcf513b26f182befc0f6b89116c9482b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5dd57ac345a297a772afeaad86e2a59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e26d9bd8805f31c3ddda52f71a3b8e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33fca6910bab54f261b06eed17d78e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc84df865742a3624d9ee8741d1d141(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56a6d6a48e9504b5dae65fa85b1f0ce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82a0cf1c7f5036d6dec2b8bce490c2c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c09e172beee0e44b71c4ace533478105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5c27a67301bf6f35b5ca98ac5ef52d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b2609d319b033ffeddd956fae2ff832(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1a66a5648525411351ec26ddad5545a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_264da08812db7a9725864a27d21d1d6c
        def get_inputs(self):
            return [
                paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d4f21304855e3b01af9694c5ead4af5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e576057a26875d2f90bc162cfe5e6586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cca0f04324c4999cfe168d796abead8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_478cff45f2d148ef0bc95cbc0a9fff75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3242ea426d8259f74163b0252d4dd04c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_540f3578e725176182bddec7586406d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65e64e5ef8d83dbbea12cdc7e340f2a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e77a10c358ab764b9b8dbea6163f167e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e8e9f42979f7dfa562970eee17cb253(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9500b5944be53b8c8ce937cb64b12f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ab089e58552c78db951787389728d40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1280], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe5a0c3330a9d11dffb6edb6b8b48488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90c33943123ff7476491ecbc207bd47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7339c3cb72e0de040cab10f85690dce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb2707b3db5b34968f46f7aa471fe46c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e50838ba8bc09a848e8553ea67f1b1c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a003b550febfabd0bed382a1cfe2e5
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70ecf3d5e0105f28c767819eb986f617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70030537caa7c657235fc920381d248c
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06e6750dbc9b761b47ad990270ba92e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d189612c3ff490afb7177f44332778(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0bbdb98c086adf63912707b3f789dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_192cb9608382147e3e4fe78b71083271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 11, 7, 7, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6d2e076df5e0612f36ab419e2467064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41df78616de1fae5772758399963dd58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71a429b8dc2d19f94b091d50ed10767a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2cbe61257b5b5f85a58d3202b5d6071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_864dcf7b7ccdf907be7e2df2bc4985d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4e3014e13fe03f8357c83dfd236c68d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c06da9caa3facc877ed274204421c64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 400], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cad110a083939271ec8df0395d164f6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8464], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa550fe3fbc0754177b641aaaece17a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 8464], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c3a6aeda83033db5dc12084a19ec0f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d059f2f2bd063dd561654dc8de1ecb5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9db6ab841e392ee858fc0ff0b39b1fe6
        def get_inputs(self):
            return [
                paddle.uniform([6, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_478ff844c31dbb24f67094c63cf840cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2c5629001266ffad6f170fe7787c62e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c25d615d238558d0ed9f861ffab1d6e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_674354e90c13e2f4c4ad14349542d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 5, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_096a9b4f12d890a6dc0260e6a150fdb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d976fa21645ad24c2a8208bbb3e8abe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65a5e76238c02007ad99ad43504aa342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 200, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1339429fb68d4d118fe564404450ef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 100, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73b3a7d944adc766361c49e7c421e158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 50, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70d3a2c994601c4b3b612ba45d36d850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 25, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a17775b31165b28d2c006c4c8004c01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 13, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45451236b884efe8dcb6e7fd89d4892(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 200, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dedf1aecaec86dc61852e687843adb46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 100, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82a3bc8580630b0d8b08bd6909718f72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 50, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51ff1789685b4dd8cf292d4015ca358a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 25, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30d9898ef43060bc8ffb2ecce0a29651(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 13, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdb7741a8b63b544da1a0a95cc77f9a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 5, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8456eb2429a5dc7f8bc3a31ca199bf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3951b2ee07054b594ec667573549e7b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b371c9084ab42f9ec1c214c692927cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 5, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd49893579bae2f92705da65180ed614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fc87d29c468e64e6b70298a9c6906b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0bbdb98c086adf63912707b3f789dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57f54a63ee668ae6289b4c5245c2710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0553b86d36c42f67a1aa4741f49a0f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_117e9f59fdced30f11fad526b2f3fd04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6119bbe5a851bcb8954ada87fcbf9e4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be22e81a3f110b5e0521ee4cdace383a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c325e8019a6605a0da84af3c755bfb83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48582d7f9636bc1813000808f66d9bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19045dc804a7f7ecbf48527fcdf0c76d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a8c9678be75a507d0f62ae69cb2db7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ac2d9c95fcfafb327e9707823f0584c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64436944f7ffea65890bc5d882f7c9ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36e8b9cfa905a89562181209341a6ebc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cfc2289683d931e55ed51e1dd9fb4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70ee2ad3ecc84a817d7bc17835133aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2d1a2d8c2d190f3dac383c5ab0ade3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 176, 176], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36cadbc6d69f8c8afd64b359abb49376(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_158554f7019acdac8b9138e55919755f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2b9669dd827995c1854dc22c8edce50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d05b5ea7f644d59471fe1d81de8fd3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95cec26075f65b00c78a318f50f03d76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 176, 176], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9549683f407f97db400ee968dcf9331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fbbcc4f494586e62dd8c16b57bd54d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a59fff257dd10d32847f931e44695975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e0d8de6e4938d594306d03960fc9ebc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475bcd04fb9af52baa73a6f559185088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3848a4c1103517270c96a5c615a74f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4feb71362b97bbf81b8d76ab8578369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8fb4cb728926f887452b160fdd2a18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4950a2033c869b6ef473e7ac34ecf2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_040db168f6e5d52ca046334af8321ce0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa25bfc778fbdc0e7ae8ddc95b799b87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a63dac8ca41b3e7867cc20afc359926c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac50058365ea3a2b74bf9bb8037c7149(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_063f509c7c4ba422012b02c0df569c5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be7fdac4266094e79afd322bab5411ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_063f509c7c4ba422012b02c0df569c5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b58dbc51f0ae3ad3aa3dd21587cf2341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c78fc790a2a4133a73e348acaa5559b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1444], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ffc474ff15de877cbc63fca90dc7e72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1444], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b141a9dbc25def759c797584fe71eefe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 116, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01d749ef5d4303b48e5b3fb44052f1c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7772d840062dfcb0baeb9cecada0165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd5b8641f1b4e6dbd945d5210bedbf56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0501fb950daace01af6ecf3d12f57f00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e65b23fcb406fe0472eb17abcab3ea1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_111142167bd18c6708bb9c3494cf3d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32f873a401b25ba0e2dd96f767070dce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1764], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77f45c9b4bf4a3da8cf584e003aa33a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 1764], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_664326ea73be78ceae355590da0dd2c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c62ee5a60be858951a7938ad7e85973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc54844d93516e808fbbb690adf6dddf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_827079a0ff0f2f78232da056f9c003fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_292411665807eff5818ce2569c471da9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13d65f3f8449dff4b816687b3d637638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b2fd46aa4c3b06a442060cd9a76a01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c1335ec1f1e67f845f075f4a66b4747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce5d9d3b7b8f2444dfedf1b30d11962f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac5ce58f83994d7dbc4c3b2572dd0793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c46bd184e2d14cf85ef38af98b8e4a3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b56b77fe0d788380acd565a61f9622e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d990aa303a431f7f2cd461b913a92afc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f7b4cea86426de5bf1de022546df174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 2, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c05610d6d2b2b6782f2ef407846d60c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47f7390fd446db12f3c13034b1e53f19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_904386c7444165f51823128d173e660e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 184, 280], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1fafdc26fa02adc964dd497fb12ed50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71cb22d0adcd31dc6379cf0917fab3bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc7a4c85917bf2b080239953d96aec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac7fe5124d262f86950b2b18814afb08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7527c58a46c004e07faca7adacf924cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 184, 280], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c151b5f48f40bbb523163299bffd534e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_482a69dfd3f143e7640c2ad3833815a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a12def8f4d3edd88841952da73dcca55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3a801f16acb30a4ccf198bb6ddbaf57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dc7623242d37ccf8676f1925c86c7ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([4, 16, 64, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83599f43ec6ba103524b7af54148d6c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d79d320a8ea4583644601ec43d4d421f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 80, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1de69f01dd33c36b12cee7b223d52018(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82faff39aa15542b4872d3fe06a96f93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b896c15c35f3e355b98996d3d0dcf72c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ba451454459713541520d8950375f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 1, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5bfd595f9cad979da98d9b61af01ff0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_919178180e1744a2da2aa38ee5ff8569(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_752568b27c90a0d28b64453a3f4b0f3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 2, 1, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64ef010779017a5bbf30c17678046ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd8b2acad77ff2c0b1d4ede157059bf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8f88b5c1e95d4ca083702b07859ae18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38859f45e3575221da9671dd1e2f7b89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([4, 8, 64, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af940678715f68a944b128f66f8cc49f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f5048ce0a5ece9a5e198bf125fdee00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d97de2c18c48e6d98341a466b2cd1de0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23c2536500f3c64f2733af4095d70d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23c2536500f3c64f2733af4095d70d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3a94e3d6d96bb79b925d736a83e6bc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4945df4736265460081be0c1baf1df5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 196, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddc4602c24e6395a22847f66c776b359(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([4, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bfda3a53efc8642c90ff25b37bcbb4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([38416, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b60553ad4badd6b25f6d1dec3afbe57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020494f0f46240c214d3680a64cf4d1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74a02ee76af605dd61d4b1943fe78448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c17080b043103323c28cc8f5cb7afc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac7fe5124d262f86950b2b18814afb08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54be1e58b7c452cb029b681c1d044ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12c82b12dc9e023b35c8d1d957f96472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26e47a3db5d0c75e4ab244054dde4b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_167d463c747ebf5e6d774a8f74a5e225(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3a801f16acb30a4ccf198bb6ddbaf57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd78c33eab3bd894d5c42b08c760a816
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5867fa1d82ea8543711c8baff81ef213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 3, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7602a7c4e86b2df82a8314aed48c388d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca3e1684eecc536c843e9d9c9c198b26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cde89065b1f4e82062ec441ca1d8d694(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea4407b41ebd00ead88f903dc2072e7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([8, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79f013083ddb9d01403dc984c750cfd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([9604, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb187002b9a2e005e3d441cd8ce71b73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 196, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bf76e68b547460318adb42ae33e09b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bf76e68b547460318adb42ae33e09b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79e7bee93d243093756e45b7b617a3f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_296dc85655ec11910bfaf4d6a3f9f5f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([22, 12, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b85e857845fa5b0f9cdf61d40eb8469(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([12, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9a890a6f73609b3995a6033cb8c507a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e11c5ad3a1b4f5510a5a0d7b6ed9ba
        def get_inputs(self):
            return [
                paddle.uniform([256, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4118653e66cde66aa5ab3ee89b7a9554(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 3, 6, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f67137a1c674028badf6a4174d928c5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdce7b8cf3660704647cf4084a7828da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f04455794c642e7009d2b93b31aedf15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7ba0f6f22bdaf75711773552f77ff0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5246b56bf74afa4119b5c39410cc37e
        def get_inputs(self):
            return [
                paddle.uniform([4, 16, 32, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f4bf47826059d456a0e35518badca4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_725d7c6e33da57a0c6c854444d5ba08d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df1e41d4762cd92f55c58572a49ef37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 58, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c3fe0ea2f0d96ec0d8690aa253da18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2dcbb40ee2c40f94aed6c6234a772b89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 4624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33b385c519350d83efd3e0648705f856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a1650e059736a99e337e4867e1e33c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6832bb4c5ba5c17ff4642eb074d99808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_025fd4ff5ecc070d16197dddfbabb06f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 3, 12, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1d57da25ed94cd60a6e781b96f95f03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_737f81254d46902196deb9498325b1bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08880a140e33f3ef198c610ba6ec15e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 1, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36fccfc1e90322cb9909ebe944afbc28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b2fd46aa4c3b06a442060cd9a76a01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9678c7f1a8bb7553425a0581ff95d3a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 2, 1, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e35148bbaf0841ee6a32463225a3d840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_044c19813473edf77e6d892e445c0464(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf15623d36f739d61c8248b9c71d91d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd18f82c3a669093362d4e3435c92996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a31565c137412fe5084594e73488a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 3, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b043650c4cc1d4c051bc15c24783ca46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90e9e49093eb904dd6e39377cc7cf353
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79324470afbb1e76376d2b6b13b30d3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbe4f7783d832e3c5986d79e39aa3857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcc4c8da0129b4b532b5823156df71ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ac46d0f44d16b22d37fbc98b520109
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70f6edcaa55f86d20a1b04136e893163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe43dcd7bf9dcec8d64ccfcd8d89b637(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72d295b2d1d65135d7640b8d40a2daf5
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()