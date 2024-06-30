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
    class PrimitiveOp_6df536c868d53aa469f0899cc78a77a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff91618d26e377ad2dba0d685e9e92ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6df536c868d53aa469f0899cc78a77a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7aa945b239bb1d79b86ddcfdd4bac87d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d52ca88e0c09a0f8f4a9f98bca98ae9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aa945b239bb1d79b86ddcfdd4bac87d
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9cf81bc896d2755a148cb5a888414db2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a87af525b9285aa1667050cd24f5c3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_157f894c607d4bdef7e6150acc96953a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a87af525b9285aa1667050cd24f5c3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f7a0a43cff2861115d9ba2176e37363a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e11e6d1398b3a127c1cb48b7eefc98b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7a0a43cff2861115d9ba2176e37363a
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b8517065a3324a8a60653519018e81d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2a271d04ec09ffdb9e4db71e1c782de3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed23d116f34e0bdbc4e1d22ba90a7b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a271d04ec09ffdb9e4db71e1c782de3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8f24b8d49c8106b92574c223692c100a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a31c02fa959800e05e25e1e417606ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f24b8d49c8106b92574c223692c100a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_218041950fa9c944f41aaca9420563ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d85dc0397394c8a1445f179756cf5f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_218041950fa9c944f41aaca9420563ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_66571b35a902515eddee38ab0df4555a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_902679044dd49fa1e56e975955339a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66571b35a902515eddee38ab0df4555a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c2ca2468955e0aa0442cb9c3282941f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_26083b3d84288ff1de4182c406cd599c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df7ceda55e7b4f0f29ed7345ea283388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26083b3d84288ff1de4182c406cd599c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_902679044dd49fa1e56e975955339a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66571b35a902515eddee38ab0df4555a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b91708bd9e4ead5dd3308b484499f5df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7145e10fbc1ccfec53b115451f9af2b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b91708bd9e4ead5dd3308b484499f5df
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.648627519607544]], [[2.1073200702667236]], [[1.7450263500213623]], [[1.0221208333969116]], [[1.6265045404434204]], [[1.3541351556777954]], [[0.8098263144493103]], [[1.3339897394180298]], [[1.4461839199066162]], [[2.339721918106079]], [[1.599525809288025]], [[0.8188310265541077]], [[2.1065781116485596]], [[1.1028285026550293]], [[1.7398396730422974]], [[1.1419532299041748]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be08b116a8d8ee2ca6b2a08f0e921bbb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_00c1427671adbf4aee160deb370c228f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be08b116a8d8ee2ca6b2a08f0e921bbb
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157f894c607d4bdef7e6150acc96953a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a87af525b9285aa1667050cd24f5c3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_902679044dd49fa1e56e975955339a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66571b35a902515eddee38ab0df4555a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff91618d26e377ad2dba0d685e9e92ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6df536c868d53aa469f0899cc78a77a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed23d116f34e0bdbc4e1d22ba90a7b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a271d04ec09ffdb9e4db71e1c782de3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2640b85dba774d71cba4191a20440d01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24f1a6218c703a415fbe5adb40547340(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2640b85dba774d71cba4191a20440d01
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1024bfd2e45bea8f190c42cdc5eeed25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc65b04b5f3d0d1cbdfeb2120c4bcb22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1024bfd2e45bea8f190c42cdc5eeed25
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff91618d26e377ad2dba0d685e9e92ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6df536c868d53aa469f0899cc78a77a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d85dc0397394c8a1445f179756cf5f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_218041950fa9c944f41aaca9420563ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a63a3d04814849ae071f487d2a72eb85(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_631614a257fc86cca39698c0cce1facf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a63a3d04814849ae071f487d2a72eb85
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dd95d7fd42e67008eccb20580421bc99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9cdbb516e79d98f47430865e8154fe1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd95d7fd42e67008eccb20580421bc99
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_902679044dd49fa1e56e975955339a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66571b35a902515eddee38ab0df4555a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_05330d6d451d6833925970dee974a0bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14718f6a8988b732e58ff2fc769f0495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05330d6d451d6833925970dee974a0bc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fd397604f2be9098af9ee305e690d9c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0f3098058acb5a67eecea5bce16eb1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd397604f2be9098af9ee305e690d9c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fa2a51faf6e32b9d24ac151af1ace63(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd0b8dc91d10b4d53ec01d9dfea51514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fa2a51faf6e32b9d24ac151af1ace63
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_631614a257fc86cca39698c0cce1facf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a63a3d04814849ae071f487d2a72eb85
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf460e9c974a0d4f14719a5956b6f4f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e47e1384ae9178aade2b0a6c842147d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf460e9c974a0d4f14719a5956b6f4f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a31c02fa959800e05e25e1e417606ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f24b8d49c8106b92574c223692c100a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9ac19c97cf0cbad3750b91404ed1500e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0d9bc1ca6923e80ea60c5ff03bb08ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ac19c97cf0cbad3750b91404ed1500e
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e11e6d1398b3a127c1cb48b7eefc98b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7a0a43cff2861115d9ba2176e37363a
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d52ca88e0c09a0f8f4a9f98bca98ae9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aa945b239bb1d79b86ddcfdd4bac87d
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0d9bc1ca6923e80ea60c5ff03bb08ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ac19c97cf0cbad3750b91404ed1500e
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f3098058acb5a67eecea5bce16eb1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd397604f2be9098af9ee305e690d9c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7ceda55e7b4f0f29ed7345ea283388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26083b3d84288ff1de4182c406cd599c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a31c02fa959800e05e25e1e417606ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f24b8d49c8106b92574c223692c100a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7ceda55e7b4f0f29ed7345ea283388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26083b3d84288ff1de4182c406cd599c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d85dc0397394c8a1445f179756cf5f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_218041950fa9c944f41aaca9420563ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc65b04b5f3d0d1cbdfeb2120c4bcb22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1024bfd2e45bea8f190c42cdc5eeed25
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5947320908906d78ede2e0885625246f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3c035ede55f7dfca29149e948cbf858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5947320908906d78ede2e0885625246f
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e11e6d1398b3a127c1cb48b7eefc98b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7a0a43cff2861115d9ba2176e37363a
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5f4599976b4c41700e523d3ef0e6be5a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c7e8a19ed26e01fd98139257688e5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4599976b4c41700e523d3ef0e6be5a
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7ceda55e7b4f0f29ed7345ea283388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26083b3d84288ff1de4182c406cd599c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_47bda5b18070ec17b390af347b6c5931(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a13a2bca0f5b960b4f2338efd7599687(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47bda5b18070ec17b390af347b6c5931
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157f894c607d4bdef7e6150acc96953a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a87af525b9285aa1667050cd24f5c3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8449d377e2d9d74c5b5c374ac5526567(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_31a76dc6e87dd0fd2277baae954eab9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8449d377e2d9d74c5b5c374ac5526567
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e47e1384ae9178aade2b0a6c842147d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf460e9c974a0d4f14719a5956b6f4f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7ceda55e7b4f0f29ed7345ea283388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26083b3d84288ff1de4182c406cd599c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98ab2eb1b52f2fb2af97bfc447845ea7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b91708bd9e4ead5dd3308b484499f5df
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3249717950820923]], [[1.268149733543396]], [[1.8114428520202637]], [[1.8886284828186035]], [[1.1243458986282349]], [[1.1445993185043335]], [[1.998379111289978]], [[1.9974102973937988]], [[2.0026330947875977]], [[1.9430570602416992]], [[1.677241563796997]], [[1.7557249069213867]], [[1.8666810989379883]], [[2.1212213039398193]], [[1.9509660005569458]], [[1.3167351484298706]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a13a2bca0f5b960b4f2338efd7599687(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47bda5b18070ec17b390af347b6c5931
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14718f6a8988b732e58ff2fc769f0495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05330d6d451d6833925970dee974a0bc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1d069d0d2dbd672101aec2a391b40576(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 400, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a41d212523665fb367ce546090df28b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d069d0d2dbd672101aec2a391b40576
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f3098058acb5a67eecea5bce16eb1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd397604f2be9098af9ee305e690d9c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c51c0f9393bb42bc711f0bf63a42327c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe44d87a275abc6bd64ac5729f099972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c51c0f9393bb42bc711f0bf63a42327c
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff91618d26e377ad2dba0d685e9e92ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6df536c868d53aa469f0899cc78a77a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd0b8dc91d10b4d53ec01d9dfea51514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fa2a51faf6e32b9d24ac151af1ace63
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d040471c51ec1aaf9014f18f9a51f08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bdedc2487f8505aa28d9a26c555e31b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d040471c51ec1aaf9014f18f9a51f08
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd0b8dc91d10b4d53ec01d9dfea51514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fa2a51faf6e32b9d24ac151af1ace63
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00c1427671adbf4aee160deb370c228f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be08b116a8d8ee2ca6b2a08f0e921bbb
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e47e1384ae9178aade2b0a6c842147d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf460e9c974a0d4f14719a5956b6f4f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a41d212523665fb367ce546090df28b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d069d0d2dbd672101aec2a391b40576
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a9d0f333c63be4c211916a5fcbe2b35c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3ff82aaa91117bec13bb65f9478b060(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9d0f333c63be4c211916a5fcbe2b35c
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff91618d26e377ad2dba0d685e9e92ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6df536c868d53aa469f0899cc78a77a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3c035ede55f7dfca29149e948cbf858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5947320908906d78ede2e0885625246f
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157f894c607d4bdef7e6150acc96953a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a87af525b9285aa1667050cd24f5c3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd0b8dc91d10b4d53ec01d9dfea51514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fa2a51faf6e32b9d24ac151af1ace63
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f2e03c3ee2e8b05e5f4dcc49d15cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f2e03c3ee2e8b05e5f4dcc49d15cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f2e03c3ee2e8b05e5f4dcc49d15cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f2e03c3ee2e8b05e5f4dcc49d15cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf4757d997025ce90cf312a559ea8303(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_525e2a850ba48c628e9bc683a9bae199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf4757d997025ce90cf312a559ea8303
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_525e2a850ba48c628e9bc683a9bae199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf4757d997025ce90cf312a559ea8303
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_525e2a850ba48c628e9bc683a9bae199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf4757d997025ce90cf312a559ea8303
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_525e2a850ba48c628e9bc683a9bae199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf4757d997025ce90cf312a559ea8303
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e47e1384ae9178aade2b0a6c842147d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf460e9c974a0d4f14719a5956b6f4f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed23d116f34e0bdbc4e1d22ba90a7b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a271d04ec09ffdb9e4db71e1c782de3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed23d116f34e0bdbc4e1d22ba90a7b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a271d04ec09ffdb9e4db71e1c782de3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed23d116f34e0bdbc4e1d22ba90a7b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a271d04ec09ffdb9e4db71e1c782de3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a4a0c2a0deffb9abc9e17ec4e631a2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 200, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0477aa97c80805a74733df005b63fd58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a4a0c2a0deffb9abc9e17ec4e631a2f
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a13a2bca0f5b960b4f2338efd7599687(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47bda5b18070ec17b390af347b6c5931
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed23d116f34e0bdbc4e1d22ba90a7b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a271d04ec09ffdb9e4db71e1c782de3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9cdbb516e79d98f47430865e8154fe1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd95d7fd42e67008eccb20580421bc99
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f3098058acb5a67eecea5bce16eb1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd397604f2be9098af9ee305e690d9c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_47b19e8fb7e68168fd6bf741e65d32e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f561098f3e52207a4d4bf7e93097e05a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b19e8fb7e68168fd6bf741e65d32e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e11e6d1398b3a127c1cb48b7eefc98b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7a0a43cff2861115d9ba2176e37363a
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c7e8a19ed26e01fd98139257688e5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4599976b4c41700e523d3ef0e6be5a
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a31c02fa959800e05e25e1e417606ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f24b8d49c8106b92574c223692c100a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e11e6d1398b3a127c1cb48b7eefc98b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7a0a43cff2861115d9ba2176e37363a
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0477aa97c80805a74733df005b63fd58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a4a0c2a0deffb9abc9e17ec4e631a2f
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_631614a257fc86cca39698c0cce1facf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a63a3d04814849ae071f487d2a72eb85
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e47e1384ae9178aade2b0a6c842147d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf460e9c974a0d4f14719a5956b6f4f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_435f2199beb773782a83a8cbf8e25586(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc5c402cf962f0e48dcf67ebd016bf88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_435f2199beb773782a83a8cbf8e25586
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e47e1384ae9178aade2b0a6c842147d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf460e9c974a0d4f14719a5956b6f4f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7ceda55e7b4f0f29ed7345ea283388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26083b3d84288ff1de4182c406cd599c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c7e8a19ed26e01fd98139257688e5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4599976b4c41700e523d3ef0e6be5a
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f3098058acb5a67eecea5bce16eb1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd397604f2be9098af9ee305e690d9c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed23d116f34e0bdbc4e1d22ba90a7b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a271d04ec09ffdb9e4db71e1c782de3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157f894c607d4bdef7e6150acc96953a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a87af525b9285aa1667050cd24f5c3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24f1a6218c703a415fbe5adb40547340(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2640b85dba774d71cba4191a20440d01
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc65b04b5f3d0d1cbdfeb2120c4bcb22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1024bfd2e45bea8f190c42cdc5eeed25
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17ad05949e927f9540ae412ebc970bc7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8de7a092569725f4287774f2c5d3780(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17ad05949e927f9540ae412ebc970bc7
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdedc2487f8505aa28d9a26c555e31b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d040471c51ec1aaf9014f18f9a51f08
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3c035ede55f7dfca29149e948cbf858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5947320908906d78ede2e0885625246f
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74530d7e38c87b7a48c6036824e4d178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6113eb82fa16cfbd61e85e9846ca4d41
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3ff82aaa91117bec13bb65f9478b060(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9d0f333c63be4c211916a5fcbe2b35c
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0d9bc1ca6923e80ea60c5ff03bb08ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ac19c97cf0cbad3750b91404ed1500e
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd0b8dc91d10b4d53ec01d9dfea51514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fa2a51faf6e32b9d24ac151af1ace63
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_631614a257fc86cca39698c0cce1facf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a63a3d04814849ae071f487d2a72eb85
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed23d116f34e0bdbc4e1d22ba90a7b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a271d04ec09ffdb9e4db71e1c782de3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_902679044dd49fa1e56e975955339a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66571b35a902515eddee38ab0df4555a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157f894c607d4bdef7e6150acc96953a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a87af525b9285aa1667050cd24f5c3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8de7a092569725f4287774f2c5d3780(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17ad05949e927f9540ae412ebc970bc7
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a31c02fa959800e05e25e1e417606ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f24b8d49c8106b92574c223692c100a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe44d87a275abc6bd64ac5729f099972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c51c0f9393bb42bc711f0bf63a42327c
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5898195759a5a7d8353bb4d6f3c42fc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1b4d35ac0cc2846add5c607cae2cbb3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d85dc0397394c8a1445f179756cf5f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_218041950fa9c944f41aaca9420563ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0477aa97c80805a74733df005b63fd58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a4a0c2a0deffb9abc9e17ec4e631a2f
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a41d212523665fb367ce546090df28b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d069d0d2dbd672101aec2a391b40576
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9cdbb516e79d98f47430865e8154fe1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd95d7fd42e67008eccb20580421bc99
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8cc8ab9c2b8b6962b839dd32a5dc46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cf81bc896d2755a148cb5a888414db2
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e47e1384ae9178aade2b0a6c842147d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf460e9c974a0d4f14719a5956b6f4f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157f894c607d4bdef7e6150acc96953a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a87af525b9285aa1667050cd24f5c3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_631614a257fc86cca39698c0cce1facf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a63a3d04814849ae071f487d2a72eb85
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc65b04b5f3d0d1cbdfeb2120c4bcb22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1024bfd2e45bea8f190c42cdc5eeed25
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157f894c607d4bdef7e6150acc96953a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a87af525b9285aa1667050cd24f5c3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e47e1384ae9178aade2b0a6c842147d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf460e9c974a0d4f14719a5956b6f4f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8c0fc09f3d1e614ae52af010577271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59ca2f20dc7d14730fcf7fe61302968c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c2ca2468955e0aa0442cb9c3282941f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c2ca2468955e0aa0442cb9c3282941f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c2ca2468955e0aa0442cb9c3282941f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c2ca2468955e0aa0442cb9c3282941f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cc34ce0bda2798dd28af71752bc59cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf4757d997025ce90cf312a559ea8303
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[49465.01171875]], [[49468.0]], [[55984.37890625]], [[49585.3359375]], [[42391.609375]], [[50689.87890625]], [[53767.9140625]], [[75058.0078125]], [[47822.71484375]], [[46676.1015625]], [[29999.0078125]], [[49514.0625]], [[51921.265625]], [[43394.1953125]], [[53613.578125]], [[44050.02734375]], [[50738.79296875]], [[71354.734375]], [[67753.09375]], [[53249.52734375]], [[53999.73828125]], [[49082.63671875]], [[50448.37109375]], [[51759.33984375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_55df726c251e6618c47f7c12668caf6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf4757d997025ce90cf312a559ea8303
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[61521.04296875]], [[77799.140625]], [[84940.5625]], [[70657.1953125]], [[77459.671875]], [[71598.859375]], [[51684.8515625]], [[50701.6484375]], [[83752.5859375]], [[62646.92578125]], [[72152.9375]], [[72591.90625]], [[57318.58984375]], [[67836.9609375]], [[58445.80078125]], [[32860.55859375]], [[59785.12109375]], [[58574.04296875]], [[59851.484375]], [[40407.8203125]], [[27642.65234375]], [[65305.703125]], [[58894.609375]], [[68620.9140625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f7887eb581d4794de7ea9ae0810106b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf4757d997025ce90cf312a559ea8303
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41777.6015625]], [[53148.0546875]], [[60683.1796875]], [[69294.03125]], [[62141.17578125]], [[51800.28515625]], [[45193.22265625]], [[54674.41796875]], [[71413.3515625]], [[61676.68359375]], [[106759.90625]], [[59531.80859375]], [[96026.4921875]], [[72264.3671875]], [[59338.96484375]], [[66140.71875]], [[43828.43359375]], [[77976.15625]], [[67321.9140625]], [[47203.640625]], [[28314.517578125]], [[44215.2578125]], [[72175.4609375]], [[39906.40234375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_914ad5eefd6dcf77a9d4a404f10bc824(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf4757d997025ce90cf312a559ea8303
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[103100.4921875]], [[76648.625]], [[64397.14453125]], [[91837.484375]], [[53317.1171875]], [[72990.84375]], [[46590.7421875]], [[82668.796875]], [[54914.89453125]], [[71202.75]], [[32888.9921875]], [[34030.546875]], [[70562.78125]], [[56306.61328125]], [[58270.6328125]], [[59603.296875]], [[95179.390625]], [[71455.0859375]], [[83349.265625]], [[59204.1328125]], [[80695.875]], [[33479.17578125]], [[70000.4921875]], [[84439.21875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_902679044dd49fa1e56e975955339a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66571b35a902515eddee38ab0df4555a
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f561098f3e52207a4d4bf7e93097e05a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47b19e8fb7e68168fd6bf741e65d32e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c2ca2468955e0aa0442cb9c3282941f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfe3e7bc0e365a2f688c5ead8d0d82f5
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a31c02fa959800e05e25e1e417606ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f24b8d49c8106b92574c223692c100a
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4827044d6f4135d47132166fbde51e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8517065a3324a8a60653519018e81d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d874a0bfc32314b9dfaae7b7a6ab2ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54e4497cd3f6f9e1240a0e00b6aef01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018fd5151aa768370dcfba0727a8c8d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdb54868220fdb1c540d84a8191b08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba2dee8c5c9ab2df8ce2feb5c514c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9a45f759e91a7066f67ac4e44a9d8dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a4fffa5c4d71949ffba76c7e99818d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87a00311a0924af19b14fff025d43e5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61436d80835bba669e50228be41883aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9f4a795b120fa4949cbc86ff1c3285b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87a00311a0924af19b14fff025d43e5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f78f54e819a49252c321252a29887b76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.648627519607544]], [[2.1073200702667236]], [[1.7450263500213623]], [[1.0221208333969116]], [[1.6265045404434204]], [[1.3541351556777954]], [[0.8098263144493103]], [[1.3339897394180298]], [[1.4461839199066162]], [[2.339721918106079]], [[1.599525809288025]], [[0.8188310265541077]], [[2.1065781116485596]], [[1.1028285026550293]], [[1.7398396730422974]], [[1.1419532299041748]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_755196e8bbbbd649a740afe12f393994(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018fd5151aa768370dcfba0727a8c8d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87a00311a0924af19b14fff025d43e5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d874a0bfc32314b9dfaae7b7a6ab2ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba2dee8c5c9ab2df8ce2feb5c514c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7149c5ad51f75aa48e2bfb8ff180a48f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59d9d397bdcfc95d51b9cd45e7a88e02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d874a0bfc32314b9dfaae7b7a6ab2ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a4fffa5c4d71949ffba76c7e99818d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd00028ad9769b0086f8040b395a74e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef3b585aa25738a605fd7ab65fc725f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87a00311a0924af19b14fff025d43e5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bd0eea2a9685dda356fda949fc3f7a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee13a2e08094d23ebebadcde33cc1fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfa569d501477e58d4b034615bd41a21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd00028ad9769b0086f8040b395a74e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_782587cc52eff7355355a61af295fe60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9a45f759e91a7066f67ac4e44a9d8dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ade0397fcb9f8a541d1c1ccd5f62120(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdb54868220fdb1c540d84a8191b08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54e4497cd3f6f9e1240a0e00b6aef01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ade0397fcb9f8a541d1c1ccd5f62120(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee13a2e08094d23ebebadcde33cc1fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9f4a795b120fa4949cbc86ff1c3285b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9a45f759e91a7066f67ac4e44a9d8dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9f4a795b120fa4949cbc86ff1c3285b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a4fffa5c4d71949ffba76c7e99818d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59d9d397bdcfc95d51b9cd45e7a88e02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b16d69f6dc4aeeca6bc67eb0dfd17da4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdb54868220fdb1c540d84a8191b08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e066652fb0a452a38af9e2d6d8e7dcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9f4a795b120fa4949cbc86ff1c3285b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de2a816019bfc959d15577dc364e6aef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018fd5151aa768370dcfba0727a8c8d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3964acd3dbb29e66379239e61c0c660e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_782587cc52eff7355355a61af295fe60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9f4a795b120fa4949cbc86ff1c3285b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76ae16b15dfd134bb2b12849b2da445b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3249717950820923]], [[1.268149733543396]], [[1.8114428520202637]], [[1.8886284828186035]], [[1.1243458986282349]], [[1.1445993185043335]], [[1.998379111289978]], [[1.9974102973937988]], [[2.0026330947875977]], [[1.9430570602416992]], [[1.677241563796997]], [[1.7557249069213867]], [[1.8666810989379883]], [[2.1212213039398193]], [[1.9509660005569458]], [[1.3167351484298706]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de2a816019bfc959d15577dc364e6aef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bd0eea2a9685dda356fda949fc3f7a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84238be9362396c64912859707486c6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee13a2e08094d23ebebadcde33cc1fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9385ccd44e22fc0fa5824d7d20a0eb56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d874a0bfc32314b9dfaae7b7a6ab2ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfa569d501477e58d4b034615bd41a21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c793b4b3ae8d9d6c76ba4880720c2fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfa569d501477e58d4b034615bd41a21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_755196e8bbbbd649a740afe12f393994(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_782587cc52eff7355355a61af295fe60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84238be9362396c64912859707486c6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4be1f9df15028ec16e0dd0ac2be9043f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d874a0bfc32314b9dfaae7b7a6ab2ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b16d69f6dc4aeeca6bc67eb0dfd17da4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018fd5151aa768370dcfba0727a8c8d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfa569d501477e58d4b034615bd41a21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1209eb15fa5adf85b42641574a23b0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1209eb15fa5adf85b42641574a23b0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1209eb15fa5adf85b42641574a23b0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1209eb15fa5adf85b42641574a23b0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdf1b4dbde6842f22484f317b6804206(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdf1b4dbde6842f22484f317b6804206(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdf1b4dbde6842f22484f317b6804206(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdf1b4dbde6842f22484f317b6804206(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_782587cc52eff7355355a61af295fe60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba2dee8c5c9ab2df8ce2feb5c514c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba2dee8c5c9ab2df8ce2feb5c514c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba2dee8c5c9ab2df8ce2feb5c514c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0357d3cf841a30a17efcd51d59cb03fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de2a816019bfc959d15577dc364e6aef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba2dee8c5c9ab2df8ce2feb5c514c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef3b585aa25738a605fd7ab65fc725f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee13a2e08094d23ebebadcde33cc1fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c7cde37ecd902125a5e348e8e0f7bbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdb54868220fdb1c540d84a8191b08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e066652fb0a452a38af9e2d6d8e7dcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9a45f759e91a7066f67ac4e44a9d8dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdb54868220fdb1c540d84a8191b08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0357d3cf841a30a17efcd51d59cb03fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd00028ad9769b0086f8040b395a74e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_782587cc52eff7355355a61af295fe60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_403a9041ac08930336b8d83543426658(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_782587cc52eff7355355a61af295fe60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9f4a795b120fa4949cbc86ff1c3285b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e066652fb0a452a38af9e2d6d8e7dcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee13a2e08094d23ebebadcde33cc1fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba2dee8c5c9ab2df8ce2feb5c514c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018fd5151aa768370dcfba0727a8c8d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7149c5ad51f75aa48e2bfb8ff180a48f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59d9d397bdcfc95d51b9cd45e7a88e02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b21b6bd57456e7b6efa1d6fe956eff56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c793b4b3ae8d9d6c76ba4880720c2fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b16d69f6dc4aeeca6bc67eb0dfd17da4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15a7a721393e1ed3edf6abcd38e046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4be1f9df15028ec16e0dd0ac2be9043f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ade0397fcb9f8a541d1c1ccd5f62120(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfa569d501477e58d4b034615bd41a21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd00028ad9769b0086f8040b395a74e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba2dee8c5c9ab2df8ce2feb5c514c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87a00311a0924af19b14fff025d43e5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018fd5151aa768370dcfba0727a8c8d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b21b6bd57456e7b6efa1d6fe956eff56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9a45f759e91a7066f67ac4e44a9d8dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9385ccd44e22fc0fa5824d7d20a0eb56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9ab32fb0639dbc1a3b0e95efe3a0aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a4fffa5c4d71949ffba76c7e99818d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0357d3cf841a30a17efcd51d59cb03fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84238be9362396c64912859707486c6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef3b585aa25738a605fd7ab65fc725f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecec177270bc8190491b14df38a141d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_782587cc52eff7355355a61af295fe60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018fd5151aa768370dcfba0727a8c8d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd00028ad9769b0086f8040b395a74e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59d9d397bdcfc95d51b9cd45e7a88e02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018fd5151aa768370dcfba0727a8c8d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_782587cc52eff7355355a61af295fe60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fd7e12979e57deb382745149a9c4be2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61436d80835bba669e50228be41883aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61436d80835bba669e50228be41883aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61436d80835bba669e50228be41883aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61436d80835bba669e50228be41883aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03ce4da80e66909c3be4679cc1b64a03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[49465.01171875]], [[49468.0]], [[55984.37890625]], [[49585.3359375]], [[42391.609375]], [[50689.87890625]], [[53767.9140625]], [[75058.0078125]], [[47822.71484375]], [[46676.1015625]], [[29999.0078125]], [[49514.0625]], [[51921.265625]], [[43394.1953125]], [[53613.578125]], [[44050.02734375]], [[50738.79296875]], [[71354.734375]], [[67753.09375]], [[53249.52734375]], [[53999.73828125]], [[49082.63671875]], [[50448.37109375]], [[51759.33984375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_c413059733bdc8db49c0d4e5f2e9d99d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[61521.04296875]], [[77799.140625]], [[84940.5625]], [[70657.1953125]], [[77459.671875]], [[71598.859375]], [[51684.8515625]], [[50701.6484375]], [[83752.5859375]], [[62646.92578125]], [[72152.9375]], [[72591.90625]], [[57318.58984375]], [[67836.9609375]], [[58445.80078125]], [[32860.55859375]], [[59785.12109375]], [[58574.04296875]], [[59851.484375]], [[40407.8203125]], [[27642.65234375]], [[65305.703125]], [[58894.609375]], [[68620.9140625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_fcdc8689e516c28bdf0f185864f502d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41777.6015625]], [[53148.0546875]], [[60683.1796875]], [[69294.03125]], [[62141.17578125]], [[51800.28515625]], [[45193.22265625]], [[54674.41796875]], [[71413.3515625]], [[61676.68359375]], [[106759.90625]], [[59531.80859375]], [[96026.4921875]], [[72264.3671875]], [[59338.96484375]], [[66140.71875]], [[43828.43359375]], [[77976.15625]], [[67321.9140625]], [[47203.640625]], [[28314.517578125]], [[44215.2578125]], [[72175.4609375]], [[39906.40234375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_670e1de343c7deefc093b991c53d4f77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[103100.4921875]], [[76648.625]], [[64397.14453125]], [[91837.484375]], [[53317.1171875]], [[72990.84375]], [[46590.7421875]], [[82668.796875]], [[54914.89453125]], [[71202.75]], [[32888.9921875]], [[34030.546875]], [[70562.78125]], [[56306.61328125]], [[58270.6328125]], [[59603.296875]], [[95179.390625]], [[71455.0859375]], [[83349.265625]], [[59204.1328125]], [[80695.875]], [[33479.17578125]], [[70000.4921875]], [[84439.21875]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_87a00311a0924af19b14fff025d43e5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c7cde37ecd902125a5e348e8e0f7bbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61436d80835bba669e50228be41883aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9a45f759e91a7066f67ac4e44a9d8dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e2328ca9b6d8a9abb08320d5a916f6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8547b563ff4e9adf61b2c8b5ffb83e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a130d1a3c36f532dc5f6b2122d5b87f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()