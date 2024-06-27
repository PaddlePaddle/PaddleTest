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
    class PrimitiveOp_b04a7a4d380828220079e45d5e1467d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 19, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12e19db4a06859b81e8d63d471d2643f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b04a7a4d380828220079e45d5e1467d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98ddb8e3955cc665604c46a7a5c9dc28(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 21, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6344a9ee90964e30f61e61cf51b07ef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98ddb8e3955cc665604c46a7a5c9dc28
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8bb5c7a533dbc9ff4a822dc28bd71dcb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ddcd98bbc5740a453a6aa4ae5e0b7a0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bb5c7a533dbc9ff4a822dc28bd71dcb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_daeced83f7005ed81bf1338428cd30fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_858903632ebc1abc6c05a380ee93d120(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daeced83f7005ed81bf1338428cd30fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2cc12a1ceffa89f0eb6d9e8a51174c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bb5c7a533dbc9ff4a822dc28bd71dcb
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_920feef8a2f00c7fb3a601627976e8da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daeced83f7005ed81bf1338428cd30fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95279e661b21381d3b7c4538ad1480dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea56b7b1d35238c917f2d2125cd31f90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95279e661b21381d3b7c4538ad1480dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0342cd8a6c6f0012a8b591bad8d06a08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9abad5949f07818a051f7256ee449cd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0342cd8a6c6f0012a8b591bad8d06a08
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bce9a3b8c44fc8791da1bb64a5d342f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95279e661b21381d3b7c4538ad1480dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4366eb12a85a405daf2e8f2a9625d7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0342cd8a6c6f0012a8b591bad8d06a08
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b5bd564487344d7276fdc59a02d8ecf8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 32768, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f642d1593c187626a5a6519d905b36c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5bd564487344d7276fdc59a02d8ecf8
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_740ee0766f5f2cc14d71529e47e097fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 16384, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4efea60e4eb877642f16e30f30ddc1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_740ee0766f5f2cc14d71529e47e097fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3c55a611995dd5554ebdf1852813da67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 64, 4096], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cee04dbdd3dbda562bb0f3261f1fd96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c55a611995dd5554ebdf1852813da67
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6cc48f5a65c2b9f1343192038af27331(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2db28a0d9ae62619cf2cbbe7b73d5a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cc48f5a65c2b9f1343192038af27331
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5cf7cef9df522c351ff42fc6d76d19de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 64, 8192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_914f2e80b28d661294588949f2d7e1ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cf7cef9df522c351ff42fc6d76d19de
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_82380360d32cd66d641dad5f11f67abb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9f3f94d4ccd7bc33a475e39ce18f7d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82380360d32cd66d641dad5f11f67abb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a507d8babed037c63efe563a3f544ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8192, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a04a2dcd91febf0645ab457442b6a99c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a507d8babed037c63efe563a3f544ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b94ca982c3c87323e0ea285ce25ee082(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 512, 8192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1f9d4eb61dde0a626e7e56a8d22f044(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b94ca982c3c87323e0ea285ce25ee082
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c75d3688af894889731c88d8eb6c38ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4096, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be2d7242513217f4ee950dabd91287ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c75d3688af894889731c88d8eb6c38ff
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c63a686057cc3256570a25cb38f5776(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 512, 4096], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_197054d7e5d213005c84b2d548cf5ff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c63a686057cc3256570a25cb38f5776
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_992d6f243b266b84f5c02deccda8eeff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d748d1840cd567f4ce411ab277594985(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e0afb8d92844c4d8a67208a54eaeb81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f87a0ba821e8dcebc7db81e0f5abdc9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c59fd5a2788dba111c73b4e474255bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb742dd04d6e17cd4428fffa42e4cf20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fdbcc733bfd6b186c47310db285b56e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f23d67bc3c9a3e06d02c09a275d2a01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6e4696efa0cd1a41d1779521689fc93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f6a6513f2efeb484837f484b0dea95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1e45802bf8a88098a3cc599ce6795e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()