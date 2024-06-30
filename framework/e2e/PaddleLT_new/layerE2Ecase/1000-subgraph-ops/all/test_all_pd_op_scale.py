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
    class PrimitiveOp_ac9d1d70fe871e96e839ad584322fa3f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b10a9085a9009c2131d0b18facd02741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac9d1d70fe871e96e839ad584322fa3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8f3c6e95d028939d79c5d501dd916485(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b12a5cfa30751f82821880bb05858101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f3c6e95d028939d79c5d501dd916485
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d09ae04244d10ffedb632df870a3aa91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 198, 198], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ba908e0b7e62a4baa3a86d9ca97ed71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09ae04244d10ffedb632df870a3aa91
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c900eeea87abdeb80554080d1bfbf6a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.85, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3682e1a192e0305b0ebe3217b42d502(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c900eeea87abdeb80554080d1bfbf6a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.47711414098739624]]], [[[0.48290640115737915]]], [[[0.9205596446990967]]], [[[0.48893752694129944]]], [[[0.47654974460601807]]], [[[0.945101261138916]]], [[[0.5162755846977234]]], [[[0.2689559757709503]]], [[[0.13848865032196045]]], [[[0.6119210124015808]]], [[[0.9737712144851685]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    
    class PrimitiveOp_62084bc3294f846f1db04644a1442863(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.1764700412750244
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0175b771fed801e908c849ac38b67e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62084bc3294f846f1db04644a1442863
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0ead1036e663537388e88cca9c282284(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.875, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba246b3b04f07153b62a1fde5bd59f49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ead1036e663537388e88cca9c282284
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7ebc2b632cf02877b3de2fd7e84a133(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.1428600549697876
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_efda602d03af5edca481a7bb8c4c947d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ebc2b632cf02877b3de2fd7e84a133
        def get_inputs(self):
            return [
                paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.5
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b456c63b89cd28ec40a34c715f9f83a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[1.0475993156433105]], [[1.3977339267730713]], [[1.1620545387268066]], [[1.302907109260559]], [[1.187252402305603]], [[1.586622953414917]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_691ccf7d0b92e550de85b72ae033b757(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[1.5293124914169312]], [[1.118884563446045]], [[1.217949628829956]], [[1.4280527830123901]], [[1.4317429065704346]], [[1.4073894023895264]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7607802d9cc04b82f1dacacab3c33286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8c378322e6b0009697443d39f56edac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f21cff98c9324808e92ecdeec1180945(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f098771130c44ca9e8d71d93b59fcdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9ca21a3fa96e8b18be4da1294232d97e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.10000000149011612
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20faa3873f665431767c42990d18abd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ca21a3fa96e8b18be4da1294232d97e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.41576087474823]], [[0.015224261209368706]], [[0.03696832433342934]], [[0.35867393016815186]], [[0.4307701587677002]], [[0.41868856549263]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_efcfc6055e9eb15c3ef37e63e8f11c90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ca21a3fa96e8b18be4da1294232d97e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.11388634890317917]], [[0.06622624397277832]], [[0.13420802354812622]], [[0.02208777517080307]], [[0.32159924507141113]], [[0.13441090285778046]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_90f0c9904512f2ccfef207ab9f13db8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.20000000298023224
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c5bacc83df2d5e97c61f1f506b1a352(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90f0c9904512f2ccfef207ab9f13db8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.3540762960910797]], [[0.4518500864505768]], [[0.2554236650466919]], [[0.377299040555954]], [[0.13825984299182892]], [[0.27891501784324646]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_209fcc58c50fdcdc0b9f0351c920c7ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90f0c9904512f2ccfef207ab9f13db8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4278714954853058]], [[0.14499519765377045]], [[0.3763442039489746]], [[0.23963865637779236]], [[0.4272620379924774]], [[0.002761698793619871]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.5
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9af6331f7e2b8c3ede97a55fca33122(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 9.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06e13373549b850f40258dedbc40da49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da479f60824321a0738da73412a0443f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df4bebbd214b80d70ee130121f266d89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b114d0394649af575014139ca430f4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a6f72b847077d007ed34effc3198777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_118f1751df7236dce37fa248afa33c48(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46b15c47acac509117c59531d493ccfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.to_tensor([[1]], dtype='int64').reshape([1, 1]),
            ]


    
    class PrimitiveOp_9678ff1abf4f89a99f77d3fcdb098d64(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d50093d5b9240fe83878b3cad753a4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9678ff1abf4f89a99f77d3fcdb098d64
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.24569982290267944]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_e3a98f673343d4e292bf763e5d1a560a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7643cfdc55b87d6e6615a5f85c63a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_203b516020a9e352f0a4b5a4a9ed4d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f1779aac405b33776aca3878970f95f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3a614282720b4cb812fc5b67efeddd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f1779aac405b33776aca3878970f95f
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fcc69ba211c9a2a188492f1f0a640c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04feae090d68f4a1b95cd732c5cafa61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fcc69ba211c9a2a188492f1f0a640c7
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8d4a5cb74cff473f5742cad280842ac5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da9bbe2f40bb83db41b01cc19d67e191(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d4a5cb74cff473f5742cad280842ac5
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_907d5bf891bc3d68f759aaaccb0c5a24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_677427594f5f8590ccb7d5ee92f14238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_907d5bf891bc3d68f759aaaccb0c5a24
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8c8ba054e02737e32d8daf8ce06a819(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9702f30523366666d6dae1b74867af7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8c8ba054e02737e32d8daf8ce06a819
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            ]


    
    class PrimitiveOp_7343cca5ec77f4c5a35e5ea640595c42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ecfabe2bc8ade5f9fbe1c7750525c934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7343cca5ec77f4c5a35e5ea640595c42
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_b67f7538dccef28ec4afc02b12255360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1cfe19cddde2132f90369751d42c6851(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ccaf1958bad370953b37ed140a866fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cfe19cddde2132f90369751d42c6851
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04feae090d68f4a1b95cd732c5cafa61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fcc69ba211c9a2a188492f1f0a640c7
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3b4fee886efc0607c8f6d8aac4c9ecbc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcfc300d9a223aa6674135c384a8fbcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b4fee886efc0607c8f6d8aac4c9ecbc
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_677427594f5f8590ccb7d5ee92f14238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_907d5bf891bc3d68f759aaaccb0c5a24
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49a7007c13884472499cfca05596b46f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_edec9a221140ca45a2ae7ea0bbfccc78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49a7007c13884472499cfca05596b46f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_14b6ec293ea01e9eb4cdd23430e87f35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7343cca5ec77f4c5a35e5ea640595c42
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_d5f651f18d1ad632440ac56f1b6d5546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4539879e90bc700d8f7dab5b7c27904e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2.5
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7bb2b3cc78c3d0b493e1d2ebc5ebc480(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2d50b636a4d28ba527a744dcf9b95c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fbc2160cb59a6c3183ae633b3d95723(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6aa71ec6e566e0cf8e23c190cad96f6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(1073.86474609375, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.00390625
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_86aeb6d74b57ce6de68ef1bd22c3a053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(179.33457946777344, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_bd91dd62e403f6cee056ad98b3de2e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(5.321410179138184, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_ab13770c1c8e73f0c8c33025d87e6b5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc15d824fbb144fec18e9793e10865e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab13770c1c8e73f0c8c33025d87e6b5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.00010448904504301026], [2.6467989300726913e-05], [0.0014538541436195374], [0.021787526085972786], [0.005431822035461664], [0.00016644439892843366]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_de596242590af9d50c37d58bb4faa479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab13770c1c8e73f0c8c33025d87e6b5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[2.793305611703545e-05], [0.0038048368878662586], [0.0025045897345989943], [0.0007974092150107026], [0.0007925934041850269], [0.0001352709368802607]]], dtype='float32').reshape([1, 6, 1]),
            ]


    
    class PrimitiveOp_c8465df2a75f3c60dde60a5935b0acdb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -6.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3bf5f2f9b24cf713cda09634f823aeaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8465df2a75f3c60dde60a5935b0acdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_031926322182bd3892e632c3dc516a60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.08333329856395721
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b728d1787929b693e4b3a68a2310ec0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_031926322182bd3892e632c3dc516a60
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.02352176047861576], [0.1643000692129135], [0.1283266544342041], [0.1822909265756607], [0.12502682209014893], [0.02151423506438732]]], dtype='float32').reshape([1, 6, 1]),
            ]


    
    class PrimitiveOp_273215780e84c7da687e0d3ad184cbdf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 6.28318977355957
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56f5b8a38ce5927c10e40c6701684ce2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_273215780e84c7da687e0d3ad184cbdf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.001960145775228739], [0.013691666536033154], [0.010693883523344994], [0.015190904028713703], [0.010418897494673729], [0.0017928521847352386]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_e5b37783148433b1378bd742ec874fa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9678ff1abf4f89a99f77d3fcdb098d64
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.012315968051552773], [0.08602733910083771], [0.0671916976571083], [0.09544733166694641], [0.06546390801668167], [0.011264830827713013]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_77d7bb0d6f1abddb984219d02b200185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e511fc182a29090a8fb07f3b8b6c5a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad52d41a06c23856247e08781b1b74a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c17637ae3f408b857b40ce62b3686e10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.9125, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb4b7b4d891d4d219ce0ee68394d9b47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c17637ae3f408b857b40ce62b3686e10
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b7f7df3f4ced853bbfbd3ffe92c0a90f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0958900451660156
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dee8bbe52fde111b335202fdfbf4a21b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7f7df3f4ced853bbfbd3ffe92c0a90f
        def get_inputs(self):
            return [
                paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91a8a8669394de3bdcc4356cea30c269(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f3c6e95d028939d79c5d501dd916485
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6fd950033e496cd13f3cdd9c9a97cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            ]


    
    class PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.25
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c20ec976fb6b3a1af53207d15aec7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.to_tensor([1.823246955871582, 2.051959991455078, 2.0047521591186523, 2.0986404418945312, 2.0738117694854736, 2.2010343074798584, 2.2370846271514893, 2.0122716426849365, 1.9902360439300537, 2.1795969009399414, 2.0542681217193604, 2.046412229537964, 2.133507490158081, 1.8996853828430176, 2.2763595581054688, 2.2851154804229736], dtype='float32').reshape([16]),
            ]


    
    class PrimitiveOp_549818d247c2f9764c426cc976066460(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.25
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43528a6abd61fe8ee8e868f3fdaef36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(2.0265331268310547, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_ac86be1b700219c9f93930a997de2133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43cfee128c96bebbd6919e6fdc5ad7ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22c2cccf6ef68d86a507f30833a69693(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d94482f1eae6261eaa543878ebe5808d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.5
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c82da99c7b7e93d4a50d66d0e2b1ed18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d94482f1eae6261eaa543878ebe5808d
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c82da99c7b7e93d4a50d66d0e2b1ed18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d94482f1eae6261eaa543878ebe5808d
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47fcf005405ef197a6440a8f8c21a2d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7607802d9cc04b82f1dacacab3c33286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71fab5e08f945192faa93e469329bcb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d76f7f66d33e0a55d258b8b6d1bc2cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea5fd51b8c2e3bf57cfa6498b0572286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f639e6904d531a1d14510c914d1ac557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e069f8d10c16199271b34812ada89fc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(37.86699295043945, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_b4a6f83e9516ddda198ae398cef845ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_709228d1b19715d514f1e2554404a0c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4a6f83e9516ddda198ae398cef845ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57589b61b753e1bec45f712b1c22936b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2b4d51baee2ccad1c2b2ca32a8d2f862(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 80.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_853a07fd0f5a797439b288522ac5413a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b4d51baee2ccad1c2b2ca32a8d2f862
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    
    class PrimitiveOp_4e0161d80430d39567f3df0ae9557e75(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d8e9f42d9ebc9afe37d07deddded7a51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e0161d80430d39567f3df0ae9557e75
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_03b7c841681cdae509b8a3b3929348e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c52e26335e980699085d3cd57b0b68df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_353248803624d0af2db2b56aec96499e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c0eba9469a54ba71c383cb84dbfe22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_56da15306a87e40ff4b91583b5b2a9a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40eef2024c7527db313c0a72b56bc933(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56da15306a87e40ff4b91583b5b2a9a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_00b12f29ed446d69dd4dda937595c107(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5fcc2408ef938654163a7f083d90981b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fcc2408ef938654163a7f083d90981b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2ddcaf3490cfd58b52df5e1b71aa078(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aadb9f5a1740aca287fd12197186109b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b5570a5615323312b26b87548f3e5ee0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88bd3d39af3a459d779e5731620223a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_7acde060267c3c392fdeafdb588672fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_afb0f43530280625e1e441c2b9a4ecc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8adfada8dcf3bf0f7bf02701da5a69ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_8adfada8dcf3bf0f7bf02701da5a69ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_3d0cba48db4657988deadfabf5378a7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef8b96811288a149ca8c33b3ee46ec59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83e686f9d145322656ecc5a1eca07db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ead1036e663537388e88cca9c282284
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.2360266000032425]]], [[[0.01978977769613266]]], [[[0.5263948440551758]]], [[[0.7462403774261475]]], [[[0.5974621772766113]]], [[[0.5967616438865662]]], [[[0.3465817868709564]]], [[[0.9959466457366943]]], [[[0.9737491011619568]]], [[[0.9637360572814941]]], [[[0.9228842258453369]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_84bf886880164e57100de4e90c912f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ebc2b632cf02877b3de2fd7e84a133
        def get_inputs(self):
            return [
                paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1fe00bc5e235247d2fafbc8b2785f387(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.95, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a4e9474c6d90fc7efdebe4f36549e308(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fe00bc5e235247d2fafbc8b2785f387
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.385583758354187]]], [[[0.49460527300834656]]], [[[0.7205386161804199]]], [[[0.3620162904262543]]], [[[0.09405810385942459]]], [[[0.44892749190330505]]], [[[0.5897328853607178]]], [[[0.308209091424942]]], [[[0.2562907338142395]]], [[[0.6189999580383301]]], [[[0.08149252086877823]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    
    class PrimitiveOp_1aeef020f34d701f6589063868529179(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0526299476623535
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5542695a28f6196aebaeb4f236be8bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1aeef020f34d701f6589063868529179
        def get_inputs(self):
            return [
                paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_509b572f4c5755b30fd32f5d1c1396bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 9.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ea5e9f3c5a375b66d2fe47f417335b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_509b572f4c5755b30fd32f5d1c1396bc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_085964f52f151f4e708fc875cb644db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e0161d80430d39567f3df0ae9557e75
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_840cd6a7c503b7c8704db6dcc405d4ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(93.6373062133789, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_7e22890478b7f9afe08973f8483f52d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(3.5218822956085205, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_a23bf077408167d166b66851d6237e8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ccd16204147ef4197a0cc221472bd9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbb4c282412f631502e11cfe5fc96829(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05b47e546439f0ebee1d780a39f2d444(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b99cf15c3b9dbc5c7d131d2bb684e501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7597459bbf09a180d67f766934ed1d63(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_66a41fbb56209b544ae111ef9a92b91d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7597459bbf09a180d67f766934ed1d63
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5caf7f1106ed26aca6b20533857f76c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ffeca38259a6dc86a396577b7ab0b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5caf7f1106ed26aca6b20533857f76c0
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_106a451e6fdd81474e228a016b40c261(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55faaf4a1ae996948fc7d466185a5173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_106a451e6fdd81474e228a016b40c261
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07670ea66b4132474d4c2096c6d5d815(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ddca9322b5abca156cba14772675f421(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07670ea66b4132474d4c2096c6d5d815
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_31e93dfd8d1722800631487b3862bb5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07ea01a2d68ffbc8b50c269bda248cce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31e93dfd8d1722800631487b3862bb5c
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    
    class PrimitiveOp_eb21d77a110dd5ae0f4315b857b9b0ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a2fad3d7dae8c19d2390d7a4b01bc13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb21d77a110dd5ae0f4315b857b9b0ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_69fe73e127b510e2606bd383627bc0ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4e48da2adc588a1269b95dbc846bfe83(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ef01d08afa2b13447ecbace3277bd5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e48da2adc588a1269b95dbc846bfe83
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ffeca38259a6dc86a396577b7ab0b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5caf7f1106ed26aca6b20533857f76c0
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3586bce3af5e611308cb6cc6147c3b79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d32c31f4c3e05eaa2480fa17c59b4236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3586bce3af5e611308cb6cc6147c3b79
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddca9322b5abca156cba14772675f421(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07670ea66b4132474d4c2096c6d5d815
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_11aeeba5f2fe7adb74d615b8b421bdd3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56213328bb637a39089d87cb85643301(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11aeeba5f2fe7adb74d615b8b421bdd3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_2822ae33052697cd494142775be73ce8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb21d77a110dd5ae0f4315b857b9b0ed
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_d6b43dab4e00b1116c970b2abdb6678d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b4d51baee2ccad1c2b2ca32a8d2f862
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_0bf875a78e6910f098a263027d20c92c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e0161d80430d39567f3df0ae9557e75
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_023b41fbea0f0970107b92662c6949dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(134.2923126220703, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_b853104c9908e51801fbe2ee1eea61fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(3.1249592304229736, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_d2b77b1ebbcabe3bf8c49847a4e955d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.975, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe6efce0ee2b60153ef12b7edc8153f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2b77b1ebbcabe3bf8c49847a4e955d0
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3fd1941d1ac5468465f30b9b0b28c616(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0256400108337402
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f647e677c6a0cff662e4bf5c691325f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd1941d1ac5468465f30b9b0b28c616
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f098771130c44ca9e8d71d93b59fcdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e0f73bc811f6da601617c7cabea4fa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94f47fa76ac4b7e8b8b5ced0583eb83d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d77de2ae3aa17134e140200ce2fc0dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c75b3897e203df41ee2d6fe3327cfa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13e38044ea260212bae10eaa0f09f4ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4dcfc0b40d70ea2fcf90908d014d35e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.01010000705719
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aaeb186d301592132c73829c515c3bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcfc0b40d70ea2fcf90908d014d35e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2f13cb24b61e6fb58a98a328443c088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.006484865676611662], [0.05156746506690979], [-0.11047624051570892], [0.0472310408949852], [-0.024203266948461533], [0.03868870064616203], [-0.07902292162179947], [0.15735942125320435], [0.08276878297328949]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_f40abf651a3f945b694579e1b3b47c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.02537374570965767], [0.031217578798532486], [-0.04251473769545555], [0.022202739492058754], [0.11096224188804626], [0.0752519965171814], [0.03307155892252922], [0.06225350871682167], [0.09541821479797363]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_ff3f833396ee0db603309ceb623228d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8bf5193616150b15877fe8fe519cfc74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.7444261312484741], [0.6518726348876953], [1.5985398292541504], [1.1272618770599365], [-1.2181216478347778], [-0.4858780801296234], [-3.3894524574279785], [1.5277197360992432], [-0.13256831467151642]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_84155f50e6ea448825382c489dc3908c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f107338497b2a385f99d224be9b4bdd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.7444261312484741], [0.3481273651123047], [-0.5985398292541504], [-0.12726187705993652], [2.2181215286254883], [1.4858781099319458], [4.3894524574279785], [-0.5277197360992432], [1.132568359375]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_7dfcdf90270b648ad7ed77006a478dc4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e4900a3744f5135f8eb0c44d967fda77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dfcdf90270b648ad7ed77006a478dc4
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_736da84668d96aa70c9133b4d649ebfc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.09090910106897354
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e30385e43a24ac70da5286de189f5c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_736da84668d96aa70c9133b4d649ebfc
        def get_inputs(self):
            return [
                paddle.to_tensor(11646.103515625, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_93053cae0cad912fd27e70989afdc7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(1058.73681640625, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_949835718ed2787242a502043b979e74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06662624329328537], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ca0351031241df8678931bd920e8bbfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[1.2875372171401978]], [[1.33564293384552]], [[1.5743564367294312]], [[1.2732023000717163]], [[1.0654644966125488]], [[1.3299976587295532]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_3c83bcf408a955b2307ec379fa2c3c34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[1.5494780540466309]], [[1.1539307832717896]], [[1.102842092514038]], [[1.6390326023101807]], [[1.4597944021224976]], [[1.2427841424942017]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_526f4daf1bd3e86aad132b029f29fe4d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 128.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c64fdc0aa24cbe3f4898fbbbb82f47c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_526f4daf1bd3e86aad132b029f29fe4d
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_27c8e5c251f3039f8cc47c6e2e1003dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b6d76d6a7045561a780a9d4994a07de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27c8e5c251f3039f8cc47c6e2e1003dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c64fdc0aa24cbe3f4898fbbbb82f47c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_526f4daf1bd3e86aad132b029f29fe4d
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4148687ca6a6079c52d690cd4f1349c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09ae04244d10ffedb632df870a3aa91
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a27809c46b15b5deef0177866b2d49aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a27809c46b15b5deef0177866b2d49aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d744264d5c731384cfe69855ce49d0f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de03cd2a3cc4000644626597bccdea9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_843acfcb95e29541d7f74529dbdd4e76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_987b6d8b94a103351d48cff0b0d3d4ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d0391cb28768854d6fb188b135d00e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_4d0391cb28768854d6fb188b135d00e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_437ce0b604c4036dc22b76e06cd42165(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.111109972000122
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_064b9f4a40ad7b9ff793fafae77706f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_437ce0b604c4036dc22b76e06cd42165
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d9881eaf7c4176d56b57f61446a516aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.5
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eac8333cb66bbf6c45b8eda4ad2834ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9881eaf7c4176d56b57f61446a516aa
        def get_inputs(self):
            return [
                paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae58cd584d74ed5bd97e1ebdcf441328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9881eaf7c4176d56b57f61446a516aa
        def get_inputs(self):
            return [
                paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f18644a44e5ad48586f74a1eaab066c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.8375, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ad1f8b135389f42b68dede347281224(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f18644a44e5ad48586f74a1eaab066c6
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_78f66699062fd7c835e7a5fe4c6ffa05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.1940300464630127
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae97f7f8f3d8ac0c53d9761fa0abe2ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78f66699062fd7c835e7a5fe4c6ffa05
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5799b3e5bdc91e103ab5ccf6c2a4a526(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(100.98829650878906, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_9147dd13085a7021fff0666d2dd43fca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(5.6871795654296875, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_8792d9f54d768e72f8a695c5111e113c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_cd7bfdc9f893afe79177c02ee3cd6e7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd3a2c47190f0a9bdb82b03c21b82c5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(6.652104377746582, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_dca89b17e318a5f2be49caceda43812e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.111109972000122
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12a6364b74610175276733b82e31289e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dca89b17e318a5f2be49caceda43812e
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_749255f34ad708b5931aef847329741e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab5e50e05e2bd5d4b80763cb099d1a70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1501445472240448], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1df11d93a321453606bb3016838d24e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.36977577209472656], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_591ac0165c4965ab780ad979db2b141e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.5
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d538cdd7e2a1d10c6a4cfeebda9accdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.8370978832244873], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b279953df0230905e9e8f3f20d7cd258(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f036ad84c5f9f5665d0718ee2e67c965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83749efa7e14fbd125be8f3896095e36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a46e04f7f15f6b9e3284f37b109fbf7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(145.07211303710938, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_a347b03d29be4648f4b3a3cc459ef848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(89.48736572265625, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_f0503b239de7f6782926f7cc996f5119(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b50506ac31e77f5dd0ae23a2db93a36f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0503b239de7f6782926f7cc996f5119
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaeb186d301592132c73829c515c3bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcfc0b40d70ea2fcf90908d014d35e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6051304680960668197f94f80d54dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6863819360733032, 0.5550834536552429, 0.5681371092796326, 0.25456511974334717, 0.5099325180053711, 0.24781492352485657], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f9060565171d68b8f545e1451b2dcdd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5956051349639893, 0.4590386152267456, 0.32019469141960144, 0.5512910485267639, 0.6469229459762573, 0.07742799818515778], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_dff0c5e215553eebf41aeeacc71b8c5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.720000684261322, 0.8646494746208191, 0.589462161064148, 0.8502786755561829, 0.7028647661209106, 0.18222540616989136], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_5c5cdb406e3d5285bfe4344868a1fc65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.7023019790649414, 0.4557735323905945, 0.7801243662834167, 0.10672207176685333, 0.6334706544876099, 0.7819021940231323], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_6fe4e3afb46f56e88957b595216df89d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a159cd68f69bab95a43ca71d0cd40009(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe4e3afb46f56e88957b595216df89d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02505781129002571, 0.017090944573283195, 0.004393713548779488, -0.0008124950109049678, 0.017373912036418915, 0.0037726969458162785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a8ba0e0e2bb02b2dc6fc932275eb446a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe4e3afb46f56e88957b595216df89d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0031286091543734074, 0.023960445076227188, 0.052997514605522156, 0.13812905550003052, 0.009350953623652458, 0.12514647841453552], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_474a1f0b8a9844c5cf5ff92c3335929d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe4e3afb46f56e88957b595216df89d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.05580516159534454, 0.21465319395065308, 0.09616874158382416, 0.3242293894290924, 0.04169569909572601, 0.15700317919254303], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_316277a8de572c09c037ea3280794036(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.4052850008010864
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9785daf15bd941ce250d01a788dec4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316277a8de572c09c037ea3280794036
        def get_inputs(self):
            return [
                paddle.to_tensor([0.08393514156341553, 0.7033591270446777, 2.8352646827697754, 0.059799134731292725, -0.1445942521095276, 0.6819069981575012], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_1549ce8f0624f6cad262f5c25e7762ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83f63c0e7d95d650295780b6518b04dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1549ce8f0624f6cad262f5c25e7762ec
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, -0.0, 0.0, -0.0, -0.0, -0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b165594ee9e94c49a2a2236a318de00c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe4e3afb46f56e88957b595216df89d
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0028553009033203, 1.200500249862671, 4.257975101470947, 1.0014492273330688, 1.008473515510559, 1.188456416130066], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0df670ba014043394a21c4cc1521d16b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([1.056071162223816, 1.1451103687286377, 4.043917655944824, 1.4260247945785522, 1.2243378162384033, 1.826979160308838], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_2d6acd5787b0df385fda08f89c4aa7f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 10.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_606707021f290774b147ace02f7263b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6acd5787b0df385fda08f89c4aa7f9
        def get_inputs(self):
            return [
                paddle.to_tensor(1.7870736122131348, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_b12a5cfa30751f82821880bb05858101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f3c6e95d028939d79c5d501dd916485
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f839e2744303ed33987bfcdac325b3e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f839e2744303ed33987bfcdac325b3e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd5fc252422b4b54a20d0f39f27d1d72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92c26c14924d64222f27bd59a59dcde3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9564d577bee897979927ae8e07ab7edb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_bb5cb775e653b354620968d9ff6d204d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1d70bc2b9725485f4ff039d6666c836a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49e6e040ab216d7c75279d0fc5d67181(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d70bc2b9725485f4ff039d6666c836a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_49e6e040ab216d7c75279d0fc5d67181(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d70bc2b9725485f4ff039d6666c836a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_2f2870960429ee041fd73eea36873ef8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.111109972000122
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b539627a91885733b8220c2a68c773f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2870960429ee041fd73eea36873ef8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d203f32b799b86cbdf03e4e71348d46a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2870960429ee041fd73eea36873ef8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa21da8791218be36bc72cf1694c79a8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71e28710e1d3b7ccc89be8a328fb33f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa21da8791218be36bc72cf1694c79a8
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f629530e7c8c7238ee939e39913db82f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a29817cb75f80fb514b9719c9e153ba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f629530e7c8c7238ee939e39913db82f
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5151eb09cfadff24d7a6645fd114071c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ca21a3fa96e8b18be4da1294232d97e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.20113162696361542]], [[0.01791820302605629]], [[0.1196950152516365]], [[0.4878165125846863]], [[0.25780704617500305]], [[0.3775233328342438]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_71e7593ebe6b6282bf1d7d621cb0fb42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ca21a3fa96e8b18be4da1294232d97e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.0816684290766716]], [[0.17039605975151062]], [[0.4779324531555176]], [[0.10027485340833664]], [[0.247049942612648]], [[0.3199577331542969]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_1b636221dac85b2d7036fa05c476c18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90f0c9904512f2ccfef207ab9f13db8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.19105783104896545]], [[0.07815577834844589]], [[0.2450648546218872]], [[0.4223167300224304]], [[0.21929742395877838]], [[0.10183952748775482]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_84b870f3a1dd9830e6a8c3fc64002a0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90f0c9904512f2ccfef207ab9f13db8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.18758375942707062]], [[0.1623629331588745]], [[0.4591658115386963]], [[0.12450161576271057]], [[0.3159920573234558]], [[0.3298834562301636]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_3d0cba48db4657988deadfabf5378a7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c3c826b3f48a267c683069314719058(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab4b2160c98bbd1d475fef7812af39c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.43301984667778015], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_f8638f2c96ca58af820de85e3e969f4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2.5
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_85d599641763cfce68cc781bffdcafc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8638f2c96ca58af820de85e3e969f4c
        def get_inputs(self):
            return [
                paddle.to_tensor([0.04718353599309921], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_eb7d2237f86ea801a935b51885b1a37e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.05000000074505806
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49a2488b09af6a37d3e47af64155f067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb7d2237f86ea801a935b51885b1a37e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.38502153754234314], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.111109972000122
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34880938add65f0ef6154becaed8dcab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1c963e96dbb01500ba9da2f32b09855f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59913bc46b52005edf5d14dcc3960521(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c963e96dbb01500ba9da2f32b09855f
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b85058c99bc9962484b4d88881b51aee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_edd048bba6184842d5e9edd00365cbfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b85058c99bc9962484b4d88881b51aee
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fc087c62eff9798ca74becf43d05d40(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d09a22999aa11cbeb901feb919c34fa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc087c62eff9798ca74becf43d05d40
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4dbc83d76caa79e9a733192fa89918ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8566d017edccc32845f0088fa11cd27f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dbc83d76caa79e9a733192fa89918ae
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_565b8afedd613b53c0ac2189bc0fbdec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c401dc4d420659b41454b8c662024219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_565b8afedd613b53c0ac2189bc0fbdec
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            ]


    
    class PrimitiveOp_90964e0046798c999cb3be50c357c3c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bcf7c3da776c5da19ee1515995308c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90964e0046798c999cb3be50c357c3c7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_0fee9067adfbd23aa5dc00efae46f97b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_83148f554ecdb075e27426996403a872(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1eed0321bb8cd1794f4850d046244b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83148f554ecdb075e27426996403a872
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_edd048bba6184842d5e9edd00365cbfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b85058c99bc9962484b4d88881b51aee
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_404aaa16739dcc780ddc0b2769918559(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa59a91a09cdb96951858b5fb6125baf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_404aaa16739dcc780ddc0b2769918559
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8566d017edccc32845f0088fa11cd27f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dbc83d76caa79e9a733192fa89918ae
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d62d62fb5d4070e8b1549e5c778fc958(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_50713892c5cc66256226f8667f7c8632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d62d62fb5d4070e8b1549e5c778fc958
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_561e9bf8de49c0f5806b429025ba5f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90964e0046798c999cb3be50c357c3c7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_7b5922f9467b5aca67fe1634830e3dbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_b5d95d6d25e7b3a417913335d8b159c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.to_tensor([2.042098045349121, 2.1673107147216797, 2.046053409576416, 2.187094211578369, 2.039552688598633, 2.066070079803467, 1.9485445022583008, 2.015756368637085, 2.2058639526367188, 2.0801162719726562, 2.115180492401123, 2.0773661136627197, 2.1335859298706055, 2.024040937423706, 2.078754186630249, 2.0380187034606934, 2.1577341556549072, 2.129593849182129, 2.2176289558410645, 1.9216270446777344, 1.9373674392700195, 2.045713186264038, 2.1750686168670654, 1.9353487491607666], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_b4d7e40c8604b1ef62a31a34efe8156a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(3.5862324237823486, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_f38227a1898719c5bcf2b8f31a7a8345(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ce35da88a388e99ee582936ac0545dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f38227a1898719c5bcf2b8f31a7a8345
        def get_inputs(self):
            return [
                paddle.to_tensor([0.43497511744499207], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_cee48b2965d8f3c34de2071916873bff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_359e3739ff3cfc608cbdaeed3109791b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cee48b2965d8f3c34de2071916873bff
        def get_inputs(self):
            return [
                paddle.to_tensor([0.38216280209126474], dtype='float64').reshape([1]),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2d50b636a4d28ba527a744dcf9b95c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2fddbb6a528a7c93b3866f78a751890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c7fa08fa11e1334469e857235f9222e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbb4c282412f631502e11cfe5fc96829(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03232d5fe76ea29afc881e2d7db44d6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4365cf6d986d3c98556d5c5f8453f481(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b26637affae4ef9eda84978f9be5ed54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b26637affae4ef9eda84978f9be5ed54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b97e46a3a8ed393e2b264b742d8dce3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a30276f23ea722e556bf9bcf579c6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de6b7057663c4741880980e1568af029(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ba0340dccde81085e31e104af3b819b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0666ea4df63bd40204a0202092dddfab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_0666ea4df63bd40204a0202092dddfab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_d8258c884d6dc84d25317fa2600fe2bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa02099785e654681a3d84bf08b036d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8258c884d6dc84d25317fa2600fe2bf
        def get_inputs(self):
            return [
                paddle.to_tensor([[9]], dtype='int64').reshape([1, 1]),
            ]


    class TestPrimitiveOp_a36f9899cc667d681baa2496203a857a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9678ff1abf4f89a99f77d3fcdb098d64
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2439064234495163], [0.24398155510425568]]], dtype='float32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_ac86be1b700219c9f93930a997de2133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3cc1ea34e5aee9740b82608264458c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf82d03b1a1cd742f431c5af366487eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f169ea5e8273c1d9ca0ee0cfccea73ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_666850d03d7b3924433f21bd579c13dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.to_tensor([2.025592088699341, 2.0849061012268066, 1.9092397689819336, 2.135376214981079], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_9a91a1c7f34cacacc4a18c61f1a4cf8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(0.604914128780365, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_8b114d0394649af575014139ca430f4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db588caa2c2dbfa7924a56d53255ec1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_221acb9c76b4cf77ab2398883b86cbc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_800a07d97880f0d8fc8f21655a5a8ad8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(147.2303466796875, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_2165ddd655d4997a3e97ffb43d503dba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(4.197933197021484, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_f845b8667f7c2294e275b5e5588f9163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(137.94223022460938, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_f03eb5295661b2559dfe7c4d21f870b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(4.009830474853516, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_f832bd1f680bda98abbc81ae5aa6c11d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.25
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b739e2fd45db4e037fe2d33d322aec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f832bd1f680bda98abbc81ae5aa6c11d
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c86b74ac0c50a36ceb191dd55507180b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89ff448e17a93b652015c4ba4da227fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c86b74ac0c50a36ceb191dd55507180b
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cb0cb18ab122b9b9357ee65469ae8a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(141.23519897460938, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_aa3d6e27538f3df0742be105f0ba2511(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(8.003020286560059, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_293c5db58182917a5469932a7478cb54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01941882260143757]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_83518554c2ca8becfd26c0d3a152511f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.025589246302843094]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_993def4b3035112dc237d159a5e01a36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.24113346636295319]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c9e8997baabc3bbc0ac5463364ca9226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.241133451461792]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1489de721f83eaa80867d29398aff6ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.043292563408613205], [-0.02788546308875084], [0.05995785444974899], [0.03143922984600067], [-0.0009129986283369362], [-0.034111231565475464]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_3357c9602fa0b501674421b2cdaa9ab1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.002421251032501459], [0.012762386351823807], [0.12483116239309311], [0.007755537051707506], [0.057623084634542465], [0.02943781390786171]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4396919826e7af5f41c9cbb48299d56d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[-18.880247116088867], [-3.1849725246429443], [-0.5196884274482727], [3.053778648376465], [-1.0158443450927734], [-2.1587555408477783]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_61cb47530b931380be1fbb1775d7cb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[19.880247116088867], [4.184972763061523], [1.519688367843628], [-2.053778648376465], [2.0158443450927734], [3.1587555408477783]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_200a1ef9972c09dfa37c224eb01607d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2b77b1ebbcabe3bf8c49847a4e955d0
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.997045636177063]]], [[[0.2572757303714752]]], [[[0.027550404891371727]]], [[[0.17211119830608368]]], [[[0.1591709554195404]]], [[[0.3672061860561371]]], [[[0.9256317615509033]]], [[[0.3660320043563843]]], [[[0.07933235168457031]]], [[[0.8728136420249939]]], [[[0.6442326903343201]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_ba198c3687d30352279f1abebd0f6b64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd1941d1ac5468465f30b9b0b28c616
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2fcf03ad7de12d88b4d89b9867a99709(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17fe8002986ab7a40a1faa05d7d7d1b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fcf03ad7de12d88b4d89b9867a99709
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype='float32').reshape([14]),
            ]


    
    class PrimitiveOp_e1de710d7ff05bbe2cbe55dd4294173f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82d969f5be0cf59de78c90178c1a62d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1de710d7ff05bbe2cbe55dd4294173f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], dtype='float32').reshape([14]),
            ]


    
    class PrimitiveOp_915747a65cf8b0e464937ab3016dc89a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -80, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8978699efbe8d83930848171ba741d3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915747a65cf8b0e464937ab3016dc89a
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8978699efbe8d83930848171ba741d3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915747a65cf8b0e464937ab3016dc89a
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7aa2c3f1316bc0bba57a186aeb695722(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 80, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cca331d121ee034a48e76c9efb2aa3d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aa2c3f1316bc0bba57a186aeb695722
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cca331d121ee034a48e76c9efb2aa3d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aa2c3f1316bc0bba57a186aeb695722
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19d4fccd7ad9a54a52215dfa454acd6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0103deae7d2c5fb617bb2cc7bb8e64bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19d4fccd7ad9a54a52215dfa454acd6f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], dtype='float32').reshape([28]),
            ]


    
    class PrimitiveOp_c2b2f8e33049cf4b26a17146936e0697(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de582cfbaf04eb2e96a63d3c1cb64891(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2b2f8e33049cf4b26a17146936e0697
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5], dtype='float32').reshape([28]),
            ]


    
    class PrimitiveOp_c64c07c171e995affa1ee929c15ac0ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -40, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd798ef51bb13c61f2390401d30d1434(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64c07c171e995affa1ee929c15ac0ed
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd798ef51bb13c61f2390401d30d1434(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64c07c171e995affa1ee929c15ac0ed
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1fea8ba2a2904848753141307bda3378(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 40, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_614cc37937d7b0faf3b5ea1327d003ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fea8ba2a2904848753141307bda3378
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_614cc37937d7b0faf3b5ea1327d003ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fea8ba2a2904848753141307bda3378
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b927c01a3e6f77c3589794d98a030c5b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_804d69e48d3bd16d7a6b0bb705462eda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b927c01a3e6f77c3589794d98a030c5b
        def get_inputs(self):
            return [
                paddle.uniform([56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a32ae5328a2dbe1e9814bf682bcd722(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9bac8c420f210835b861ffda2afd06b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a32ae5328a2dbe1e9814bf682bcd722
        def get_inputs(self):
            return [
                paddle.uniform([56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c965cdd79391b3f2e3bff1fd18ed7d42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -20, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca7ecd2d8c1174ba25abd12bdda24acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c965cdd79391b3f2e3bff1fd18ed7d42
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca7ecd2d8c1174ba25abd12bdda24acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c965cdd79391b3f2e3bff1fd18ed7d42
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b66597dfb7d7e680d7ce4a30ab45f72f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 20, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_660a2aa9e21df3929e70c2b8867152a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66597dfb7d7e680d7ce4a30ab45f72f
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_660a2aa9e21df3929e70c2b8867152a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66597dfb7d7e680d7ce4a30ab45f72f
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1f15e8c5a79999c7fee97caf0a50c6a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_710140a53524499dd92535b65b67715e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f15e8c5a79999c7fee97caf0a50c6a7
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_fd5860b084afe67d76098e7c33aae991(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f15e8c5a79999c7fee97caf0a50c6a7
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c3a303adac7bf1d7ec02aea4245dff86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f15e8c5a79999c7fee97caf0a50c6a7
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c3a303adac7bf1d7ec02aea4245dff86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f15e8c5a79999c7fee97caf0a50c6a7
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5ad29cbef7a64ba010c325406cc3d547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.to_tensor([[3]], dtype='int64').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c17f9704fd00dcdb89ad621f2fdaa209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9678ff1abf4f89a99f77d3fcdb098d64
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.24656285345554352]]], dtype='float32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_296500bac3856aa4e643062804500fc0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2100d29e2884808ef04e52fad8de1cbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_296500bac3856aa4e643062804500fc0
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3b67d5d0024e106348f03adb0b0fefbb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0015625000232830644
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_13220a9db0d64bb56773b8ae08ec06d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b67d5d0024e106348f03adb0b0fefbb
        def get_inputs(self):
            return [
                paddle.to_tensor(4.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c7a9d73f3b4aa24d1e1c8494d3f70279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b67d5d0024e106348f03adb0b0fefbb
        def get_inputs(self):
            return [
                paddle.to_tensor(7.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_d6ea5888c6b8aa24f36639a8f2ae3496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0503b239de7f6782926f7cc996f5119
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aeeea9bb8d1620a3b95e340e3159f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(157.97816467285156, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_859f5041f1d0334c81bcbcb9666edf7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(2.6843674182891846, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_078288c8912e0c106f1647f33402967b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a4d85341388a9260a0b60a8c1708b91b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_078288c8912e0c106f1647f33402967b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a23bf077408167d166b66851d6237e8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86dda8233f6b093caded9f5a91499128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd34e452c6410030bf15f3fe88bf439c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_afba1cf8682ceb676f18753ab6359fcf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 197, 197], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21fc1d596052e46a1e6bf7840cf6490f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afba1cf8682ceb676f18753ab6359fcf
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5f651f18d1ad632440ac56f1b6d5546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df807387f53add025873a3d542a1f741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f40aabca31372e0ab7df472beebe5cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(166.91445922851562, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_4fff3baf264afbc28c1858151f1ee92f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(60.17794418334961, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_d682a78b6ec1e9916db305478deec5ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cb59e5ade6ae98e61fa1849d42216ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d682a78b6ec1e9916db305478deec5ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_108f435a07debe4b81f61beb55787a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7dc7b213fba3bb402c943a9795dfc78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8e99bd59d381ab2b73e1a2b7a34a7e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8e99bd59d381ab2b73e1a2b7a34a7e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_921b98b8f4fe35a308aa9ab25025eacf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a522821de87ff833f0df141c957d338(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e0295115e0d935aac6f9103b6217672(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_60c06d02c97b33f1a3ef0d977722c870(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7e24efd1e2e5f3366dc12c5818e958b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_e7e24efd1e2e5f3366dc12c5818e958b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_93c4c9596f238aeca7d4da622b8ee90a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_078288c8912e0c106f1647f33402967b
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ea0a49a469227c5640cb9f6ba6a9ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(146.36666870117188, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_4c3f3dc388e3da10e630c0069d272b29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(4.160505294799805, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_edec9a221140ca45a2ae7ea0bbfccc78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49a7007c13884472499cfca05596b46f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            ]


    
    class PrimitiveOp_e1e86eb74d58de537afad436f3e3b79c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ea113891c580a202364d8a7406b56b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1e86eb74d58de537afad436f3e3b79c
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_98f8cf3be25d5a138816e9ce21da2600(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.02942965365946293], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d36c552518cebfad9bb6e4ecdd6a7d3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09304554015398026], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ef55d66672689ba2c977b9eacefba8ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.040439870208501816], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4258718d31956e6f856504486dfe3cee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2780453562736511], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_31b753b6b66b21103fc004b2dbff3c12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4882940351963043], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cb1be7bb9e895791474bdcceedefa018(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5158790946006775], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_eadb53d7ad56fa5a7bf1026e2ab83832(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.48973387479782104], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_817600b32f35588ff0d793d304e17b2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5623858571052551], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_022bb82596ecc6403ac0926aee9d6604(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.49126216769218445], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cd77ceb7e92d1d908edd935a07669af1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5432940125465393], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_545b82b2902c60d64a6e1d026d7ba8dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.03385039418935776], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_eaf5459e847b0bf4a11eb33355d5796f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.10091172158718109], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f96216ccd879f904b070de4c3e8804d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4407913386821747], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_eb0169b8c6a873c138a5aa99c0dd7d43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6364049911499023], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1d0503ce577b5eb0a6305807ea375543(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3561382293701172], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f24484d6d3c3c650e90231c91f028796(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6890357136726379], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4505ad39c766338420a0b4d8f36220f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_749255f34ad708b5931aef847329741e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2267407923936844], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ab9ae0618b7d916495cbb98cd165789f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.39462366700172424], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a4c88302e5655fa464e72b2551f19039(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0655159056186676], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6235d5c671f1b607153818ddad8872d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.24741116166114807], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ee6414569488a6ea052788e196c8f8cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.04495077580213547], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5fb78cb99a7e97df042dfd8587ff36c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.44348499178886414], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_00b16b6faca1cabb81f8a0cf93ab91a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_138878cdd8acdc0bd5c2e77cae0a6bf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32b29d21d9e6de1fa60c54d37a173a3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71d8c4dad6f9192521b2cb72e340e2d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1ae0a58781cc6e48966d2b2211f1ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e37fe63a1fa196411593939cc1034fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_098e208d9e30370a6e0e8ee20e268ddd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e272b8d08ce02cdf00d45aae76767b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098e208d9e30370a6e0e8ee20e268ddd
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d50cf64dfe0cd8bb1c53f4dfb3b63241(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27c8e5c251f3039f8cc47c6e2e1003dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e272b8d08ce02cdf00d45aae76767b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098e208d9e30370a6e0e8ee20e268ddd
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d6769f7ff8948de33e60d7106e847c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d6769f7ff8948de33e60d7106e847c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd119f4480dc41da7c05e1738be3f506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb1b14ffdfa326e57a79beb6e74ad9c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c2b79f92d1f54d7d9c27b76d2535d5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_a8dd47eb09a9b3bc214182be05dc1f1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32d694b3ec979962eb0ab4003f0ff92f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_32d694b3ec979962eb0ab4003f0ff92f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_91a8a8669394de3bdcc4356cea30c269(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f3c6e95d028939d79c5d501dd916485
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd4d18e1732c0f73b901051ddd948382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(100.44140625, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_75d4e8a4751898ff9479abbf07d25dd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(308.20684814453125, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_3161a722c3ee6925e2035fbeccd9c14b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3161a722c3ee6925e2035fbeccd9c14b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93140650bac0e405ea2d44db32c22180(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96eb15100430e64bc3bb20208edc50cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2232a2fc7e2bca4250fbd135c199ac4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ba960798d2cc95708763939539f8b2b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d53bcea9c00496e14a8b7e69e3cf438(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_2d53bcea9c00496e14a8b7e69e3cf438(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_2393f317c68d0da16ade852ffd77c8dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f3ae3a6873898a8422eed7dd31b4e65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2393f317c68d0da16ade852ffd77c8dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1951ec678b4be08e351e4014c7a49573(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -50.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c5e7ec945530c6c16095fecf6758605d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1951ec678b4be08e351e4014c7a49573
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a7c56a0f0e00e9f7063bd373353711e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e13bee2111d98ca1ac4a0aabcdd7bfb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c56a0f0e00e9f7063bd373353711e1
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fa49a9b27290490705804df84e7cb21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(117.01022338867188, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_972d84435fd8d54dbbda0767ea88aa47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(3.625927686691284, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_a286a77b4d7270c6e75c60d36e679200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dfcdf90270b648ad7ed77006a478dc4
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc325f4cfeedceef383b2dbcff0f069a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77d7bb0d6f1abddb984219d02b200185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f414651b75fb4117f6952df72c87df0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5e64b00a3005f9034432383dd1527b16(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.75
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_523fe1f070650c0ef3e771cfcb8a1ab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e64b00a3005f9034432383dd1527b16
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ed23f4a32f45e96e21ada4cc3bd7c0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aef523ee20fb4a9317dc39ad7a3c0ac4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1e-08, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0827fd623f47c0436c93667413640909(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef523ee20fb4a9317dc39ad7a3c0ac4
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e0016f70395d41df37b8d78530c46f2c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6423ce9964b21a8b724b2a8b61152c69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0016f70395d41df37b8d78530c46f2c
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5aabf326668c5b3605786df51d820473(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.25
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64fe1e46a1afa270c46a4f1be2b3c9bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5aabf326668c5b3605786df51d820473
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0827fd623f47c0436c93667413640909(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef523ee20fb4a9317dc39ad7a3c0ac4
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6423ce9964b21a8b724b2a8b61152c69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0016f70395d41df37b8d78530c46f2c
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e0f73bc811f6da601617c7cabea4fa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bc6d0c9bbbc804d4aa0b480acdb87b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf0745789044cab6ae4fe288e3dd8a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_894cfb53ee6960b4b1e2daeb3e64586c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b3287f3afbb979c334c2875ecd9ebbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_894cfb53ee6960b4b1e2daeb3e64586c
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a76e4ee498d1c8bd43508b3185258d01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a81c78870c10cfff59eeb9baac7a3de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a76e4ee498d1c8bd43508b3185258d01
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d28a7265fc6d267bdf50aba4989249a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2846dc9b2354bfee4d9421705e21b634(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d28a7265fc6d267bdf50aba4989249a
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_359cff6254df64ce32f6d33a5c342076(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_35e4a0f994fa629501d0e6ef1b0f3e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_359cff6254df64ce32f6d33a5c342076
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ec7f3b61c72b678c3e5db82fe637a86f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3aa408cb6885ee3888bf9b9f8c0482f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec7f3b61c72b678c3e5db82fe637a86f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            ]


    
    class PrimitiveOp_5ca00aaac6497f521742b2fb5ae464ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a70fa84516bf8c2e352e5c136f2cabc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ca00aaac6497f521742b2fb5ae464ce
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_fa5cb3d1462e8ef8406db9d853234d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6ebf84253d5b9a906d61a53d8b7b098e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80c0d79732b83df434a453fff6167363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebf84253d5b9a906d61a53d8b7b098e
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a81c78870c10cfff59eeb9baac7a3de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a76e4ee498d1c8bd43508b3185258d01
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_907943072b47183773cb72c53df38a8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_987b9d4a26cacbc3032609e485cab7ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_907943072b47183773cb72c53df38a8a
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35e4a0f994fa629501d0e6ef1b0f3e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_359cff6254df64ce32f6d33a5c342076
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e74fde2aec73258fa74dc6a1ce25bd5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af922b3d5e571d0ebb5c11a20009837e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e74fde2aec73258fa74dc6a1ce25bd5f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_5fe36124316d05abec9bf475f7a7559d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ca00aaac6497f521742b2fb5ae464ce
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_064b9f4a40ad7b9ff793fafae77706f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_437ce0b604c4036dc22b76e06cd42165
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_064b9f4a40ad7b9ff793fafae77706f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_437ce0b604c4036dc22b76e06cd42165
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_064b9f4a40ad7b9ff793fafae77706f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_437ce0b604c4036dc22b76e06cd42165
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49b8ddde455e83b8e44d75b4a3f6b054(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.111109972000122
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d5dc6c04b4aee51b5e6733b4c83f41e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b8ddde455e83b8e44d75b4a3f6b054
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_988a57555ebb8cb66c5d2c2f17c98827(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e64b00a3005f9034432383dd1527b16
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a95202ee25bfb76cae8c8c55a405c6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bac737f8d8c92ae437b8efe4ee91249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef523ee20fb4a9317dc39ad7a3c0ac4
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12353d258fd2407afb7dc07ae2b9b9db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0016f70395d41df37b8d78530c46f2c
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cf39d3e804fe509a40ffd63965c8bf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5aabf326668c5b3605786df51d820473
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bac737f8d8c92ae437b8efe4ee91249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef523ee20fb4a9317dc39ad7a3c0ac4
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12353d258fd2407afb7dc07ae2b9b9db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0016f70395d41df37b8d78530c46f2c
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ced98fa0373c97b728a251e39163ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01143362931907177], [0.00884125754237175], [0.06832607835531235], [0.0002330932766199112], [0.1191963478922844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_739f7a1bc28cb730062ccff8097ecf04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.004484008066356182], [-0.06190362200140953], [0.0013638997916132212], [0.05762871354818344], [0.01101756189018488]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_718e2f2a92283a02710e9e305169e329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.549868106842041], [-1.1428229808807373], [49.096107482910156], [-0.995955228805542], [9.818758964538574]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_bf322ec820d22a46d5988199cad5ac61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.549868106842041], [2.1428229808807373], [-48.096107482910156], [1.995955228805542], [-8.818758964538574]], dtype='float32').reshape([5, 1]),
            ]


    
    class PrimitiveOp_e019734bfe8190fe1e0c5c4755dda902(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1875b382f972e1c6a3076187bd19a564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e019734bfe8190fe1e0c5c4755dda902
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c36db3f71bff76a71e9bc8f72828d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    
    class PrimitiveOp_6122bf3d6e2f4a711d8afa66f5767cea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 64.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e448f86a00c299406166f171b57b90aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6122bf3d6e2f4a711d8afa66f5767cea
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_8c36db3f71bff76a71e9bc8f72828d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_e448f86a00c299406166f171b57b90aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6122bf3d6e2f4a711d8afa66f5767cea
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_c799d05b1bcd72cb853a0f3157fba972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            ]


    
    class PrimitiveOp_50be9c14adb924915c0af546c3e4ba22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 128.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_85ad6e7c6f19dd9106a1b899491efbf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50be9c14adb924915c0af546c3e4ba22
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            ]


    class TestPrimitiveOp_c799d05b1bcd72cb853a0f3157fba972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            ]


    class TestPrimitiveOp_85ad6e7c6f19dd9106a1b899491efbf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50be9c14adb924915c0af546c3e4ba22
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            ]


    
    class PrimitiveOp_79c4c78bfee469cac4b0e04076c5c80a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_edc00fe0b55ce695a551be8b97a09010(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79c4c78bfee469cac4b0e04076c5c80a
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6c29a9e6295bd1a449287f33c9d27bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9a9d58c9e9d927236a0dc9d7e4328f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_437ce0b604c4036dc22b76e06cd42165
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9a9d58c9e9d927236a0dc9d7e4328f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_437ce0b604c4036dc22b76e06cd42165
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9a9d58c9e9d927236a0dc9d7e4328f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_437ce0b604c4036dc22b76e06cd42165
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a18357abfcf88372b8a14b032881405a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b8ddde455e83b8e44d75b4a3f6b054
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_30ac66b4d56b180837d255ad5cee31f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a11810c4341b7544bae144dcada1e1c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30ac66b4d56b180837d255ad5cee31f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e50d758eea80f51196d9f3a009bd8092(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27c8e5c251f3039f8cc47c6e2e1003dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a11810c4341b7544bae144dcada1e1c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30ac66b4d56b180837d255ad5cee31f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c52e26335e980699085d3cd57b0b68df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0226da7fc98fac03d1c38a35ccefb77d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d77de2ae3aa17134e140200ce2fc0dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3c483aea38d1f3cc131613ae6e04801(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20f9c9f87e5c615e695ae7294dd79440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0162d62abfc04ea2cbfc2b5fa71d287b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1cf3b90f39c790688e4db45490ac54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09171438962221146], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5a2bce73cf68c720378d2cb61c1a93be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8638f2c96ca58af820de85e3e969f4c
        def get_inputs(self):
            return [
                paddle.to_tensor([0.28419357538223267], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7a1e432fc3afbca9323bd804a1bbbf63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4586580693721771], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1f3a6ec86b02afa88c88436b2a6fe0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4a6f83e9516ddda198ae398cef845ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aea8f41ff9cc4f5b2e73ad5fa379d182(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_509b572f4c5755b30fd32f5d1c1396bc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_6b93ab9f8fc5280abe5d43202c1e470e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e0161d80430d39567f3df0ae9557e75
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_cae2f8602593e8e6bf3f800002278c57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cae2f8602593e8e6bf3f800002278c57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ffedf6d70d3e1db6f1d0ef905ffa832(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_339e562099ec71470313133120d6598d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29b837d3b5e1c2a118140d290b6b64b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_22fd74794cd9ee2602461c8220819692(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccc7d563a99d249efca43681f026c4bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ccc7d563a99d249efca43681f026c4bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_368b135bb11c06041b382f55737297b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_368b135bb11c06041b382f55737297b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f24101950e4142b67435b579500f26e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d62e73c1c75351abbf20b4cc7c97341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6160023027644b4ffeff4bdfe3bf5421(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_d92a13982fbee43aa60a3d32adec5796(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e22e3a0343a9fade0f73ed9ebf423800(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_e22e3a0343a9fade0f73ed9ebf423800(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_a84c2252f5d33348a799f5f53feb74a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a84c2252f5d33348a799f5f53feb74a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6040b3bab9f6e21f71f1d8d3d07992f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_607c3a194a576c599b3726135a6e576c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db15d906ae3e5c8b5cd05aecafc0a67b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_aa611e0cbf6b36922f2c2148239cb6a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad009066dcf233868d5989ca1dfa8b40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ad009066dcf233868d5989ca1dfa8b40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_391386eda6a4ece0a7097549a607e406(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 64.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc214f7df8a8972cb904fc85798ced6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_391386eda6a4ece0a7097549a607e406
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fa4801485598df67f8155ed8384ff94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27c8e5c251f3039f8cc47c6e2e1003dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc214f7df8a8972cb904fc85798ced6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_391386eda6a4ece0a7097549a607e406
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6d77b7fbee379eeeeeed5da420a3dd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8189a2e823a64d2a57abc080a51a5891(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3b02d563b7df7516340fb53900f818b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9ad36fcf8c721bac3a72d414a44fa728(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.925, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_689c053e610c4ccedfd699bec6ee75ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ad36fcf8c721bac3a72d414a44fa728
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.7168525457382202]]], [[[0.028730157762765884]]], [[[0.5656723976135254]]], [[[0.3021770119667053]]], [[[0.6223515868186951]]], [[[0.612608790397644]]], [[[0.6727516651153564]]], [[[0.5322846174240112]]], [[[0.7834969758987427]]], [[[0.6203384399414062]]], [[[0.3852967917919159]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    
    class PrimitiveOp_bb3e794075813e6b6ec0164accf28a6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0810799598693848
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b1fba21c676db60899e8dfa47a3bc94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb3e794075813e6b6ec0164accf28a6f
        def get_inputs(self):
            return [
                paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00b16b6faca1cabb81f8a0cf93ab91a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1db9ba60d01102bbb47f3b3afa313fb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_746f028dab8de0558eda19370a61358f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c69b0d18fd92caaea6b6a459d88af36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_746f028dab8de0558eda19370a61358f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5eaded081d34f0fe3e5980715cf1e07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_391dc50fca7183cf1b99d9a374ad0a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_377d22de9b54b06d64941bcea99568ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eb3699d623d3a4cc59f1f313a59b9155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_377d22de9b54b06d64941bcea99568ee
        def get_inputs(self):
            return [
                paddle.uniform([152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23c32a50c61107240f03d34691c02c9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d74a40f725eb1ddc993d7f4ec09fdce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23c32a50c61107240f03d34691c02c9e
        def get_inputs(self):
            return [
                paddle.uniform([152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0e991eec81b47e62369d6383ea45197(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1502a5d4ebfb46b87a1f969bf00f4e45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0e991eec81b47e62369d6383ea45197
        def get_inputs(self):
            return [
                paddle.uniform([100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_61ac9282f94048d42a7a3208d866123e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51f58523c2f4eabb6d015d86589406e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61ac9282f94048d42a7a3208d866123e
        def get_inputs(self):
            return [
                paddle.uniform([100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b356b690d864b23c3cd95a9b11105dfe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -32, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36d80ae10f9324ded0846c677b7f8199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b356b690d864b23c3cd95a9b11105dfe
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36d80ae10f9324ded0846c677b7f8199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b356b690d864b23c3cd95a9b11105dfe
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_43655bb8394d146c638dd3fd69adade3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 32, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d121fd951df9d25cb7241e7f6dbf0ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43655bb8394d146c638dd3fd69adade3
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d121fd951df9d25cb7241e7f6dbf0ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43655bb8394d146c638dd3fd69adade3
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_12d8b8e162077a41419ec4c6c49012a8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_151a5175675aea8ef1afc034ff86088a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d8b8e162077a41419ec4c6c49012a8
        def get_inputs(self):
            return [
                paddle.uniform([76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_148f62426f2d47971f6bd84b950ccf40(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6fed440f49812d50e3c7e14a4c0a02cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_148f62426f2d47971f6bd84b950ccf40
        def get_inputs(self):
            return [
                paddle.uniform([76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b7186417eca30c329e29fc4ffa2566d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82f3ad961ca837d840dba1e41223ac62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7186417eca30c329e29fc4ffa2566d8
        def get_inputs(self):
            return [
                paddle.uniform([50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa62fb12a18dcc247a4e6f55ed9af190(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a03ecdafbbb8a5ef156b1cb5bcd9ac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa62fb12a18dcc247a4e6f55ed9af190
        def get_inputs(self):
            return [
                paddle.uniform([50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c328e6c3b982313ffe5f7770de42375b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -64, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e3acf762aecddafe71446af5a66327a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c328e6c3b982313ffe5f7770de42375b
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e3acf762aecddafe71446af5a66327a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c328e6c3b982313ffe5f7770de42375b
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3ed8dee5747003374b08afca59779af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 64, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e00ee717218ebac21987b04b1a9cfc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3ed8dee5747003374b08afca59779af
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e00ee717218ebac21987b04b1a9cfc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3ed8dee5747003374b08afca59779af
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0cefa1aa9398f0c8fad95da88e50e0c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2696f8fd36ee2e5030562e67e6bfd43c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cefa1aa9398f0c8fad95da88e50e0c5
        def get_inputs(self):
            return [
                paddle.uniform([38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bbae4ebd7e12ad3ce0957819740acf57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be099cd34cfcff4f4a65cb30fd230b84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbae4ebd7e12ad3ce0957819740acf57
        def get_inputs(self):
            return [
                paddle.uniform([38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45a7260af0d9ce94e79594ee31a334dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd0859eabc7d5cd9c8f6dd180c94c37b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45a7260af0d9ce94e79594ee31a334dd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype='float32').reshape([25]),
            ]


    
    class PrimitiveOp_6439a746ff9f2f80cde36dcfba83de4d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd475f0b4aa841c4cd5646f22c116854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6439a746ff9f2f80cde36dcfba83de4d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5], dtype='float32').reshape([25]),
            ]


    
    class PrimitiveOp_0fc670b89ac304cf418e797c9ceb66ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -128, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b76c2f4ee9efc33836469294b86bb749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fc670b89ac304cf418e797c9ceb66ef
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b76c2f4ee9efc33836469294b86bb749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fc670b89ac304cf418e797c9ceb66ef
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_296529db2cc3c11743f0f14cff29e33f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 128, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f191fb1ae20bd2a063ae0b8db1369a55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_296529db2cc3c11743f0f14cff29e33f
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f191fb1ae20bd2a063ae0b8db1369a55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_296529db2cc3c11743f0f14cff29e33f
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c7ef4e917e2f2f0c5bde7e5835e9f28(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1d78e527f7d1a3b4ff5e6a311e24660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ef4e917e2f2f0c5bde7e5835e9f28
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype='float32').reshape([19]),
            ]


    
    class PrimitiveOp_39611369ee376aa87d1f138f4fdf88b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 64.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8ad39c6f6a94481120c881d75956721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39611369ee376aa87d1f138f4fdf88b6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5], dtype='float32').reshape([19]),
            ]


    
    class PrimitiveOp_6bf25ff5caf5ade6f1b15c43df9fe992(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11d7cd303bed6891d1a3567b4b9cff94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bf25ff5caf5ade6f1b15c43df9fe992
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32').reshape([13]),
            ]


    
    class PrimitiveOp_09329bc866aee05f275a5612047cd1a6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 64.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0aef72a8ed2793fb3e911ae29c975ca9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09329bc866aee05f275a5612047cd1a6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype='float32').reshape([13]),
            ]


    
    class PrimitiveOp_6dc665fc0903a53b96de03c659f6d31e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -256, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64322b3ea070fd4535635f720dd1da88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dc665fc0903a53b96de03c659f6d31e
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64322b3ea070fd4535635f720dd1da88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dc665fc0903a53b96de03c659f6d31e
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3b8244d0fcbab8e693c02f055ff84da5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 256, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c57eb93c7077400174f94bcb0024091f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b8244d0fcbab8e693c02f055ff84da5
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c57eb93c7077400174f94bcb0024091f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b8244d0fcbab8e693c02f055ff84da5
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_94ff99ca082682438538dc03f5c8ef14(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed47c7f6eb3cb6ab6ad24fb6786e024c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94ff99ca082682438538dc03f5c8ef14
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype='float32').reshape([10]),
            ]


    
    class PrimitiveOp_c4edd1e504301d697bfbccf0a3047391(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 128.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed0a9210839bb936bc6906ec3e083779(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4edd1e504301d697bfbccf0a3047391
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dtype='float32').reshape([10]),
            ]


    
    class PrimitiveOp_64ceedf5149135f00aa89863df9b157e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a61e60f4c663ce6acafed73954abeee0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64ceedf5149135f00aa89863df9b157e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32').reshape([7]),
            ]


    
    class PrimitiveOp_7748294fa2d8598e07ae8e275c35b279(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 128.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f95ee3e1d4e4bea5598024db8788fd00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7748294fa2d8598e07ae8e275c35b279
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype='float32').reshape([7]),
            ]


    
    class PrimitiveOp_b64f07201c8940b739c1fd4a973f9835(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -512, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5bc1abfdb83d44961a50b7c4be3189c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b64f07201c8940b739c1fd4a973f9835
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5bc1abfdb83d44961a50b7c4be3189c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b64f07201c8940b739c1fd4a973f9835
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7a5c42dd5c0fb5e17976e5bea343e4af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 512, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a4c88b956d6357268d7ea298d3b5ac7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a5c42dd5c0fb5e17976e5bea343e4af
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a4c88b956d6357268d7ea298d3b5ac7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a5c42dd5c0fb5e17976e5bea343e4af
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba6aa21921b50367ea156b9bcabcae6e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.111109972000122
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e3dbfca58ee7f75c914d14f55bc27cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba6aa21921b50367ea156b9bcabcae6e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50506f1acc43b1fb2c33222148754db7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f2870960429ee041fd73eea36873ef8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bd89d1bbb1a79848632b575058e1d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_fb06fa17a10eee898b05dbe9d448c2cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.to_tensor([2.201258659362793, 2.0960209369659424, 2.029393196105957, 2.0144710540771484, 2.0976173877716064, 2.191070795059204, 2.0642569065093994, 2.0307860374450684, 1.978666067123413, 2.211827278137207, 2.1965835094451904, 2.234017848968506, 2.076364517211914, 1.9699567556381226, 2.1472253799438477, 2.0376336574554443, 2.170506477355957, 2.3005428314208984, 2.0516340732574463, 2.1196653842926025], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_12200b463cebc0a6180462a16517f299(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(2.345494508743286, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_30435898f39bed3f52e303caffe00bdf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64bd886f4e043501ed965898bc2ec751(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30435898f39bed3f52e303caffe00bdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c2d472c5dcc6ef317541475482cd8372(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b46d8b2d71f8cb9c43a79b02b632469(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2d472c5dcc6ef317541475482cd8372
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3adb9c84ea59fb1c34c9d60391b9ed73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(354.8050537109375, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_2b212603cf876c798e916ddb9d562890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_746f028dab8de0558eda19370a61358f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba6edc4744bb3484acc4729e9c5f0cf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.043187495321035385], [-0.013158541172742844], [-0.05806963890790939], [0.015963705256581306]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_cf1d4f6f1c2cd1c62dd9f69a42d8c907(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04242425411939621], [-0.01673356629908085], [0.09204878658056259], [0.010856859385967255]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_84c87e5f3f2b98691ce232ba83d8885f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.017990680411458015], [-0.21364393830299377], [-1.630857229232788], [0.47037965059280396]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9af866f19ed00626d6f8b105a3f7a999(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.9820092916488647], [1.2136439085006714], [2.630857229232788], [0.529620349407196]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_b192894977e7d37e0ec34e2232b66f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4b7db55a3cdcf0524363c2a62101509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(30.832040786743164, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_bfc41994aab3b18cc26da1b36dcb5fbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fe00bc5e235247d2fafbc8b2785f387
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5af5f2e5b8f3863931463ffdadd02cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1aeef020f34d701f6589063868529179
        def get_inputs(self):
            return [
                paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1217c801a9c3392b980095d8584fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1217c801a9c3392b980095d8584fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f57b5eee2a60dada77485ab304b88b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74d95e5f40182f489f8442765fdefcd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3250d344bb82a9e48ccbe1c1d43864c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_b93715d5a45b7c97bf8f29621d23ac2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0164285f200319335840718e44141048(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_0164285f200319335840718e44141048(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_9ae6c01874e8b7f965ae950c0f971ad2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_754c9a96961775053727e2a2925d9098(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ae6c01874e8b7f965ae950c0f971ad2
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_913255f8b81d53c841337b33707891b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3aff02cffa7eacc060683d05c82406f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_913255f8b81d53c841337b33707891b2
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14f043cfd0bdaf099b59931fdc169c98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_437ce0b604c4036dc22b76e06cd42165
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9364f193dd21da245c6eb56e5855af6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afba1cf8682ceb676f18753ab6359fcf
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_138d65e5eab37deb7eadf8f0cba1a85a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90fca697550415213fc195bb4ec940fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_138d65e5eab37deb7eadf8f0cba1a85a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8bd2d53ca47875035167e1493a22e8ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f264697943b0b351e56d6cd6d8c0249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bd2d53ca47875035167e1493a22e8ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92219428af8d36dd4d46fc6a63b927e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27c8e5c251f3039f8cc47c6e2e1003dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f264697943b0b351e56d6cd6d8c0249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bd2d53ca47875035167e1493a22e8ca
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4789fd3787f0647a31ea5cff0373f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f21cff98c9324808e92ecdeec1180945
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_426522a94bb1f85d99fe7f5d85707121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3812f6031133a40cf34e1a497ee8ab92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_053a0d846d6ce4c5e45b4561ea46ee7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3812f6031133a40cf34e1a497ee8ab92
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_28fb0c60451f21e787ed8601aa42980d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0bda4c39b5f6db7de6a3e2e2b3a32717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fb0c60451f21e787ed8601aa42980d
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a5f99cc24d925915e75615bf9f3ba2f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ebe269cf520464c29a2d2db9c6d78dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f99cc24d925915e75615bf9f3ba2f5
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea5175794f9df0e8dcde5656c13439fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_303a73d1379bae080b18f8096f1afafb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea5175794f9df0e8dcde5656c13439fb
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cce2ff6c4c97eb5b9c6642d231358f49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f2fcf68a43499582eea3905c6746b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cce2ff6c4c97eb5b9c6642d231358f49
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            ]


    
    class PrimitiveOp_8cf5f0d058ae51ec709c5deecee1d2a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_befbb5d9ca88beb3675aca7d515e88bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cf5f0d058ae51ec709c5deecee1d2a2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_302eeaab6c1b5d9faaeeea953d5f7ae2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c211a90651097af6b6af92199305fa0d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20618b5adf076840402056af1929f263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c211a90651097af6b6af92199305fa0d
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bda4c39b5f6db7de6a3e2e2b3a32717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fb0c60451f21e787ed8601aa42980d
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_91240186a7f6615c196bacc7dcdb44a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea21460d8cd6303be213cdf7a11aaae2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91240186a7f6615c196bacc7dcdb44a2
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_303a73d1379bae080b18f8096f1afafb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea5175794f9df0e8dcde5656c13439fb
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2d0542bddecf4d7dfdf2cabf44869eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.5, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3caadcf3c8ab68df17aae8ce9a8b72e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2d0542bddecf4d7dfdf2cabf44869eb
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_06616ead3a6f7c9c5e828195a3f563f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cf5f0d058ae51ec709c5deecee1d2a2
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_264ab44f1344f96f06a07c5d83c384ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(164.5021514892578, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_ae1e9f4d4a61d7996d5ee1796562f5fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(3.465967893600464, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_20f9c9f87e5c615e695ae7294dd79440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f3f5fdbd8eeb5dc0145c712c2badc41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed60d62343112eb2efb74a054225b8de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4d3e137762b9240ae12d645360301de4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8b5976bcedfa36dfcb66c7f2b3b5d7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d3e137762b9240ae12d645360301de4
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d4f98330a4566228be25ab9d00316b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c86b74ac0c50a36ceb191dd55507180b
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3571fcdba8978db607871d8a688fdcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9a9d58c9e9d927236a0dc9d7e4328f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_437ce0b604c4036dc22b76e06cd42165
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf2ccb7be35f2a80021049ed92504f8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34880938add65f0ef6154becaed8dcab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea5fd51b8c2e3bf57cfa6498b0572286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d669d76cec7b739c0efb639b3b81be1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0608e99c9dab8fc403181b911eee47d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4db763415ecfa8623fe051b12dad7975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(38.16582107543945, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_bca965141b2dcb3b17d40b1c6d752842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3f46c7ab9c186e947f04431b129ac53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cb7d3b7a51acdd0f5216a9e44ef01cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cb7d3b7a51acdd0f5216a9e44ef01cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00b12f29ed446d69dd4dda937595c107
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f25c7ee03aabb03676a63c1131a8da1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29d7beb1a1525e56061a2cbae747ac9
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d03be7ff94c0c699e030d364551231c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34f918ed05118ae0b2b384e83060cfc9
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06e11ad911fafbcfa03757aeb10c8a75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5570a5615323312b26b87548f3e5ee0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_7837c20e0f08691132cc85c8bd8a4516(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acde060267c3c392fdeafdb588672fb
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d4fd7216b1265d6c2e30333adeb2aee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_4d4fd7216b1265d6c2e30333adeb2aee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69801119e9d6255d50cc804ba90b2ef7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_d4a7c2680cbcd9006eff92267d527108(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4a6f83e9516ddda198ae398cef845ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_155744112a6d0b2aad5667f4cf46fefc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a734ed0a6eee67994e492a5081bbaad8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_155744112a6d0b2aad5667f4cf46fefc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61b2a52efc794f00a41cad3c2a06ba1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(173.64675903320312, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_316e9a13b3e9939c018a7e2aa2d9c589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(4.998120307922363, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abc761a823d3211bae3f31a725de49f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d3e137762b9240ae12d645360301de4
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bca965141b2dcb3b17d40b1c6d752842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_703ae14165c6999db8e37114c95af7f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b48072bcf4c7541b901202c5c4da5f72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_252994c8ea087eb78a65e9c2bd1cf6ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(37.80712890625, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_b10a9085a9009c2131d0b18facd02741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac9d1d70fe871e96e839ad584322fa3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.17677700519561768
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b078265b5454580b60fe50a4e8aa9757(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_72f8d93ffb684ed068f0414570852bbc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.125
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab939471275682293c4bc2608179afa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1cac7b6481e1404ae9f245bac2c0a32b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.85, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ca3c9b3250ce547edb7d09fffa28930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cac7b6481e1404ae9f245bac2c0a32b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.47711414098739624]]], [[[0.48290640115737915]]], [[[0.9205596446990967]]], [[[0.48893752694129944]]], [[[0.47654974460601807]]], [[[0.945101261138916]]], [[[0.5162755846977234]]], [[[0.2689559757709503]]], [[[0.13848865032196045]]], [[[0.6119210124015808]]], [[[0.9737712144851685]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_c0175b771fed801e908c849ac38b67e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62084bc3294f846f1db04644a1442863
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b66442db04aaf71d5d1ded7131d040c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.875, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32832927e7f2abe2b7ca35baf9659ccc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b66442db04aaf71d5d1ded7131d040c
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efda602d03af5edca481a7bb8c4c947d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ebc2b632cf02877b3de2fd7e84a133
        def get_inputs(self):
            return [
                paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b456c63b89cd28ec40a34c715f9f83a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[1.0475993156433105]], [[1.3977339267730713]], [[1.1620545387268066]], [[1.302907109260559]], [[1.187252402305603]], [[1.586622953414917]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_691ccf7d0b92e550de85b72ae033b757(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[1.5293124914169312]], [[1.118884563446045]], [[1.217949628829956]], [[1.4280527830123901]], [[1.4317429065704346]], [[1.4073894023895264]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_7607802d9cc04b82f1dacacab3c33286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8c378322e6b0009697443d39f56edac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5cf701d401b94c63a1b4d2e6d8b4a370(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_810ca271a5e6621b97494c1193805aa2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.10000000149011612
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc5ab2ac2f199b7163eeb142b440caf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_810ca271a5e6621b97494c1193805aa2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.41576087474823]], [[0.015224261209368706]], [[0.03696832433342934]], [[0.35867393016815186]], [[0.4307701587677002]], [[0.41868856549263]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_9aac652e81bb4e846e78d0be0d58815d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_810ca271a5e6621b97494c1193805aa2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.11388634890317917]], [[0.06622624397277832]], [[0.13420802354812622]], [[0.02208777517080307]], [[0.32159924507141113]], [[0.13441090285778046]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_f31932fd292365d2c30537a265002452(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.20000000298023224
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f346d58e784416d78c40d27e7b4dad46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f31932fd292365d2c30537a265002452
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.3540762960910797]], [[0.4518500864505768]], [[0.2554236650466919]], [[0.377299040555954]], [[0.13825984299182892]], [[0.27891501784324646]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_9afd2c67712f2bf82fd6922fba60c8ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f31932fd292365d2c30537a265002452
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4278714954853058]], [[0.14499519765377045]], [[0.3763442039489746]], [[0.23963865637779236]], [[0.4272620379924774]], [[0.002761698793619871]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_a9af6331f7e2b8c3ede97a55fca33122(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06e13373549b850f40258dedbc40da49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df4bebbd214b80d70ee130121f266d89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b114d0394649af575014139ca430f4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a6f72b847077d007ed34effc3198777(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b15c47acac509117c59531d493ccfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.to_tensor([[1]], dtype='int64').reshape([1, 1]),
            ]


    class TestPrimitiveOp_0d3f3d12d4dfa7e89d9e9297f3beea59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab13770c1c8e73f0c8c33025d87e6b5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.24569982290267944]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_e3a98f673343d4e292bf763e5d1a560a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7643cfdc55b87d6e6615a5f85c63a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_203b516020a9e352f0a4b5a4a9ed4d92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2957b85a70eaff95509806b6ad88a8d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78d79157d8a2e39458b654ca6ae4d6ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_223a93370d382dd4eeefd543267ba8b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8913a29af627309d2df2f1ea366d9e85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65b4f90b34bacc1da083ca3920634d19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_8f6feb368205d142dc126f97e4c06227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_b67f7538dccef28ec4afc02b12255360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e9cb9d1d1f42f1503e8cff29f3a3903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78d79157d8a2e39458b654ca6ae4d6ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77602384ac1e4947054d3296fd5fbd1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8913a29af627309d2df2f1ea366d9e85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dd91e19a6b7f90e92232093c0acd55a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_45ee7f95c2953a4826d9c0790ed095d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_d5f651f18d1ad632440ac56f1b6d5546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4539879e90bc700d8f7dab5b7c27904e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bb2b3cc78c3d0b493e1d2ebc5ebc480(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2d50b636a4d28ba527a744dcf9b95c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fbc2160cb59a6c3183ae633b3d95723(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6aa71ec6e566e0cf8e23c190cad96f6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(1073.86474609375, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_86aeb6d74b57ce6de68ef1bd22c3a053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(179.33457946777344, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_bd91dd62e403f6cee056ad98b3de2e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(5.321410179138184, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_fc15d824fbb144fec18e9793e10865e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab13770c1c8e73f0c8c33025d87e6b5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.00010448904504301026], [2.6467989300726913e-05], [0.0014538541436195374], [0.021787526085972786], [0.005431822035461664], [0.00016644439892843366]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_de596242590af9d50c37d58bb4faa479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab13770c1c8e73f0c8c33025d87e6b5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[2.793305611703545e-05], [0.0038048368878662586], [0.0025045897345989943], [0.0007974092150107026], [0.0007925934041850269], [0.0001352709368802607]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_3bf5f2f9b24cf713cda09634f823aeaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8465df2a75f3c60dde60a5935b0acdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc7904c6f6c64113ec7e3c3ed8e8fc3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.08333329856395721
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59cbf700eb363dd244bc1e546baa19ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc7904c6f6c64113ec7e3c3ed8e8fc3e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.02352176047861576], [0.1643000692129135], [0.1283266544342041], [0.1822909265756607], [0.12502682209014893], [0.02151423506438732]]], dtype='float32').reshape([1, 6, 1]),
            ]


    
    class PrimitiveOp_849742d0a4d396f16f4193111162aa20(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 6.28318977355957
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7dc1273bb111de435b53f217446ad7e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_849742d0a4d396f16f4193111162aa20
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.001960145775228739], [0.013691666536033154], [0.010693883523344994], [0.015190904028713703], [0.010418897494673729], [0.0017928521847352386]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_d0ffe993326d14ce2212e540483b4ae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab13770c1c8e73f0c8c33025d87e6b5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.012315968051552773], [0.08602733910083771], [0.0671916976571083], [0.09544733166694641], [0.06546390801668167], [0.011264830827713013]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_77d7bb0d6f1abddb984219d02b200185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e511fc182a29090a8fb07f3b8b6c5a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad52d41a06c23856247e08781b1b74a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_236d25f522f9ea5932fa8a5c09c7cd84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.9125, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_775fd929776ae89fb4399807d2d98d25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_236d25f522f9ea5932fa8a5c09c7cd84
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dee8bbe52fde111b335202fdfbf4a21b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7f7df3f4ced853bbfbd3ffe92c0a90f
        def get_inputs(self):
            return [
                paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5085936b053c2f67ff5edfff3eeebc42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6fd950033e496cd13f3cdd9c9a97cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_9c20ec976fb6b3a1af53207d15aec7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.to_tensor([1.823246955871582, 2.051959991455078, 2.0047521591186523, 2.0986404418945312, 2.0738117694854736, 2.2010343074798584, 2.2370846271514893, 2.0122716426849365, 1.9902360439300537, 2.1795969009399414, 2.0542681217193604, 2.046412229537964, 2.133507490158081, 1.8996853828430176, 2.2763595581054688, 2.2851154804229736], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_43528a6abd61fe8ee8e868f3fdaef36d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(2.0265331268310547, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_ac86be1b700219c9f93930a997de2133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43cfee128c96bebbd6919e6fdc5ad7ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb466e79e26dc4bb2ad259120cd2afe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d23844fd0bcb66ffb5c5cefe4f86ed2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d23844fd0bcb66ffb5c5cefe4f86ed2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a8949222ff2e2f550bd981a35af3ee2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7607802d9cc04b82f1dacacab3c33286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71fab5e08f945192faa93e469329bcb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d76f7f66d33e0a55d258b8b6d1bc2cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea5fd51b8c2e3bf57cfa6498b0572286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f639e6904d531a1d14510c914d1ac557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e069f8d10c16199271b34812ada89fc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(37.86699295043945, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_04844eb551dd53a44e30ef9abd3eedc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57589b61b753e1bec45f712b1c22936b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_853a07fd0f5a797439b288522ac5413a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b4d51baee2ccad1c2b2ca32a8d2f862
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_d8e9f42d9ebc9afe37d07deddded7a51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e0161d80430d39567f3df0ae9557e75
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_03b7c841681cdae509b8a3b3929348e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c52e26335e980699085d3cd57b0b68df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_353248803624d0af2db2b56aec96499e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c0eba9469a54ba71c383cb84dbfe22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27575670956482a0f366880131b613ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b2c6533c5ee4d7daa6a9d52ab6c6ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b2c6533c5ee4d7daa6a9d52ab6c6ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bf622df8b7bd968d60ea4f757d95d51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7898145fc8b036fd3cb9df0bc7ef851e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f788670d621586c3d6abce5969f7b2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51ef01c6ef2bb52d48447478495990ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_e60197005fe19b62e0ebbfd4f57a6cb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff8ca5392a0aa2151878faa3c8b6359a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ff8ca5392a0aa2151878faa3c8b6359a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_3d0cba48db4657988deadfabf5378a7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40ea87a11c21ff17ad70e376be754535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9b1396025d6a678b98002d67c9e54b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b66442db04aaf71d5d1ded7131d040c
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.2360266000032425]]], [[[0.01978977769613266]]], [[[0.5263948440551758]]], [[[0.7462403774261475]]], [[[0.5974621772766113]]], [[[0.5967616438865662]]], [[[0.3465817868709564]]], [[[0.9959466457366943]]], [[[0.9737491011619568]]], [[[0.9637360572814941]]], [[[0.9228842258453369]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_84bf886880164e57100de4e90c912f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7ebc2b632cf02877b3de2fd7e84a133
        def get_inputs(self):
            return [
                paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8cb56769145003c56028d1f31188b4c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.95, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_564f3e8f876978411c8636934eb99e28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb56769145003c56028d1f31188b4c2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.385583758354187]]], [[[0.49460527300834656]]], [[[0.7205386161804199]]], [[[0.3620162904262543]]], [[[0.09405810385942459]]], [[[0.44892749190330505]]], [[[0.5897328853607178]]], [[[0.308209091424942]]], [[[0.2562907338142395]]], [[[0.6189999580383301]]], [[[0.08149252086877823]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_d5542695a28f6196aebaeb4f236be8bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1aeef020f34d701f6589063868529179
        def get_inputs(self):
            return [
                paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ea5e9f3c5a375b66d2fe47f417335b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_509b572f4c5755b30fd32f5d1c1396bc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_085964f52f151f4e708fc875cb644db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e0161d80430d39567f3df0ae9557e75
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_840cd6a7c503b7c8704db6dcc405d4ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(93.6373062133789, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_7e22890478b7f9afe08973f8483f52d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(3.5218822956085205, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_a23bf077408167d166b66851d6237e8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ccd16204147ef4197a0cc221472bd9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbb4c282412f631502e11cfe5fc96829(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05b47e546439f0ebee1d780a39f2d444(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b99cf15c3b9dbc5c7d131d2bb684e501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3cee784b31c982693037eb139b5fdae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1dd1a3b5263d911f145f9b004c9817b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b91e25dd045818fb512f598e06bcc71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_080ef0ae433ed258dc5b9edd112b66a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79f3e65ce16ca56b45399f8c18d5d2bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_aaf047806f2e2904240304139ca8ce7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_69fe73e127b510e2606bd383627bc0ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1dd1a3b5263d911f145f9b004c9817b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_080ef0ae433ed258dc5b9edd112b66a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c36db3f71bff76a71e9bc8f72828d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_80cf1ff2022f9313395281af5b8289f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_d6b43dab4e00b1116c970b2abdb6678d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b4d51baee2ccad1c2b2ca32a8d2f862
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_0bf875a78e6910f098a263027d20c92c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e0161d80430d39567f3df0ae9557e75
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_023b41fbea0f0970107b92662c6949dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(134.2923126220703, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_b853104c9908e51801fbe2ee1eea61fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(3.1249592304229736, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_051bf3e5a9d87164153c5615831366d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.975, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_768cef05000d8b52bf1bc56163804104(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_051bf3e5a9d87164153c5615831366d3
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f647e677c6a0cff662e4bf5c691325f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd1941d1ac5468465f30b9b0b28c616
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cf701d401b94c63a1b4d2e6d8b4a370(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e0f73bc811f6da601617c7cabea4fa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94f47fa76ac4b7e8b8b5ced0583eb83d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d77de2ae3aa17134e140200ce2fc0dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c75b3897e203df41ee2d6fe3327cfa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13e38044ea260212bae10eaa0f09f4ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaeb186d301592132c73829c515c3bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcfc0b40d70ea2fcf90908d014d35e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2f13cb24b61e6fb58a98a328443c088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.006484865676611662], [0.05156746506690979], [-0.11047624051570892], [0.0472310408949852], [-0.024203266948461533], [0.03868870064616203], [-0.07902292162179947], [0.15735942125320435], [0.08276878297328949]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_f40abf651a3f945b694579e1b3b47c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.02537374570965767], [0.031217578798532486], [-0.04251473769545555], [0.022202739492058754], [0.11096224188804626], [0.0752519965171814], [0.03307155892252922], [0.06225350871682167], [0.09541821479797363]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_8bf5193616150b15877fe8fe519cfc74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.7444261312484741], [0.6518726348876953], [1.5985398292541504], [1.1272618770599365], [-1.2181216478347778], [-0.4858780801296234], [-3.3894524574279785], [1.5277197360992432], [-0.13256831467151642]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_f107338497b2a385f99d224be9b4bdd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.7444261312484741], [0.3481273651123047], [-0.5985398292541504], [-0.12726187705993652], [2.2181215286254883], [1.4858781099319458], [4.3894524574279785], [-0.5277197360992432], [1.132568359375]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_0dbf0393c460102bd4a17459064d1afc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e30385e43a24ac70da5286de189f5c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_736da84668d96aa70c9133b4d649ebfc
        def get_inputs(self):
            return [
                paddle.to_tensor(11646.103515625, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_93053cae0cad912fd27e70989afdc7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(1058.73681640625, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_949835718ed2787242a502043b979e74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06662624329328537], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ca0351031241df8678931bd920e8bbfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[1.2875372171401978]], [[1.33564293384552]], [[1.5743564367294312]], [[1.2732023000717163]], [[1.0654644966125488]], [[1.3299976587295532]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_3c83bcf408a955b2307ec379fa2c3c34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[1.5494780540466309]], [[1.1539307832717896]], [[1.102842092514038]], [[1.6390326023101807]], [[1.4597944021224976]], [[1.2427841424942017]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_f0b8234ddc45fced739f4978a5bd91cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 128.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2214f8f54d9bbd72a3a11d8814313c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0b8234ddc45fced739f4978a5bd91cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_454ad6aa18b43e6ea4d79623d8f71381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c56a0f0e00e9f7063bd373353711e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2214f8f54d9bbd72a3a11d8814313c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0b8234ddc45fced739f4978a5bd91cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a52209ce9424bde9bc2b472302a028bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6775e62b9dd9e1b779a0240170a18cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6775e62b9dd9e1b779a0240170a18cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_504e83a30276c346583a2c3ae2c04c7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbb884d5e0621f13c030d0e5e375be29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ee783d944681a81bfa0ad6cca6f3c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_aa4418408f275828b2bede8f7ec90b14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fdc190ae38110626c7e6c2a11d2c739(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_8fdc190ae38110626c7e6c2a11d2c739(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_a574ff8b6f967143e0e3dfc50f3f6ce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33a532294197600bebd5564e49551983(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e71658b495e612824c3466267a69c05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc1cd66e920bad8ef7cb27d9331b0e1f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.8375, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d74496fee5cdca42d7326588efd61d29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc1cd66e920bad8ef7cb27d9331b0e1f
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae97f7f8f3d8ac0c53d9761fa0abe2ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78f66699062fd7c835e7a5fe4c6ffa05
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5799b3e5bdc91e103ab5ccf6c2a4a526(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(100.98829650878906, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_9147dd13085a7021fff0666d2dd43fca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(5.6871795654296875, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_8792d9f54d768e72f8a695c5111e113c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_cd7bfdc9f893afe79177c02ee3cd6e7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd3a2c47190f0a9bdb82b03c21b82c5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(6.652104377746582, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_85293754e6216d82d35c2e240f36e431(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4d69f8aaf68e992373373620a4ad0399(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49bb878029af806d20b221a6b8e2dc25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1501445472240448], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_156a28e0e240c2326b9a04705fc563cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.36977577209472656], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d538cdd7e2a1d10c6a4cfeebda9accdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.8370978832244873], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b279953df0230905e9e8f3f20d7cd258(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f036ad84c5f9f5665d0718ee2e67c965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83749efa7e14fbd125be8f3896095e36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a46e04f7f15f6b9e3284f37b109fbf7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(145.07211303710938, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_a347b03d29be4648f4b3a3cc459ef848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(89.48736572265625, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_b31eb91c6bde9aea3f4234fa680daa1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaeb186d301592132c73829c515c3bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcfc0b40d70ea2fcf90908d014d35e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6051304680960668197f94f80d54dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6863819360733032, 0.5550834536552429, 0.5681371092796326, 0.25456511974334717, 0.5099325180053711, 0.24781492352485657], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f9060565171d68b8f545e1451b2dcdd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5956051349639893, 0.4590386152267456, 0.32019469141960144, 0.5512910485267639, 0.6469229459762573, 0.07742799818515778], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_dff0c5e215553eebf41aeeacc71b8c5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.720000684261322, 0.8646494746208191, 0.589462161064148, 0.8502786755561829, 0.7028647661209106, 0.18222540616989136], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_5c5cdb406e3d5285bfe4344868a1fc65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.7023019790649414, 0.4557735323905945, 0.7801243662834167, 0.10672207176685333, 0.6334706544876099, 0.7819021940231323], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a159cd68f69bab95a43ca71d0cd40009(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe4e3afb46f56e88957b595216df89d
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02505781129002571, 0.017090944573283195, 0.004393713548779488, -0.0008124950109049678, 0.017373912036418915, 0.0037726969458162785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a8ba0e0e2bb02b2dc6fc932275eb446a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe4e3afb46f56e88957b595216df89d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0031286091543734074, 0.023960445076227188, 0.052997514605522156, 0.13812905550003052, 0.009350953623652458, 0.12514647841453552], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_474a1f0b8a9844c5cf5ff92c3335929d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe4e3afb46f56e88957b595216df89d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.05580516159534454, 0.21465319395065308, 0.09616874158382416, 0.3242293894290924, 0.04169569909572601, 0.15700317919254303], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b9785daf15bd941ce250d01a788dec4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316277a8de572c09c037ea3280794036
        def get_inputs(self):
            return [
                paddle.to_tensor([0.08393514156341553, 0.7033591270446777, 2.8352646827697754, 0.059799134731292725, -0.1445942521095276, 0.6819069981575012], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_83f63c0e7d95d650295780b6518b04dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1549ce8f0624f6cad262f5c25e7762ec
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, -0.0, 0.0, -0.0, -0.0, -0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b165594ee9e94c49a2a2236a318de00c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe4e3afb46f56e88957b595216df89d
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0028553009033203, 1.200500249862671, 4.257975101470947, 1.0014492273330688, 1.008473515510559, 1.188456416130066], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0df670ba014043394a21c4cc1521d16b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([1.056071162223816, 1.1451103687286377, 4.043917655944824, 1.4260247945785522, 1.2243378162384033, 1.826979160308838], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_606707021f290774b147ace02f7263b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6acd5787b0df385fda08f89c4aa7f9
        def get_inputs(self):
            return [
                paddle.to_tensor(1.7870736122131348, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_b078265b5454580b60fe50a4e8aa9757(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d614289302fdbb13e27c23e0d17edec7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d614289302fdbb13e27c23e0d17edec7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_254bcd9e19dbcb598b7cbdf9ddce6d8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a373c85edc561bea5c4eabf3dc6a9fa3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0b77fe869e64ae47fe278c33c5508f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_eb2cb539f4ea34919dac56877f4216b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d99322b7a82d7e6089533e399ac887f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d9efee2ec0d1121ba27ba5977d87dd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d99322b7a82d7e6089533e399ac887f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_2d9efee2ec0d1121ba27ba5977d87dd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d99322b7a82d7e6089533e399ac887f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_3cb1cdf253faa2cbfa1731011d66694b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13383a008b8da01a18343164466b5731(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a554b2167b8c31c09da4bcfe06472caa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2eaa98dc611307fa6db7713e4a1c53ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a554b2167b8c31c09da4bcfe06472caa
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b0b94e7709cef36fd15e80a8faa6841b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_863614b8db8e976ee3882d07d69db765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0b94e7709cef36fd15e80a8faa6841b
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0ba725bb40818d0940b94caf119c589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_810ca271a5e6621b97494c1193805aa2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.20113162696361542]], [[0.01791820302605629]], [[0.1196950152516365]], [[0.4878165125846863]], [[0.25780704617500305]], [[0.3775233328342438]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_ea9b0ce6b9b4dbe2aeaa0ec747a1abba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_810ca271a5e6621b97494c1193805aa2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.0816684290766716]], [[0.17039605975151062]], [[0.4779324531555176]], [[0.10027485340833664]], [[0.247049942612648]], [[0.3199577331542969]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_03d9dd5a872cdc7a4f40025442e43e6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f31932fd292365d2c30537a265002452
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.19105783104896545]], [[0.07815577834844589]], [[0.2450648546218872]], [[0.4223167300224304]], [[0.21929742395877838]], [[0.10183952748775482]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_d84d0d5e360ca15ad642ff046e3296c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f31932fd292365d2c30537a265002452
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.18758375942707062]], [[0.1623629331588745]], [[0.4591658115386963]], [[0.12450161576271057]], [[0.3159920573234558]], [[0.3298834562301636]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_3d0cba48db4657988deadfabf5378a7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69570d511bb5da20ee4b28d8b2c18e11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab4b2160c98bbd1d475fef7812af39c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.43301984667778015], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_85d599641763cfce68cc781bffdcafc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8638f2c96ca58af820de85e3e969f4c
        def get_inputs(self):
            return [
                paddle.to_tensor([0.04718353599309921], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_49a2488b09af6a37d3e47af64155f067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb7d2237f86ea801a935b51885b1a37e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.38502153754234314], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34880938add65f0ef6154becaed8dcab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe95ba28d76542245d272ea736fec3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f64ef7a0898ebad80ee2e5012bd44e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03b7c841681cdae509b8a3b3929348e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dee2424b7a4eae5dea1e72452b65ee75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38b9b85ec5e84c919ed5fb8e85678067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_ac6bea5c0d876f49f637820748aeb419(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_0fee9067adfbd23aa5dc00efae46f97b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82692f1e7eed7a3bce1d5caac0d9a3a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f64ef7a0898ebad80ee2e5012bd44e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec61be934cdaf46e87f75e87c96a857e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dee2424b7a4eae5dea1e72452b65ee75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64f18129b0a6d125499f55d7c6a554c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_229d82637afa4bdcd8273fc78842d395(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_7b5922f9467b5aca67fe1634830e3dbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_b5d95d6d25e7b3a417913335d8b159c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.to_tensor([2.042098045349121, 2.1673107147216797, 2.046053409576416, 2.187094211578369, 2.039552688598633, 2.066070079803467, 1.9485445022583008, 2.015756368637085, 2.2058639526367188, 2.0801162719726562, 2.115180492401123, 2.0773661136627197, 2.1335859298706055, 2.024040937423706, 2.078754186630249, 2.0380187034606934, 2.1577341556549072, 2.129593849182129, 2.2176289558410645, 1.9216270446777344, 1.9373674392700195, 2.045713186264038, 2.1750686168670654, 1.9353487491607666], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_b4d7e40c8604b1ef62a31a34efe8156a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(3.5862324237823486, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_f3714e74fb126d7b2ce24727c2a88cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.43497511744499207], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_7e8c085d6c3ca99b4f60e7b2df30a8c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47d4a9a6f83a693717a9f028c2a21316(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e8c085d6c3ca99b4f60e7b2df30a8c7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.38216280209126474], dtype='float64').reshape([1]),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2d50b636a4d28ba527a744dcf9b95c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2fddbb6a528a7c93b3866f78a751890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c7fa08fa11e1334469e857235f9222e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbb4c282412f631502e11cfe5fc96829(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03232d5fe76ea29afc881e2d7db44d6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bb38ddb13d93e05d2e665590a07c59a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_788e0342bfab3d78a4078eb34aa62914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_788e0342bfab3d78a4078eb34aa62914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e298d4211d77c5ba6f7cc8e547ca4e3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_013c4d8c708f598cb382348741973eb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34c19ce7894c49811fe0218a77ff66cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_df9b6832d9cbf7f010d906a17ddeb645(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7eb4fa31dd288b9492cf7c8ed8f5de5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_7eb4fa31dd288b9492cf7c8ed8f5de5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_aa02099785e654681a3d84bf08b036d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8258c884d6dc84d25317fa2600fe2bf
        def get_inputs(self):
            return [
                paddle.to_tensor([[9]], dtype='int64').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ba47796e88da6449e74b64b3258c304b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab13770c1c8e73f0c8c33025d87e6b5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2439064234495163], [0.24398155510425568]]], dtype='float32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_ac86be1b700219c9f93930a997de2133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3cc1ea34e5aee9740b82608264458c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf82d03b1a1cd742f431c5af366487eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f169ea5e8273c1d9ca0ee0cfccea73ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_666850d03d7b3924433f21bd579c13dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.to_tensor([2.025592088699341, 2.0849061012268066, 1.9092397689819336, 2.135376214981079], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_9a91a1c7f34cacacc4a18c61f1a4cf8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(0.604914128780365, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_8b114d0394649af575014139ca430f4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db588caa2c2dbfa7924a56d53255ec1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_221acb9c76b4cf77ab2398883b86cbc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_800a07d97880f0d8fc8f21655a5a8ad8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(147.2303466796875, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_2165ddd655d4997a3e97ffb43d503dba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(4.197933197021484, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_f845b8667f7c2294e275b5e5588f9163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(137.94223022460938, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_f03eb5295661b2559dfe7c4d21f870b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(4.009830474853516, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_34a4d7b363c3521f8e86e770ec622445(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.25
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5789f09a4863a6a1eb7cfef8609f475c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34a4d7b363c3521f8e86e770ec622445
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4760d1aa4447b59cd8e36a4513b8e634(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cb0cb18ab122b9b9357ee65469ae8a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(141.23519897460938, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_aa3d6e27538f3df0742be105f0ba2511(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(8.003020286560059, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_293c5db58182917a5469932a7478cb54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01941882260143757]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_83518554c2ca8becfd26c0d3a152511f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.025589246302843094]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_993def4b3035112dc237d159a5e01a36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.24113346636295319]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c9e8997baabc3bbc0ac5463364ca9226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.241133451461792]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1489de721f83eaa80867d29398aff6ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.043292563408613205], [-0.02788546308875084], [0.05995785444974899], [0.03143922984600067], [-0.0009129986283369362], [-0.034111231565475464]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_3357c9602fa0b501674421b2cdaa9ab1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.002421251032501459], [0.012762386351823807], [0.12483116239309311], [0.007755537051707506], [0.057623084634542465], [0.02943781390786171]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_4396919826e7af5f41c9cbb48299d56d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[-18.880247116088867], [-3.1849725246429443], [-0.5196884274482727], [3.053778648376465], [-1.0158443450927734], [-2.1587555408477783]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_61cb47530b931380be1fbb1775d7cb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[19.880247116088867], [4.184972763061523], [1.519688367843628], [-2.053778648376465], [2.0158443450927734], [3.1587555408477783]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_0d4f18520d9f92929cb07d6c38886e6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_051bf3e5a9d87164153c5615831366d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.997045636177063]]], [[[0.2572757303714752]]], [[[0.027550404891371727]]], [[[0.17211119830608368]]], [[[0.1591709554195404]]], [[[0.3672061860561371]]], [[[0.9256317615509033]]], [[[0.3660320043563843]]], [[[0.07933235168457031]]], [[[0.8728136420249939]]], [[[0.6442326903343201]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_ba198c3687d30352279f1abebd0f6b64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fd1941d1ac5468465f30b9b0b28c616
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c46340a0f41cd2f007072840b56dad18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype='float32').reshape([14]),
            ]


    class TestPrimitiveOp_e86128d65e34c7a41de24a748e9e73fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], dtype='float32').reshape([14]),
            ]


    
    class PrimitiveOp_fec509e6b8a9527d8f8c9a9430deb77e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -80, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fdae3a0a5264d0221b976519af21df27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fec509e6b8a9527d8f8c9a9430deb77e
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdae3a0a5264d0221b976519af21df27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fec509e6b8a9527d8f8c9a9430deb77e
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a01679a17eabdb1442f5c4416939205(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 80, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_283d694bc728545cf9bfe7e03e4fe5f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a01679a17eabdb1442f5c4416939205
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_283d694bc728545cf9bfe7e03e4fe5f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a01679a17eabdb1442f5c4416939205
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb174679034dca01085ecd3c22617360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], dtype='float32').reshape([28]),
            ]


    class TestPrimitiveOp_2a4705a3c154520647a8ee453d9a17d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5], dtype='float32').reshape([28]),
            ]


    
    class PrimitiveOp_1cd3cd1f133ac003edca58bb4ee1c674(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -40, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7702dc012235dd1d8b2a138cc867cce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cd3cd1f133ac003edca58bb4ee1c674
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7702dc012235dd1d8b2a138cc867cce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cd3cd1f133ac003edca58bb4ee1c674
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_edacec6568a11c4f2a6bfc9940c1d830(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 40, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18140aa4f9f2bf5284cdbe87e15f8622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edacec6568a11c4f2a6bfc9940c1d830
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18140aa4f9f2bf5284cdbe87e15f8622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edacec6568a11c4f2a6bfc9940c1d830
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43f5fd55855515c679a89036320b4e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bff441c16abb57c6cc1e3d4fb0cf025e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d06fa605bebbdfd8b84bbea6aa354282(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -20, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f177bc1827e7197e89800792643443e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06fa605bebbdfd8b84bbea6aa354282
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f177bc1827e7197e89800792643443e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06fa605bebbdfd8b84bbea6aa354282
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc7ead0295cfe64bf2f4ce16db407b67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 20, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73c366e9c97f62640a71dbc0a451c2d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc7ead0295cfe64bf2f4ce16db407b67
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73c366e9c97f62640a71dbc0a451c2d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc7ead0295cfe64bf2f4ce16db407b67
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_710140a53524499dd92535b65b67715e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f15e8c5a79999c7fee97caf0a50c6a7
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_fd5860b084afe67d76098e7c33aae991(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f15e8c5a79999c7fee97caf0a50c6a7
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c3a303adac7bf1d7ec02aea4245dff86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f15e8c5a79999c7fee97caf0a50c6a7
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c3a303adac7bf1d7ec02aea4245dff86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f15e8c5a79999c7fee97caf0a50c6a7
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5ad29cbef7a64ba010c325406cc3d547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.to_tensor([[3]], dtype='int64').reshape([1, 1]),
            ]


    class TestPrimitiveOp_296a5d58a17e5be4a9e09637056a145d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab13770c1c8e73f0c8c33025d87e6b5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.24656285345554352]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_c13ab554fb8578d376ccae648c21a5d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13220a9db0d64bb56773b8ae08ec06d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b67d5d0024e106348f03adb0b0fefbb
        def get_inputs(self):
            return [
                paddle.to_tensor(4.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c7a9d73f3b4aa24d1e1c8494d3f70279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b67d5d0024e106348f03adb0b0fefbb
        def get_inputs(self):
            return [
                paddle.to_tensor(7.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_3f6e9fbbe672b87526f1d1b7b3ac6a7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8aeeea9bb8d1620a3b95e340e3159f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(157.97816467285156, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_859f5041f1d0334c81bcbcb9666edf7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(2.6843674182891846, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_438361975b7efeed499cf79706b93143(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.1, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e86883a33896c36508913aa9f1e00515(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_438361975b7efeed499cf79706b93143
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a23bf077408167d166b66851d6237e8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86dda8233f6b093caded9f5a91499128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd34e452c6410030bf15f3fe88bf439c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d216ffe4bd16fc049b9da4ea59d1c5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5f651f18d1ad632440ac56f1b6d5546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df807387f53add025873a3d542a1f741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f40aabca31372e0ab7df472beebe5cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(166.91445922851562, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_4fff3baf264afbc28c1858151f1ee92f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(60.17794418334961, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_b0ac2e5ce64faa6ff4a96b1abd913a38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_108f435a07debe4b81f61beb55787a39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7dc7b213fba3bb402c943a9795dfc78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b753d4aecb5bd6d265da642be128bc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b753d4aecb5bd6d265da642be128bc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02914caeccf7fa6b658f908f0d038cb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_096c6ceb73a55380ee5581807087e560(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a49332b6cfdc513bbf415485f317ad8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ae674b61db32fffbec5545c0bffb5968(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1313b124ac865e9608cafde1101b24c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_1313b124ac865e9608cafde1101b24c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_282a2bd368f097ad459e06aec3fc0091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_438361975b7efeed499cf79706b93143
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ea0a49a469227c5640cb9f6ba6a9ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(146.36666870117188, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_4c3f3dc388e3da10e630c0069d272b29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(4.160505294799805, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_3dd91e19a6b7f90e92232093c0acd55a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_50011af0904d08118f89bca79443d9a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_29a3a43470fe5862a6ede57b56776c99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.02942965365946293], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d36c552518cebfad9bb6e4ecdd6a7d3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09304554015398026], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7a046fc2e6906884a852cfa0f35ed538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.040439870208501816], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4258718d31956e6f856504486dfe3cee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2780453562736511], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_96c512e1a96850045edd3f80906df3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4882940351963043], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cb1be7bb9e895791474bdcceedefa018(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5158790946006775], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3dbb250be785ed096e65d063e9663006(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.48973387479782104], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_817600b32f35588ff0d793d304e17b2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5623858571052551], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b9e0cd98c217877c8e30e53c163b1295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.49126216769218445], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cd77ceb7e92d1d908edd935a07669af1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5432940125465393], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_41f9df3ccde1207077051d70f751c6e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.03385039418935776], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_eaf5459e847b0bf4a11eb33355d5796f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.10091172158718109], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_690c431afe7b8c605605bcf01cd78d40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4407913386821747], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_eb0169b8c6a873c138a5aa99c0dd7d43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6364049911499023], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a5a9f9d605e30c89c64f17470c03980b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3561382293701172], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f24484d6d3c3c650e90231c91f028796(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6890357136726379], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3525378aac74a2bf4a957babc189efa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d69f8aaf68e992373373620a4ad0399
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2267407923936844], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ab9ae0618b7d916495cbb98cd165789f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.39462366700172424], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a4c88302e5655fa464e72b2551f19039(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0655159056186676], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6235d5c671f1b607153818ddad8872d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.24741116166114807], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ee6414569488a6ea052788e196c8f8cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.04495077580213547], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5fb78cb99a7e97df042dfd8587ff36c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.44348499178886414], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_00b16b6faca1cabb81f8a0cf93ab91a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_138878cdd8acdc0bd5c2e77cae0a6bf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32b29d21d9e6de1fa60c54d37a173a3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71d8c4dad6f9192521b2cb72e340e2d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1ae0a58781cc6e48966d2b2211f1ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e37fe63a1fa196411593939cc1034fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ffabbc04fec1d3c404ece1212e504ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 16.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6d6b31046662e55b54e6669280307e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ffabbc04fec1d3c404ece1212e504ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5e5ef9dcaa25dffcafc3406d4144593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c56a0f0e00e9f7063bd373353711e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6d6b31046662e55b54e6669280307e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ffabbc04fec1d3c404ece1212e504ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c61ded054b51491c5e9c55e96cc56d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c61ded054b51491c5e9c55e96cc56d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0d46d0914676c13669126f1364ec9f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06d5f6e8e1f357d1ddb9eab7a9d22763(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72351ccd0a53851fa6b7086e15aac3f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_a9e1ac8738e2bd5fda1ce19f7968e479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_810adc7d23f2769bd19e44bc5efccd94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_810adc7d23f2769bd19e44bc5efccd94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_5085936b053c2f67ff5edfff3eeebc42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd4d18e1732c0f73b901051ddd948382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(100.44140625, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_75d4e8a4751898ff9479abbf07d25dd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(308.20684814453125, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_1ac0055a83d941a69f8155f059aaa301(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ac0055a83d941a69f8155f059aaa301(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18aac6cc4c1af41f0161f248fcef2e7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c443759fd53c683f5ac140ef2253f85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b0613373b629a33b0da9c515e348a9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_0d1c3093b19b1c33b8c088e8ee739be0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43d8639bda3cf028460f676b87d3efb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_c43d8639bda3cf028460f676b87d3efb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_59c9862dad764c228ab1a1599291657d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5e7ec945530c6c16095fecf6758605d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1951ec678b4be08e351e4014c7a49573
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e13bee2111d98ca1ac4a0aabcdd7bfb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c56a0f0e00e9f7063bd373353711e1
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fa49a9b27290490705804df84e7cb21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(117.01022338867188, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_972d84435fd8d54dbbda0767ea88aa47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(3.625927686691284, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c2ff4c36e45f593be6f0e524370681a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d4618b221b0938ffee14ed5cc45ce12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77d7bb0d6f1abddb984219d02b200185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f414651b75fb4117f6952df72c87df0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_523fe1f070650c0ef3e771cfcb8a1ab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e64b00a3005f9034432383dd1527b16
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ed23f4a32f45e96e21ada4cc3bd7c0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0827fd623f47c0436c93667413640909(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef523ee20fb4a9317dc39ad7a3c0ac4
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6423ce9964b21a8b724b2a8b61152c69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0016f70395d41df37b8d78530c46f2c
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64fe1e46a1afa270c46a4f1be2b3c9bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5aabf326668c5b3605786df51d820473
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0827fd623f47c0436c93667413640909(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef523ee20fb4a9317dc39ad7a3c0ac4
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6423ce9964b21a8b724b2a8b61152c69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0016f70395d41df37b8d78530c46f2c
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e0f73bc811f6da601617c7cabea4fa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bc6d0c9bbbc804d4aa0b480acdb87b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf0745789044cab6ae4fe288e3dd8a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_175015d86e7e0aff605b38852f1ad01c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_019ebefceedd87b638c8433086273548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcfa661cdc36758a508b29dc8f415893(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7302e18f40e89ac953251718dc4b967(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cfad9adda5c1b35f53128df26fff208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_78fe56ed32b261288743462c7e8ec5e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_fa5cb3d1462e8ef8406db9d853234d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46bb13638f3165bdb15be092a7a09bcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_019ebefceedd87b638c8433086273548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e3ae4589702d9f493cc0397b6fc2bd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7302e18f40e89ac953251718dc4b967(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c02ddaa8fa8106a70525c2dd0eac298b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_95174d5209b7583c6ef67265518da092(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_a574ff8b6f967143e0e3dfc50f3f6ce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a574ff8b6f967143e0e3dfc50f3f6ce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a574ff8b6f967143e0e3dfc50f3f6ce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d0875780a3089c212fe8d1ef17de2fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_988a57555ebb8cb66c5d2c2f17c98827(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e64b00a3005f9034432383dd1527b16
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a95202ee25bfb76cae8c8c55a405c6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bac737f8d8c92ae437b8efe4ee91249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef523ee20fb4a9317dc39ad7a3c0ac4
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12353d258fd2407afb7dc07ae2b9b9db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0016f70395d41df37b8d78530c46f2c
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cf39d3e804fe509a40ffd63965c8bf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5aabf326668c5b3605786df51d820473
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bac737f8d8c92ae437b8efe4ee91249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aef523ee20fb4a9317dc39ad7a3c0ac4
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12353d258fd2407afb7dc07ae2b9b9db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0016f70395d41df37b8d78530c46f2c
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ced98fa0373c97b728a251e39163ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01143362931907177], [0.00884125754237175], [0.06832607835531235], [0.0002330932766199112], [0.1191963478922844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_739f7a1bc28cb730062ccff8097ecf04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.004484008066356182], [-0.06190362200140953], [0.0013638997916132212], [0.05762871354818344], [0.01101756189018488]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_718e2f2a92283a02710e9e305169e329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.549868106842041], [-1.1428229808807373], [49.096107482910156], [-0.995955228805542], [9.818758964538574]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_bf322ec820d22a46d5988199cad5ac61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.549868106842041], [2.1428229808807373], [-48.096107482910156], [1.995955228805542], [-8.818758964538574]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_0ee753cf762f535516cc93467a3d3501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c36db3f71bff76a71e9bc8f72828d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_e448f86a00c299406166f171b57b90aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6122bf3d6e2f4a711d8afa66f5767cea
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_8c36db3f71bff76a71e9bc8f72828d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_e448f86a00c299406166f171b57b90aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6122bf3d6e2f4a711d8afa66f5767cea
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_c799d05b1bcd72cb853a0f3157fba972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            ]


    class TestPrimitiveOp_85ad6e7c6f19dd9106a1b899491efbf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50be9c14adb924915c0af546c3e4ba22
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            ]


    class TestPrimitiveOp_c799d05b1bcd72cb853a0f3157fba972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
            ]


    class TestPrimitiveOp_85ad6e7c6f19dd9106a1b899491efbf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50be9c14adb924915c0af546c3e4ba22
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
            ]


    class TestPrimitiveOp_4f3ab1d9d1c4d548669d5ca7fbfe6fa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16960aef419a85acfddcd35f76c44738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2145b69cf36e3309d25b6657d94e8596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2145b69cf36e3309d25b6657d94e8596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2145b69cf36e3309d25b6657d94e8596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb9974f7909e3a3e8dfee3b7cf69171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7f545b5150aa9189ede389a3f22fedf9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 8.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a3b325d53939fa09fd4512071b644c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f545b5150aa9189ede389a3f22fedf9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2efb3d8f194406b73cc8ab54c3e3be98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c56a0f0e00e9f7063bd373353711e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a3b325d53939fa09fd4512071b644c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f545b5150aa9189ede389a3f22fedf9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c52e26335e980699085d3cd57b0b68df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0226da7fc98fac03d1c38a35ccefb77d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d77de2ae3aa17134e140200ce2fc0dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3c483aea38d1f3cc131613ae6e04801(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20f9c9f87e5c615e695ae7294dd79440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0162d62abfc04ea2cbfc2b5fa71d287b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1cf3b90f39c790688e4db45490ac54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09171438962221146], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5a2bce73cf68c720378d2cb61c1a93be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8638f2c96ca58af820de85e3e969f4c
        def get_inputs(self):
            return [
                paddle.to_tensor([0.28419357538223267], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7a1e432fc3afbca9323bd804a1bbbf63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_591ac0165c4965ab780ad979db2b141e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4586580693721771], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1ae0f13589b297ea12b6de9d7803010b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aea8f41ff9cc4f5b2e73ad5fa379d182(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_509b572f4c5755b30fd32f5d1c1396bc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_6b93ab9f8fc5280abe5d43202c1e470e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e0161d80430d39567f3df0ae9557e75
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_d86f4a565fa4ec4bbfb6ffe164956b6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d86f4a565fa4ec4bbfb6ffe164956b6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b39ff44175f8bdc6571af8928ad760a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1f8695b76bd689b32ec9b98574ebcfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4679226fc4e3f3e3085ce696f9d2faa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_d71187085ca807584aa4431491398243(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8842a9407be5628c54400314c5761da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_f8842a9407be5628c54400314c5761da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_bba5574b856bec3350441713b11b9139(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba5574b856bec3350441713b11b9139(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef5d39e3f3deea727657fd8a80ba8838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7c7114d8e53032d18055fa655d162a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a8f18b0cc78b1c8ba213a7b205ecdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_090a4950c8b3adee572521e875628851(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_798f976eedc21461b8500400b500146e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_798f976eedc21461b8500400b500146e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_3b820be4fe8aba6e2dff023b96cb6161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b820be4fe8aba6e2dff023b96cb6161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4c946bdabe24b041c70f773137b160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c81b1242ae8fc4406109e65d704daebd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_533ca35dd67c0d921db2bcbea900284f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ec59a80be4b1706758810d94ea50de24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ea38e58a2e46a7fbb7c180425485c87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_3ea38e58a2e46a7fbb7c180425485c87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_c7688f70af973d039967a2995aa4fdd3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 64.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87f1f47fc779e546eced0baf0a1bf176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7688f70af973d039967a2995aa4fdd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b8473b89a2d6c69d2251a59c3e1ed4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c56a0f0e00e9f7063bd373353711e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87f1f47fc779e546eced0baf0a1bf176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7688f70af973d039967a2995aa4fdd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6d77b7fbee379eeeeeed5da420a3dd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f575e0c11f2aa4bf717fa144a6dfd8ec
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8189a2e823a64d2a57abc080a51a5891(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e936cafdc2b249d43bb0f9d73a07379
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3b02d563b7df7516340fb53900f818b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da479f60824321a0738da73412a0443f
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a92bcde7015d6619aff21e29b10d39dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 0.925, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bca429f38877414c419735c9c3aaf9c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a92bcde7015d6619aff21e29b10d39dc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.7168525457382202]]], [[[0.028730157762765884]]], [[[0.5656723976135254]]], [[[0.3021770119667053]]], [[[0.6223515868186951]]], [[[0.612608790397644]]], [[[0.6727516651153564]]], [[[0.5322846174240112]]], [[[0.7834969758987427]]], [[[0.6203384399414062]]], [[[0.3852967917919159]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_0b1fba21c676db60899e8dfa47a3bc94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb3e794075813e6b6ec0164accf28a6f
        def get_inputs(self):
            return [
                paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00b16b6faca1cabb81f8a0cf93ab91a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1db9ba60d01102bbb47f3b3afa313fb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ba2d735307fc2f0c79cbfee52de2000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5741237d3e358b8edc769c13ab2386ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_391dc50fca7183cf1b99d9a374ad0a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d7267e6c3620bdcfa3430e5f5fd3158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1682ce00963965a16495033d17b72940(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c91d4da9e7122617c5aa2968830d418(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e683487d6f521a2e60ebeaf48769684d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_490c77701725fc3db6d4596cad2b41b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -32, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c5daa24882a60a714c2a192fee683f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_490c77701725fc3db6d4596cad2b41b4
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c5daa24882a60a714c2a192fee683f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_490c77701725fc3db6d4596cad2b41b4
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_82305437611a4cb6ba23939aa151564a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 32, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34c2842ee68b78f0436ab6372ceb4a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82305437611a4cb6ba23939aa151564a
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34c2842ee68b78f0436ab6372ceb4a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82305437611a4cb6ba23939aa151564a
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2c2df459c5c1e25521d91d012113a98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7873d08b15b14bd95c3356469df8691c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5601e1e9aa6638598be41bcd954c3320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53c860120a310f38a2e7e52e823c1866(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_93f5c17710dabe35413e25d0d0ee5452(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -64, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6251b67862a642b12cbfec6b43002e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93f5c17710dabe35413e25d0d0ee5452
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6251b67862a642b12cbfec6b43002e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93f5c17710dabe35413e25d0d0ee5452
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eec4e745d8b4966fb31f0768f3213827(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 64, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f968c64281ba8fd87091d4ed15458e69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eec4e745d8b4966fb31f0768f3213827
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f968c64281ba8fd87091d4ed15458e69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eec4e745d8b4966fb31f0768f3213827
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_614198971572656efb5d295b5211a272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b4a498d115ecbfa71042c5b1d2e7adb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01166ba5203fe126b814d7939acf41df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype='float32').reshape([25]),
            ]


    class TestPrimitiveOp_8d05605c35086bcdfa2e4a07c2178286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5], dtype='float32').reshape([25]),
            ]


    
    class PrimitiveOp_def863e7ca80253c7ce55e37b8152576(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -128, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3aaad49f3e5ec8df27749d70b158f9f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_def863e7ca80253c7ce55e37b8152576
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3aaad49f3e5ec8df27749d70b158f9f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_def863e7ca80253c7ce55e37b8152576
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ff1624df9865487bff773906bd22fd04(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 128, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18b147b4f537813018180b7f61dd06a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff1624df9865487bff773906bd22fd04
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18b147b4f537813018180b7f61dd06a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff1624df9865487bff773906bd22fd04
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f76de7d77fa79856860054b77e92434(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype='float32').reshape([19]),
            ]


    class TestPrimitiveOp_bb9fa0ecb401d45357c3c8418c433828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6122bf3d6e2f4a711d8afa66f5767cea
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5], dtype='float32').reshape([19]),
            ]


    class TestPrimitiveOp_56274f47adf2d020002311c606ff6731(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32').reshape([13]),
            ]


    class TestPrimitiveOp_897249c4008b0358328d99ba64603641(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6122bf3d6e2f4a711d8afa66f5767cea
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype='float32').reshape([13]),
            ]


    
    class PrimitiveOp_8341218b472f20fd9bc38168cfdb5187(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -256, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75b8f3dd02cfe5184d208ca84ca2cae2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8341218b472f20fd9bc38168cfdb5187
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75b8f3dd02cfe5184d208ca84ca2cae2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8341218b472f20fd9bc38168cfdb5187
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a0ec4f4e6d5374b39ed0aca410766f9d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 256, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a986e8a1e4da6b5ceef6460b326f4b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0ec4f4e6d5374b39ed0aca410766f9d
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a986e8a1e4da6b5ceef6460b326f4b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0ec4f4e6d5374b39ed0aca410766f9d
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dbd088d890f712c7b9381666b8fafd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype='float32').reshape([10]),
            ]


    class TestPrimitiveOp_15b19d5d7ce1165d8ba35ffe81eac27c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50be9c14adb924915c0af546c3e4ba22
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dtype='float32').reshape([10]),
            ]


    class TestPrimitiveOp_b1b98fc96dade5bf4974261a2f6faf91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32').reshape([7]),
            ]


    class TestPrimitiveOp_ab4ae4c3c53ffda8a50a637de6fb6730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50be9c14adb924915c0af546c3e4ba22
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype='float32').reshape([7]),
            ]


    
    class PrimitiveOp_6f45a3e75a9ce42bd28b72547edb8c10(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, -512, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e84dc5ed7e6715d863a884bbfbd6e8af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f45a3e75a9ce42bd28b72547edb8c10
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e84dc5ed7e6715d863a884bbfbd6e8af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f45a3e75a9ce42bd28b72547edb8c10
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a2ddd30c9a41d4a32f7195a1e00d61e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            return paddle._C_ops.scale(input_0, input_1, 512, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ccca7b4d79ab1d845cd0c746425ece3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2ddd30c9a41d4a32f7195a1e00d61e4
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ccca7b4d79ab1d845cd0c746425ece3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2ddd30c9a41d4a32f7195a1e00d61e4
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bc688134e6b54ee41a121c83bc9ee63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f23d5b9380e6a9fb4fb548a0a8ab2383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bd89d1bbb1a79848632b575058e1d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5dacec1b471d6439a0fbdb3781f3b915
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_fb06fa17a10eee898b05dbe9d448c2cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b250b98dbe58a815e81d5e755ee0ffd5
        def get_inputs(self):
            return [
                paddle.to_tensor([2.201258659362793, 2.0960209369659424, 2.029393196105957, 2.0144710540771484, 2.0976173877716064, 2.191070795059204, 2.0642569065093994, 2.0307860374450684, 1.978666067123413, 2.211827278137207, 2.1965835094451904, 2.234017848968506, 2.076364517211914, 1.9699567556381226, 2.1472253799438477, 2.0376336574554443, 2.170506477355957, 2.3005428314208984, 2.0516340732574463, 2.1196653842926025], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_12200b463cebc0a6180462a16517f299(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549818d247c2f9764c426cc976066460
        def get_inputs(self):
            return [
                paddle.to_tensor(2.345494508743286, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_8653240f4ec65ba2b9a4a963de5eb813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42c240e1277eda4d15fb3b457a53b10b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3adb9c84ea59fb1c34c9d60391b9ed73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(354.8050537109375, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_6c4404539773f63fc9209fa46674fc8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba6edc4744bb3484acc4729e9c5f0cf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.043187495321035385], [-0.013158541172742844], [-0.05806963890790939], [0.015963705256581306]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_cf1d4f6f1c2cd1c62dd9f69a42d8c907(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04242425411939621], [-0.01673356629908085], [0.09204878658056259], [0.010856859385967255]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_84c87e5f3f2b98691ce232ba83d8885f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.017990680411458015], [-0.21364393830299377], [-1.630857229232788], [0.47037965059280396]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9af866f19ed00626d6f8b105a3f7a999(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84155f50e6ea448825382c489dc3908c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.9820092916488647], [1.2136439085006714], [2.630857229232788], [0.529620349407196]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_b192894977e7d37e0ec34e2232b66f8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4b7db55a3cdcf0524363c2a62101509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(30.832040786743164, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_c53ac447c7f80ee9af4b7c03255343f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb56769145003c56028d1f31188b4c2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5af5f2e5b8f3863931463ffdadd02cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1aeef020f34d701f6589063868529179
        def get_inputs(self):
            return [
                paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0203db6e1737e6cae421419d54411b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0203db6e1737e6cae421419d54411b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3948483599c222e82ce6094059e9dbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdf4346677de1d5e1e5a5fe84a1b8b0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3921c938c13cf8e54ac140649b8485d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_cef30e9b234f477953d674eedaaac853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9ddab8464dff930b0aefe0e09fe4a66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_a9ddab8464dff930b0aefe0e09fe4a66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_bb15c0932d69cb2ffac934fdd144b42f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59728ff8276fb61a951f9c6a0d5f90f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a2549fa0b9eff44dd33383614b52372(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96f4c370ae8b384491c3ca938e3c1eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75097a5449595821e7a28b96e0be6852(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6275f4d6aff38a173efbdd2d27a8b6f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 32.0
            return paddle._C_ops.scale(input_0, input_1, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_413c6dbf29e0fc7803e65373f4db03da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6275f4d6aff38a173efbdd2d27a8b6f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2431dbe395f49169af15ffa466b9892e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c56a0f0e00e9f7063bd373353711e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_413c6dbf29e0fc7803e65373f4db03da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6275f4d6aff38a173efbdd2d27a8b6f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94991f3d53fdd35a62c58cc590cee095(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3662df24dd60c5a843b0f96bc4582e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_426522a94bb1f85d99fe7f5d85707121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a54e2ad8065fc81dcbf8747e2d9c2c0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_360201fb317246c222df122f73295986(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79d930dd97f6ff1d0d9dd26f85f59c43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed28f64efa941e3458c4bc5408f297cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a81275ed35c9ad5025d12b163488bb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_fcc051ec722743cf4b0e35ed1dc46e3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_302eeaab6c1b5d9faaeeea953d5f7ae2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb2c50ac89a6ee6c947abb082e3069a
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f66eec8dbd24b06fa78b4caa9a63b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_360201fb317246c222df122f73295986(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9b8a43f490f9e2c35b7a45a652c94ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed28f64efa941e3458c4bc5408f297cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae27f658f588a907f038dce0f644d94c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_42e02bae011d98d818c69b55785d0136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_264ab44f1344f96f06a07c5d83c384ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(164.5021514892578, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_ae1e9f4d4a61d7996d5ee1796562f5fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(3.465967893600464, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_20f9c9f87e5c615e695ae7294dd79440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f3f5fdbd8eeb5dc0145c712c2badc41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed60d62343112eb2efb74a054225b8de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba1a0811b23147ad87fcd4b3db98a422(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cbd3788e1187871219ada1037342936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3571fcdba8978db607871d8a688fdcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2145b69cf36e3309d25b6657d94e8596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf2ccb7be35f2a80021049ed92504f8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fc97965d2e7774e342b97459c88ac9d
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34880938add65f0ef6154becaed8dcab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d24f548e77cb09b40134c66cc350d7cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea5fd51b8c2e3bf57cfa6498b0572286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d669d76cec7b739c0efb639b3b81be1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0608e99c9dab8fc403181b911eee47d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4db763415ecfa8623fe051b12dad7975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(38.16582107543945, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_bca965141b2dcb3b17d40b1c6d752842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3f46c7ab9c186e947f04431b129ac53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd4103f195e68ac2b9fb64dafacdef8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ba4586b3c12fffb7a864424dca27244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ba4586b3c12fffb7a864424dca27244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58703de55d911fbe65dcb3e869ebb62
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e023c9c6b4cea9a1b05a2d258ecac6d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_121d139edbda1fc07e31a6f9c0f76fee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08dc8c6c1c45a0a32b6541872be033fd
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a7f6590ef404c023aca16f004d6c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f788670d621586c3d6abce5969f7b2f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_8f044117d9fbd370362dfd020c84ce78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff3f833396ee0db603309ceb623228d1
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de1d7dfad5a72dfe38ee52be93d4d633(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_de1d7dfad5a72dfe38ee52be93d4d633(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118f1751df7236dce37fa248afa33c48
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_5d1f1e2400dfafcb92b917fbf9fd47f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fab70d519bea87039bb00f6bcc7c1b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72f8d93ffb684ed068f0414570852bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61b2a52efc794f00a41cad3c2a06ba1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(173.64675903320312, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_316e9a13b3e9939c018a7e2aa2d9c589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b1fc466c2af308517a079d1ea7ccd1c
        def get_inputs(self):
            return [
                paddle.to_tensor(4.998120307922363, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb83ded5bf8f7c9988e98ecb44bcbe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaf81160e14e881614843876343edf1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b551684bc8e1a3d9ff5c03df09f4cdcd
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a86315f5c5f9104d09034d519cab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc27c099af8f208ba0d0e22d87b6ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fd77f0933add7acf755bd1efab56cb
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feeef1e2c275018d23cc3ba9ac1354eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d104d1f616998ace0b1eb42eff41aefa
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f9dae8230f056b85663e538f7f9c0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e9e6c3ec57599ea6dc42df4bb85b4d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f56690ae6546c96504981e8459a57f25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb115d4a8e81cf4b2bd797964278365e
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bca965141b2dcb3b17d40b1c6d752842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5fce1dd5eaedee391b2bff6285df99b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_703ae14165c6999db8e37114c95af7f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39f4378360e1bbcd0b5c24e7f4f62b4e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b48072bcf4c7541b901202c5c4da5f72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a9a4fc92eb6ea9c02a094a15e2b421
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_252994c8ea087eb78a65e9c2bd1cf6ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75850f31c7f5e0bfe1222b69cc04355f
        def get_inputs(self):
            return [
                paddle.to_tensor(37.80712890625, dtype='float32').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()