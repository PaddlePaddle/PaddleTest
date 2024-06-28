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
    class PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df7df7c82e2ccd6dca183966fcbafdb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7df7c82e2ccd6dca183966fcbafdb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87d730245bc6852ad27eef26b8b9d22f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87d730245bc6852ad27eef26b8b9d22f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f8b9ac60994e045bef131d95095c7ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f8b9ac60994e045bef131d95095c7ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4875fc8ed7301c2c4dd09e4c7c866ba8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4875fc8ed7301c2c4dd09e4c7c866ba8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c91de78665ede121696e935759e9027e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c91de78665ede121696e935759e9027e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d4d06774b9807610dfcf7b818b6b81a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d4d06774b9807610dfcf7b818b6b81a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64592819c16b0baa05b6b1b0b6a1f409
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b7996fc6c8aedf0fefaf427dd2fa1bea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 36, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_682b1035de7d7395eecc42d3bbfd7307(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7996fc6c8aedf0fefaf427dd2fa1bea
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_682b1035de7d7395eecc42d3bbfd7307(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7996fc6c8aedf0fefaf427dd2fa1bea
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f74434c5e76411c8b9e0cd044ecb189a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 72, 32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1252a7c7b90ce0efe884eee640e65ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f74434c5e76411c8b9e0cd044ecb189a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1252a7c7b90ce0efe884eee640e65ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f74434c5e76411c8b9e0cd044ecb189a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ae713382ab832408cf8e1451a89d226(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 18, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81f7de60ef88d4d3762c4d089cacbc42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ae713382ab832408cf8e1451a89d226
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81f7de60ef88d4d3762c4d089cacbc42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ae713382ab832408cf8e1451a89d226
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0ba5231c46cade47ec7642da9b9eee8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 192, 32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6164782dbdf68a4710bc0cebd88cd339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ba5231c46cade47ec7642da9b9eee8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6164782dbdf68a4710bc0cebd88cd339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ba5231c46cade47ec7642da9b9eee8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4403590164af7507c4d53835919737c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 48, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9c7ca08c1d11465c7ca49a707c13df0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4403590164af7507c4d53835919737c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9c7ca08c1d11465c7ca49a707c13df0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4403590164af7507c4d53835919737c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3cd67d8cddcbc85bf010c5824578fc3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 96, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_daa76cb5be351ff7dc985e87045920fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3cd67d8cddcbc85bf010c5824578fc3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daa76cb5be351ff7dc985e87045920fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3cd67d8cddcbc85bf010c5824578fc3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_38e78216f53fd08892fcce3fddfa965f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.pool3d(input_0, [2, 1, 1], [2, 1, 1], [0, 0, 0], False, True, 'NCDHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b60da52eb95a07f7c085023c20d2c431(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b60da52eb95a07f7c085023c20d2c431(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99753c1a0675b8e0213c3c6e9e118867(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99753c1a0675b8e0213c3c6e9e118867(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8656a32ff9385eddf806b0358d64f65b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8656a32ff9385eddf806b0358d64f65b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eff7914f40638d7c3cd280482ee8b3a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eff7914f40638d7c3cd280482ee8b3a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17be20bf9f85e18936e80976bcd5d741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17be20bf9f85e18936e80976bcd5d741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9a0ba833d2233bf670004b7e3e3d23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9a0ba833d2233bf670004b7e3e3d23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38e78216f53fd08892fcce3fddfa965f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()