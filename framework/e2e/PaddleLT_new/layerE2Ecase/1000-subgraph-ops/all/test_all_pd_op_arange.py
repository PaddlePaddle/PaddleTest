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
    class PrimitiveOp_977309d5be18721e1920dcfe38a0357e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_234c992549701ea9f591485729aee06e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_05207c4cfe177ee99e20cd1173f0918a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([96.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3cc9093997ee3817361efab07caa938c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([48.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ab84aff99d9e1fd9b51f5c7eb7ee57ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a0b9311fa21b52b4242a0fcc4374e385(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([64.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_efaf3d93dfb6ff2f681be77a0e7c8c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ece0e0e37e13364b1a5e87c1208ddf83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ff095dbe016c221aa19a0171dc4347b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f8757447d201d7087fc1c8239cf783a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([40.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_234c992549701ea9f591485729aee06e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ff095dbe016c221aa19a0171dc4347b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e62ae3fdd08b53f1d5eebed185ed6d58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([14.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8488629310bf78badb8ebf84ce768ccd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([28.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ee0436977450b3724afaec2db6fae798(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([56.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_234c992549701ea9f591485729aee06e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_e908020192d464f739147dbdbbb3730b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0729f97ae1281d969d46515d854475e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e908020192d464f739147dbdbbb3730b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_405cb00c2964513d03ef571e70cf8ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([68.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b1320c485e260133f5e8c22a80e812fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([34.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_763a0fd90ecad552264f1555b0a87192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([17.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b6ac7d1ce8c7ffda39e3dc16da42031b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b6ac7d1ce8c7ffda39e3dc16da42031b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cc9a2638009d5b650aa2797df9bf992a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cc9a2638009d5b650aa2797df9bf992a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_22e7b3b9a1f84c45b23f97943ad9555d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([152.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0ac74e260736a8cc8883db3bdaa97781(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4b572d73efc4649a7a7804979df49cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([76.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4df8e4d684273dcc3105638226ffd5d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([50.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6dbc5f3c35d010353dc941353b9c1b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([38.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5f37cb72efffcba3e0511b026a18ff59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2d7b9e51aa0ef604381f52c92fb2ccf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([19.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_efd35d30bc06f469e29e5bb47c184a70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([13.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_be6b32b0253f8a2745753a905f492c91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([10.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f228d0844624f08d8194bb415a0a43f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a81e6cc8021aaa7106a5622413e187f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([72.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_38d42284fd71c4bb2209c5f2a72e5de0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([36.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2fad3a45852e5e48ddf3ac8d2a6c14d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([18.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_234c992549701ea9f591485729aee06e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_05207c4cfe177ee99e20cd1173f0918a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([96.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3cc9093997ee3817361efab07caa938c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([48.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ab84aff99d9e1fd9b51f5c7eb7ee57ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a0b9311fa21b52b4242a0fcc4374e385(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([64.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_efaf3d93dfb6ff2f681be77a0e7c8c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ece0e0e37e13364b1a5e87c1208ddf83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ff095dbe016c221aa19a0171dc4347b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f8757447d201d7087fc1c8239cf783a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([40.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_234c992549701ea9f591485729aee06e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ff095dbe016c221aa19a0171dc4347b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e62ae3fdd08b53f1d5eebed185ed6d58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([14.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8488629310bf78badb8ebf84ce768ccd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([28.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ee0436977450b3724afaec2db6fae798(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([56.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_234c992549701ea9f591485729aee06e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0729f97ae1281d969d46515d854475e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e908020192d464f739147dbdbbb3730b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_405cb00c2964513d03ef571e70cf8ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([68.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b1320c485e260133f5e8c22a80e812fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([34.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_763a0fd90ecad552264f1555b0a87192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([17.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b6ac7d1ce8c7ffda39e3dc16da42031b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b6ac7d1ce8c7ffda39e3dc16da42031b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cc9a2638009d5b650aa2797df9bf992a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cc9a2638009d5b650aa2797df9bf992a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_22e7b3b9a1f84c45b23f97943ad9555d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([152.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0ac74e260736a8cc8883db3bdaa97781(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4b572d73efc4649a7a7804979df49cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([76.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4df8e4d684273dcc3105638226ffd5d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([50.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e6dbc5f3c35d010353dc941353b9c1b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([38.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5f37cb72efffcba3e0511b026a18ff59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2d7b9e51aa0ef604381f52c92fb2ccf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([19.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_efd35d30bc06f469e29e5bb47c184a70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([13.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_be6b32b0253f8a2745753a905f492c91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([10.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f228d0844624f08d8194bb415a0a43f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a81e6cc8021aaa7106a5622413e187f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([72.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_38d42284fd71c4bb2209c5f2a72e5de0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([36.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2fad3a45852e5e48ddf3ac8d2a6c14d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bee07dd7a25c4f5902e9b5a1ef23867
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([18.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc368b307b307252f82900917799950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e0ba4fe6f0c6c75d59a397437d32a087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ece6880284f79a72543eeb2bf284f2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_977309d5be18721e1920dcfe38a0357e
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f7398275681846fb56fb212c26a39603(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_2c7ad0170177f76414102074d583d97d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d6164eb531cef977fbb4b07b2cfb18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b8cc4e124a81800c5d8ee988fea73f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([96.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d6c799c5a5c20c58bc1843105b0cbfea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([48.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d17f441a74b8a6305977496b9bcc9e32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_95481c7a22fd771bca2c337feb6e7a91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([64.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d3b4a98c8eba41b73e17ae62e70ab2cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2018566bc84ec96753e4da850be6d496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_20d25ab635271017240f4cb14f777405(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8ea796a83c431c054fc995df530a4635(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([40.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8d6164eb531cef977fbb4b07b2cfb18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_20d25ab635271017240f4cb14f777405(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cad6fea420656480ab237bb1bc485331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([14.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2f7a21e849e185ac2fba5738a3f50376(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([28.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_aa408b7020aa755e46c97afbfd3f779e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([56.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8d6164eb531cef977fbb4b07b2cfb18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_7923d303f3adf68bb749cf9065894057(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle.arange(input_0, input_1, input_2, dtype=paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0742bcc182e0ad0b114121c02fdcd315(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7923d303f3adf68bb749cf9065894057
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([24.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ddd73c6294b60d158b2b1c1ffd2251e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([68.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ac2e0746fe87badf47d04dbfaf5dd159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([34.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_91431b5fdde62e10d615ee52a99b0610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([17.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_07fc885d5a37e3a921572bca61bab9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_07fc885d5a37e3a921572bca61bab9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_66665031f2477d55be3a2c7840cdac14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_66665031f2477d55be3a2c7840cdac14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8d87352ddaa7a76eb87950e4f492388e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([152.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a134ceb9837c1bdd275337c73e3b7591(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_93994bbd7294be068bcbf06b371da40f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([76.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b94dcf9f39451aa981dfdaab47269d50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([50.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1de48d2873c24b28fac5f08d7590d863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([38.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_16eb023128d40ba3be33c4b4fe3d7bb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([25.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_743baf3d33c75a655e3dde773bcf55a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([19.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7de491b6fb5c125874775435ffa7d4b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([13.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6e80cebda7db226bac5ef999771a503d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([10.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e7933ff7c7cc4aa983ad59a2332582e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([7.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a2f80162b1c0d52c3dc4a32629e27b37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([72.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_de86f463acd21459aadcd5e498e585a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([36.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_10e7cc54d23b8d254c15047f14350377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7ad0170177f76414102074d583d97d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
                paddle.to_tensor([18.0], dtype='float32').reshape([1]),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_52d0d56401b6be8855eb9ca70c802089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d93ea09c2bb29a84bd566d4d154c4cec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_23914353964570f4cbd1254e22a1d43b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7398275681846fb56fb212c26a39603
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()